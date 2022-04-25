"""A suite of cardinality estimators.

In practicular, inference algorithms for autoregressive density estimators can
be found in 'ProgressiveSampling'.
"""
import time
import re
from typing import Dict, Union

import numpy as np
import torch
import pytorch_lightning as pl

import cardest.common


OPS = {
    ">": np.greater,
    "<": np.less,
    ">=": np.greater_equal,
    "<=": np.less_equal,
    "=": np.equal,
    "!=": lambda all_vals, x: ~np.equal(all_vals, x),
    "<>": lambda all_vals, x: ~np.equal(all_vals, x),
    "IN": np.isin,
    "BETWEEN": lambda all_vals, lb_ub: np.logical_and(
        np.greater_equal(all_vals, lb_ub[0]), np.less_equal(all_vals, lb_ub[1])
    ),
    "IS": lambda all_vals, x: np.full_like(all_vals, False, dtype=np.bool)
    if x is None
    else np.equal(all_vals, x),
    "IS NOT": lambda all_vals, x: np.full_like(all_vals, True, dtype=np.bool)
    if x is None
    else ~np.equal(all_vals, x),
    #  pooooor performance
    "LIKE": lambda all_vals, x: np.array(
        [
            re.fullmatch(re.escape(x).replace("_", ".").replace("%", ".*"), val)
            is not None
            for val in all_vals
        ]
    ),
    "NOT LIKE": lambda all_vals, x: np.array(
        [
            re.fullmatch(re.escape(x).replace("_", ".").replace("%", ".*"), val) is None
            for val in all_vals
        ]
    ),
}


class CardEst(object):
    """Base class for a cardinality estimator."""

    def __init__(self):
        self.query_starts = []
        self.elapsed_time_ms = []
        self.errs = []
        self.est_cards = []
        self.true_cards = []
        self.est_card_cv = []

        self.name = "CardEst"

    def query(self, query):
        """Estimates cardinality with the specified conditions.

        Args:
            columns: list of Column objects to filter on.
            operators: list of string representing what operation to perform on
              respective columns; e.g., ['<', '>='].
            vals: list of raw values to filter columns on; e.g., [50, 100000].
              These are not bin IDs.
        Returns:
            Predicted cardinality.
        """
        raise NotImplementedError

    def on_start(self):
        self.query_starts.append(time.time())

    def on_end(self):
        elapsed_time_ms = (time.time() - self.query_starts[-1]) * 1e3
        self.elapsed_time_ms.append(elapsed_time_ms)
        return elapsed_time_ms

    def add_err(self, err, est_card, true_card):
        self.errs.append(err)
        self.est_cards.append(est_card)
        self.true_cards.append(true_card)

    def __str__(self):
        return self.name

    def get_stats(self):
        return [
            self.query_starts,
            self.elapsed_time_ms,
            self.errs,
            self.est_cards,
            self.true_cards,
        ]

    def merge_stats(self, state):
        self.query_starts.extend(state[0])
        self.elapsed_time_ms.extend(state[1])
        self.errs.extend(state[2])
        self.est_cards.extend(state[3])
        self.true_cards.extend(state[4])

    def report(self):
        est = self
        print(
            est.name,
            "max",
            np.max(est.errs),
            "99th",
            np.quantile(est.errs, 0.99),
            "95th",
            np.quantile(est.errs, 0.95),
            "median",
            np.quantile(est.errs, 0.5),
            "time_ms",
            np.mean(est.elapsed_time_ms),
        )


class ProgressiveSampling(CardEst):
    def __init__(
        self,
        model: pl.LightningModule,
        db: cardest.common.DB,
        params_dict: Dict[str, Union[str, int, Dict[str, Union[str, int]]]],
    ):
        super().__init__()

        assert model.n_cols == db.vtable.n_scols

        params = list(params_dict.values())[0]

        torch.set_grad_enabled(False)
        self.model = model
        self.db = db

        self.name = "ProgressiveSamplingNARUR"
        self.sample_size = params["static"]["eval_sample_size"]
        self.n_full_rows = self.db.vtable.n_rows
        self.device = params["static"]["device"]

        with torch.no_grad():
            self.init_logits = self.model(
                torch.zeros(
                    (
                        1,
                        self.model.embs.sum_of_dims,
                    ),
                    device=self.device,
                )
            )

        for p in model.parameters():
            p.detach_()
            p.requires_grad = False
        self.init_logits.detach()

        with torch.no_grad():
            self.k_zeros = torch.zeros(
                self.sample_size, self.model.n_cols, device=self.device
            )
            self.inp = self.model.embs.encode_all_wo_unk(self.k_zeros)

            # For transformer, need to flatten [num cols, d_model].
            self.inp = self.inp.view(self.sample_size, -1)

    def __str__(self):
        return f"psample_{self.sample_size}"

    def _sample_n(
        self,
        n_samples,
        query,
        query_i=None,
        inp=None,
    ):
        n_scols = self.db.vtable.n_scols
        logits = self.init_logits.repeat((n_samples, 1))
        if inp is None:
            inp = self.inp[:n_samples]

        cols, ops, vals, tbls = query

        # tmp
        _all_tbls = {
            c.tbl_existence(): i
            for i, c in enumerate(self.db.vtable.cols)
            if c.is_tbl_existence()
        }
        _used_tbls = {t: _all_tbls[t] for t in tbls if t in _all_tbls}
        _not_used_tbls = dict(_all_tbls.items() - _used_tbls.items())

        col_to_scol_idx = [c.scol_idxes for c in self.db.vtable.cols]

        # Use the query to filter each column's domain.
        # None means all valid.
        sdomains = [None] * n_scols  # [#scols, [#categories]]
        # based on given queries
        for col_name, op, val in zip(cols, ops, vals):
            col = self.db.vtable[col_name]
            col_idx = self.db.vtable.col_name_to_idx(col_name)
            domain = OPS[op](self.db.vtable.cols[col_idx].distinct_vals, val).astype(
                np.float32, copy=False
            )
            domain = np.append(
                [
                    0.0,
                    # ^ complement mask token
                    1.0
                    if (op == "IS" and val is None)
                    or (op == "IS NOT" and val is not None)
                    else 0.0,
                ],
                domain,
            )

            if not col.is_factorized:
                scol_idx = col_to_scol_idx[col_idx][0]
                if sdomains[scol_idx] is None:
                    sdomains[scol_idx] = torch.as_tensor(domain, device=self.device)
                else:
                    # multiple predicates on a col
                    domain = torch.as_tensor(domain, device=self.device)
                    sdomains[scol_idx] = torch.where(
                        sdomains[scol_idx] == 1,
                        domain,
                        sdomains[scol_idx],
                    )  # intersection
            else:
                """
                pred is `x in (nyan, nya, meow)`
                dv=[ MASK,  NULL, nyan,   wan,  nya,   bow, meow,   wow,  grrr]
                di=[    0,     1,    2,     3,    4,     5,    6,     7,    8]
                do=[False, False, True, False, True, False, True, False, False]
                fv=[    0,     1,    2,     3,    0,     1,    2,     3,    0]
                gv=[    0,     0,    0,     0,    1,     1,    1,     1,    2]
                fd=[0, 1, 2, 3]
                gd=[0, 1, 2]

                tmp order: gv -> fv
                if gv is 0: fv in [2]
                if gv is 1: fv in [0, 2]
                if gv is 2: fv in []
                e.g.,
                    0,0 -> x
                    0,2 -> o
                    1,0 -> o
                    2,0 -> x

                f=[2,1,0]
                g=[0,1,1] # label
                x={(0,): [2], (1,): [0,2], (2,): []} <- ほしい
                x=[[0,0,1], [1,0,1], [0,0,0]] # [prev_dist, curr_dist] <- これでもいい
                """

                domain = torch.as_tensor(domain, device=self.device)
                domain = domain.bool()

                for scol in col.scols[::-1]:
                    # idx of vtable.scols
                    scol_idx = col_to_scol_idx[col_idx][scol.scol_idx]

                    sdomain = torch.zeros(
                        (self.db.vtable.scols[scol_idx].dist_size,),
                        dtype=torch.float,
                        device=self.device,
                    )
                    # project col's domain (domain) into scol's domain (sdomain)
                    sdomain[scol.factorized_orig_distinct_vals[domain].unique()] = 1

                    if sdomains[scol_idx] is None:
                        # single or first predicate on a col
                        sdomains[scol_idx] = sdomain
                    else:
                        # multiple predicates on a col
                        sdomains[scol_idx] = torch.where(
                            sdomains[scol_idx] == 1,
                            sdomain,
                            sdomains[scol_idx],
                        )  # intersection

        # based on used tables
        # table name columns represent the table exists
        for tbl, col_idx in _used_tbls.items():
            scol_idx = col_to_scol_idx[col_idx][0]
            assert sdomains[scol_idx] is None
            sdomains[scol_idx] = torch.as_tensor(
                np.append(
                    [0.0, 0.0],
                    self.db.vtable.cols[col_idx].distinct_vals.astype(np.float),
                ),
                device=self.device,
            )

        conditioned_col_idxes = [
            self.db.vtable.col_name_to_idx(col_name) for col_name in cols
        ]
        conditioned_col_idxes = sorted(
            set(conditioned_col_idxes), key=conditioned_col_idxes.index
        )  # unique
        conditioned_col_idxes = list(
            dict(
                sorted(
                    zip(
                        conditioned_col_idxes,
                        [
                            sdomains[col_to_scol_idx[i][-1]].sum().item()
                            for i in conditioned_col_idxes
                        ],
                    ),
                    key=lambda x: x[1],
                )
            ).keys()
        )  # sort by the num of el satisfying predicates (proposal)
        conditioned_col_idxes.extend(_used_tbls.values())

        global_inp = torch.empty(
            (n_samples, self.model.embs.sum_of_dims), device=self.device
        )
        self.model.embs.encode_all(
            torch.zeros((n_samples, n_scols), dtype=torch.long, device=self.device),
            global_inp,
        )
        if self.model.embs.cont_fanout:
            global_samples = torch.zeros(
                (n_samples, n_scols), dtype=torch.float, device=self.device
            )
        else:
            global_samples = torch.zeros(
                (n_samples, n_scols), dtype=torch.long, device=self.device
            )
        prob = torch.ones(
            n_samples, dtype=torch.float, device=self.device
        )  # to be aggregated
        n_iters = len(conditioned_col_idxes)
        if n_iters == 0:
            return 1.0
        for it, col_idx in enumerate(conditioned_col_idxes):
            col = self.db.vtable.cols[col_idx]

            for scol_idx in col.scol_idxes[::-1]:
                dist = torch.softmax(
                    self.model.embs.decode_as_logit(logits, scol_idx),
                    1,
                )

                # apply predicates
                dist *= sdomains[scol_idx]

                # prevent sampler from sampling mask token
                if col.is_factorized and scol_idx == col.scol_idxes[0]:
                    # 1,...,1 represents original col's mask token
                    mask_cands = (global_samples[:, col.scol_idxes[1:]] == 1).all(dim=1)
                    dist[mask_cands, 1] = 0

                curr_pred_prob = dist.sum(dim=1)

                # dist was vanished by curr predicate
                # use uniform sampling
                vanished = (curr_pred_prob <= 0).view(-1, 1)
                dist.masked_fill_(vanished, 1.0)

                prob *= curr_pred_prob
                global_samples[:, scol_idx] = torch.multinomial(
                    dist, num_samples=1, replacement=True
                ).squeeze(1)
                self.model.embs.encode(global_samples, scol_idx, global_inp)
                logits = self.model.forward_w_encoded(global_inp)

            if it == n_iters - 1:
                it += 1  # for dump number
                scale = 1.0

                if len(tbls) > 1:
                    # FIXME: if joined by not listed keys, make wrong scaling
                    # (especially, not including title)
                    # multiple candidates to be joined (need explicit key name in fanout col)
                    if self.db.name == "imdb":
                        # `imdb` has all of tbls/cols to process job-l, job-l-r, and job-m
                        # JOB *happens to need* only (static) keys which are joined to `title` table
                        # If a benchmark requires dynamic keys, need to follow it
                        join_keys = {
                            "complete_cast": "movie_id",
                            "movie_companies": "movie_id",
                            "movie_info_idx": "movie_id",
                            "movie_keyword": "movie_id",
                            "movie_link": "movie_id",
                        }
                    elif self.db.name == "imdb-job-light":
                        # `imdb-job-light` has only tbls/cols to process job-l
                        # always joined w/ a specific key, so key name in fanout col is implicit
                        join_keys = {}
                    else:
                        raise NotImplementedError()

                    fanout_cols = [
                        f"__fanout__:{tbl}{('.'+join_keys[tbl]) if tbl in join_keys else ''}"
                        for tbl in _not_used_tbls.keys()
                    ]
                    for col in self.db.vtable.cols:
                        if col.name not in fanout_cols:
                            continue

                        restored_sample = torch.zeros((n_samples,), device=self.device)
                        for i, scol_idx in enumerate(col.scol_idxes[::-1]):
                            if self.model.embs.cont_fanout:
                                log2_fanout = self.model.embs.decode_as_raw_val(
                                    logits, scol_idx
                                )[:, 1]
                                global_samples[:, scol_idx] = log2_fanout
                                self.model.embs.encode(
                                    global_samples, scol_idx, global_inp
                                )
                                logits = self.model.forward_w_encoded(global_inp)
                                fanout = 2 ** log2_fanout
                            else:
                                dist = torch.softmax(
                                    self.model.embs.decode_as_logit(logits, scol_idx),
                                    1,
                                )

                                # Don't sample from mask token nor null token
                                # As a result, restored_sample must be larger than or equal to 1
                                # orig:            [nan, idx1, idx2,...]
                                # restored_sample: [1,   2,    3,...]
                                if col.is_factorized and i == len(col.scol_idxes) - 1:
                                    # prev scols' val are all 0
                                    # then, head scol is 0 represents mask token
                                    #       head scol is 1 represents nan
                                    mask_cands = (
                                        global_samples[:, col.scol_idxes[1:]].sum(dim=1)
                                        == 0
                                    )
                                    dist[mask_cands, 0] = 0
                                else:
                                    dist[:, :2] = 0

                                global_samples[:, scol_idx] = torch.multinomial(
                                    dist, num_samples=1, replacement=True
                                ).squeeze(1)
                                self.model.embs.encode(
                                    global_samples, scol_idx, global_inp
                                )
                                logits = self.model.forward_w_encoded(
                                    global_inp.view(n_samples, -1)
                                )
                                if i > 0:  # only when factorized
                                    restored_sample |= global_samples[
                                        :, scol_idx
                                    ] << col.fact_bits * (len(col.scol_idxes) - 1 - i)
                                else:
                                    restored_sample = global_samples[:, scol_idx]

                        if self.model.embs.cont_fanout:
                            scale *= fanout
                        else:
                            scale *= col.to_val[restored_sample]

                    prob /= scale.float()

                return prob  # selectivity

    def query(self, query, i=None):
        with torch.no_grad():
            inp_buf = self.inp.zero_()
            self.on_start()
            with torch.no_grad():
                sels = self._sample_n(
                    self.sample_size,
                    query,
                    i,
                    inp=inp_buf,
                )
                sel = sels.mean().item()
                card = sel * self.n_full_rows

            elapsed_time_ms = self.on_end()
            return (
                np.ceil(card).astype(dtype=np.int, copy=False).item(),
                elapsed_time_ms,
            )
