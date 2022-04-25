import os
import re
import math
import time
import random
from typing import Dict

import numpy as np
import pandas as pd
import pyarrow.csv
import torch
from torch.utils import data
import pytorch_lightning as pl


class Column(object):
    """A column.  Data is write-once, immutable-after.

    Typical usage:
      col = Column('Attr1').Fill(data, infer_dist=True)

    The passed-in 'data' is copied by reference.
    """

    def __init__(
        self,
        name,
        table_name,
        fact_threshold,
        fact_bits,
        device,
        distinct_vals=None,
        pg_name=None,
    ):
        self.name = name
        self.table_name = (
            table_name if "." not in name else re.sub(r"\..+$", "", self.name)
        )
        self.orig_name = re.sub(r"^.+?:", "", self.name)

        self.device = device

        # Data related fields.
        self.data = None
        self.size = None
        self.distinct_vals = distinct_vals
        self.dist_size = (
            len(self.distinct_vals) + 2 if self.distinct_vals is not None else None
        )
        self.to_val = None  # will be used for fanout scaling

        # factorization
        self.fact_threshold = fact_threshold
        self.fact_bits = fact_bits
        self.is_factorized = None
        self.orig_col = None
        self.scols = None
        self.scol_idx = None
        self.m = None  # scol bridge between prev and curr
        self.factorized_orig_distinct_vals = None

        # pg_name is the name of the corresponding column in Postgres.  This is
        # put here since, e.g., PG disallows whitespaces in names.
        self.pg_name = pg_name if pg_name else name

    # special column markers
    def is_fanout(self):
        return "_fanout__" in self.name or self.name.endswith("__weight__")

    def is_tbl_existence(self):
        return "__in__:" in self.name

    def tbl_existence(self):
        assert self.is_tbl_existence()
        return self.name[7:]  # e.g., __in__:title -> title

    def set_dist(self, vals):
        """This is all the values this column will ever see."""
        if self.distinct_vals is not None:
            print(f"skip set_dist on {self.name} ", end="")
            return

        # unique
        self.distinct_vals = vals.unique()
        # non-null
        if type(self.distinct_vals) == np.ndarray:
            self.distinct_vals = self.distinct_vals[~np.isnan(self.distinct_vals)]
        else:
            self.distinct_vals = self.distinct_vals.dropna()
            # to ndarray
            dtype = self.distinct_vals.dtype.name
            if dtype == "string":
                dtype = "object"
            self.distinct_vals = self.distinct_vals.to_numpy(dtype=dtype)
        self.distinct_vals.sort()
        self.dist_size = len(self.distinct_vals) + 2  # MASK and NULL

        assert type(self.distinct_vals) == np.ndarray

    def set_to_val(self):
        assert self.distinct_vals is not None
        assert self.is_fanout()
        self.to_val = torch.as_tensor(
            # [0]: mask never appears
            # [1]: nan never appears
            # [2]: first not nan val
            # [3]: second not nan val
            # [n]: n-th not nan val
            np.insert(self.distinct_vals, 0, [-1, -1]).astype(np.float),
            device=self.device,
        )

    def set_data(self, data):
        assert self.data is None
        self.data = data
        self.size = len(self.data)

    def factorize(self):
        self.is_factorized = self.dist_size > self.fact_threshold
        if not self.is_factorized:
            return [self]

        scols = []
        for i in range(math.ceil(self.dist_size.bit_length() / self.fact_bits)):
            scol = Column(
                f"{self.name}:{i}",
                self.table_name,
                self.fact_threshold,
                self.fact_bits,
                self.device,
            )
            mask = ~(~0 << self.fact_bits) << (i * self.fact_bits)
            factorized_orig_distinct_vals = (
                np.arange(self.dist_size) & mask
            ) >> i * self.fact_bits  # [i*bits:(i+1)*bits]
            factorized_orig_distinct_vals += 1  # 0 for MASK
            scol.distinct_vals = np.append(0, np.unique(factorized_orig_distinct_vals))
            scol.factorized_orig_distinct_vals = torch.as_tensor(
                factorized_orig_distinct_vals, device=self.device
            )
            assert len(scol.factorized_orig_distinct_vals) == self.dist_size
            assert (scol.factorized_orig_distinct_vals > 0).all()
            scol.dist_size = scol.distinct_vals.size
            scol.scol_idx = i
            scol.orig_col = self

            scols.append(scol)

        return scols

    def factorize_input(self, data):
        assert self.scols is not None

        factorized_data = []
        for i in range(len(self.scols)):
            mask = ~(~0 << self.fact_bits) << (i * self.fact_bits)
            factorized_datum = (
                data & mask
            ) >> i * self.fact_bits  # [i*bits:(i+1)*bits]
            factorized_data.append(factorized_datum)
        return factorized_data

    def discretize(self, release_data=False):
        """Transforms data values into integers using a Column's vocab.

        Returns:
            col_data: discretized version; an np.ndarray of type np.int32.
        """
        assert (
            self.scol_idx is None
        ), f"{self.name} is a factorized col. apply discretize to only orig col."

        assert self.data is not None

        print(f"{self.name}... ", end="")

        # pd.Categorical() does not allow categories be passed in an array
        # containing np.nan.  It makes it a special case to return code -1
        # for NaN values and values not in distinct_vals as well.
        cate_ids_list = []
        cate_ids = pd.Categorical(self.data, categories=self.distinct_vals).codes
        assert len(cate_ids) == len(self.data)
        assert (
            # not fanout
            "__fanout__:" not in self.name
            and "__adj_fanout__:" not in self.name
        ) or (
            # when fanout, must not contain null
            (cate_ids >= 0).all()
        )

        # Since nan/nat cate_id is supposed to be 1 but pandas returns -1,
        # just add 2 to everybody.
        # 0 represents <MASK>: never appears in the original dataset
        # 1 represents <NULL>: originally -1 from pd.Categorical()
        cate_ids = cate_ids + 2
        cate_ids = cate_ids.astype(np.int32, copy=False)
        assert (cate_ids >= 1).all()

        if release_data:
            del self.data

        # discretize scols
        if len(self.scols) > 1:
            for i, scol in enumerate(self.scols):
                mask = ~(~0 << self.fact_bits) << (i * self.fact_bits)
                factorized_data = (
                    cate_ids & mask
                ) >> i * self.fact_bits  # [i*bits:(i+1)*bits]
                scol_data = pd.Series(factorized_data)
                scate_ids = pd.Categorical(
                    scol_data, categories=scol.distinct_vals
                ).codes
                assert len(scate_ids) == len(scol_data)
                scate_ids = scate_ids.astype(np.int32, copy=False)
                scate_ids += 1  # 0 for MASK
                assert (
                    scate_ids > 0
                ).all()  # 0 (MASK) never appears in the original dataset
                cate_ids_list.append(scate_ids)
        else:
            # not factorized
            cate_ids_list.append(cate_ids)

        return cate_ids_list

    def __repr__(self):
        return f"Column({self.name}, dist_size={self.dist_size})"


class Table(object):
    def __init__(self, name, cols, scols):
        self.name = name
        self.n_rows = self._validate_cardinality(cols)
        self.cols = cols
        self.scols = scols
        self.n_cols = len(self.cols)
        self.n_scols = len(self.scols)

        self.name_to_idx = {c.name: i for i, c in enumerate(self.cols)}

    def __repr__(self):
        return f"{self.name}"

    def __getitem__(self, col_name):
        return self.cols[self.col_name_to_idx(col_name)]

    def _validate_cardinality(self, columns):
        """Checks that all the columns have same the number of rows."""
        cards = [c.size for c in columns]
        c = np.unique(cards)
        assert len(c) == 1, c
        return c[0]

    def col_name_to_idx(self, col_name):
        """Returns index of column with the specified name."""
        assert col_name in self.name_to_idx
        return self.name_to_idx[col_name]


class CsvTable(Table):
    def __init__(
        self,
        name,
        file_name_or_df,
        col_names,
        fact_threshold,
        fact_bits,
        device,
        *,
        type_casts={},
        distinct_vals_dict={},
        hold_data=True,
        n_rows=None,
        **kwargs,
    ):
        self.name = name
        self.hold_data = hold_data
        self.device = device

        if isinstance(file_name_or_df, str):
            self.data = self._load(file_name_or_df, col_names, type_casts, **kwargs)
        else:
            assert isinstance(file_name_or_df, pd.DataFrame)
            self.data = file_name_or_df

        self.cols = self._build_columns(
            col_names,
            type_casts,
            fact_threshold,
            fact_bits,
            distinct_vals_dict,
        )
        self.scols = self._factorize()

        if not self.hold_data:
            del self.data

        super(CsvTable, self).__init__(name, self.cols, self.scols)

        if n_rows is not None:
            self.n_rows = n_rows

    def _load(self, file_name, cols, type_casts, **kwargs):
        print(f"Loading {self.name}...", end=" ")
        s = time.time()
        if file_name.endswith(".pickle"):
            data = pd.read_pickle(file_name)
        else:

            def _report_and_skip(row):
                print(row)
                return "skip"

            # ad-hoc patch for supporting csv files
            # from http://homepages.cwi.nl/~boncz/job/imdb.tgz
            parse_options = (
                pyarrow.csv.ParseOptions(
                    escape_char="\\", invalid_row_handler=_report_and_skip
                )
                if not self.name.startswith("imdb")
                else None
            )

            args = {
                "include_columns": cols,
                "column_types": {
                    c: tc for c, tc in type_casts.items() if tc != np.datetime64
                },
            }
            data = pyarrow.csv.read_csv(
                file_name,
                parse_options=parse_options,
                convert_options=pyarrow.csv.ConvertOptions(**args),
                **kwargs,
            ).to_pandas(split_blocks=True, self_destruct=True, date_as_object=False)
            data = data.convert_dtypes()
        # Drop weight columns
        data = data.filter(regex="^((?!__weight__).)*$", axis="columns")
        print(f"done, took {(time.time() - s):.1f}s")
        return data

    def _build_columns(
        self,
        col_names,
        type_casts,
        fact_threshold,
        fact_bits,
        distinct_vals_dict={},
    ):
        """Example args:

            col_names = ['Model Year', 'Reg Valid Date', 'Reg Expiration Date']
            type_casts = {'Model Year': int}

        Returns: a list of Columns.
        """
        print(f"Parsing {self.name}... ", end=" ")
        s = time.time()
        for col_name, typ in type_casts.items():
            if col_name not in self.data:
                continue
            if typ == np.datetime64:
                # Both infer_datetime_format and cache are critical for perf.
                self.data[col_name] = pd.to_datetime(
                    self.data[col_name], infer_datetime_format=True, cache=True
                )
            elif typ == int:
                self.data[col_name] = self.data[col_name].astype(int)

        if col_names is None:
            col_names = self.data.columns
        cols = []
        for col_name in col_names:
            distinct_vals = distinct_vals_dict.get(col_name, None)
            col = Column(
                col_name,
                self.name,
                fact_threshold,
                fact_bits,
                self.device,
                distinct_vals,
            )

            if col.is_fanout():
                self.data[col_name].fillna(1, inplace=True)

            if self.hold_data:
                col.set_data(self.data[col_name])
            else:
                col.size = len(self.data[col_name])

            if not col.is_tbl_existence():  # normal columns
                col.set_dist(self.data[col_name])
            else:  # existence marker
                col.set_dist(pd.Series([False, True]))

            if col.is_fanout():
                col.set_to_val()
            cols.append(col)
        print(f"done, took {(time.time() - s):.1f}s")

        return cols

    def _factorize(self, discrete_tables=None):
        scols = []
        for col_idx, col in enumerate(self.cols):
            scol_idx_s = len(scols)
            _scols = col.factorize()
            scols.extend(_scols)
            scol_idx_e = len(scols)
            col.scols = _scols
            col.scol_idxes = list(range(scol_idx_s, scol_idx_e))  # increment
        return scols


class DB:
    def __init__(
        self,
        name: str,
        vtable: CsvTable,
        tables: Dict[str, CsvTable],
    ):
        self.name = name
        self.vtable = vtable  # joined virtual table
        self.tables = tables  # each table

    def __repr__(self):
        return f"{self.name}({self.tables})"


class DBDataset(data.Dataset):
    def __init__(self, db: DB, cont_fanout, *, table_name="", cache_dir_root="."):
        super(DBDataset, self).__init__()
        if db.vtable is not None:
            table = db.vtable
        else:
            table = db.tables[table_name]

        cache_dir_path = os.path.join(".cache", db.name, "discretized")
        os.makedirs(cache_dir_path, exist_ok=True)
        print(f"Discretizing {table.name}... ", end="")
        cache_file_path = os.path.join(cache_dir_path, f"{table_name}.pt")
        if os.path.exists(cache_file_path):
            print(f"found cache {table_name}")
            self.tuples = torch.load(cache_file_path)
            del table.data
            return

        s = time.time()
        # [n_full_rows, n_cols]
        discretized = []
        for col in table.cols:
            if not col.is_fanout() or not cont_fanout:
                discretized.extend(col.discretize(release_data=True))
            else:
                cont_vals = col.data.values.astype(int)
                assert np.all(cont_vals >= 1)
                log2_cont_vals = np.log2(cont_vals)  # log-transformed
                # add epsilon to make a diff from mask
                log2_cont_vals[log2_cont_vals == 0] += np.finfo(
                    log2_cont_vals.dtype
                ).eps
                assert np.all(log2_cont_vals > 0)
                discretized.extend([log2_cont_vals])
                del col.data  # hack for reducing memory usage

        del table.data  # hack for reducing memory usage

        # (take a long time, high memory usage)
        self.tuples = torch.as_tensor(np.stack(discretized, axis=1))

        #      factorized           >= original
        assert self.tuples.shape[1] >= len(table.cols)

        torch.save(self.tuples, cache_file_path)

        print(f"done, took {(time.time() - s):.1f}s")

    def size(self):
        return len(self.tuples)

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        return self.tuples[idx]


class DataModule(pl.LightningDataModule):
    def __init__(self, db: DB, *, cache_dir_root: str = ".", cont_fanout: bool = False):
        super().__init__()

        self.db: DB = db
        self.cont_fanout = cont_fanout
        self.cache_dir_root = cache_dir_root

    def prepare_data(self) -> None:
        dataset = DBDataset(
            self.db,
            self.cont_fanout,
            table_name=self.table_name,
            cache_dir_root=self.cache_dir_root,
        )

        val_dataset_size = min(int(len(dataset) / 10), 10000)
        tra_dataset_size = len(dataset) - val_dataset_size
        print(f"train size: {tra_dataset_size}, val size: {val_dataset_size}")

        self.tra_dataset, self.val_dataset = torch.utils.data.random_split(
            torch.utils.data.dataset.Subset(
                dataset,
                random.sample(
                    list(range(len(dataset))), tra_dataset_size + val_dataset_size
                ),
            ),
            [tra_dataset_size, val_dataset_size],
        )

    def set_table(self, table_name: str, batch_size: int):
        self.table_name = table_name
        self.batch_size = batch_size

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.tra_dataset, batch_size=self.batch_size)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)
