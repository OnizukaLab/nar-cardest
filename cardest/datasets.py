"""Dataset registrations."""
import os

import numpy as np
import pyarrow as pa

import cardest.common


dataset_dir = "datasets"


class Loader:
    def __init__(self, fact_threshold, fact_bits, device):
        self.fact_threshold = fact_threshold
        self.fact_bits = fact_bits
        self.device = device

    def load(self, dataset_name: str, relation_type: str) -> cardest.common.DB:
        if dataset_name == "dmv":
            return self.load_dmv()
        elif dataset_name == "dmv-tiny":
            return self.load_dmv("dmv-tiny.csv")
        elif dataset_name == "dmv-1":
            return self.load_dmv_1()
        elif dataset_name == "dmv-2":
            return self.load_dmv_2()
        elif dataset_name == "dmv-5":
            return self.load_dmv_5()
        elif dataset_name == "flight-delays":
            return self.load_flight_delays()
        elif dataset_name == "flight-delays-tiny":
            return self.load_flight_delays("flight-delays-tiny.csv")
        elif dataset_name == "imdb":
            if relation_type == "ur":
                return self.load_imdb()
            else:
                raise ValueError(f"Unexpected relation type: {relation_type}")
        elif dataset_name == "imdb-job-light":
            if relation_type == "ur":
                return self.load_imdb_job_light()
            else:
                raise ValueError(f"Unexpected relation type: {relation_type}")
        elif dataset_name == "imdb-tiny":
            if relation_type == "ur":
                return self.load_imdb("imdb-1m.csv")
            else:
                raise ValueError(f"Unexpected relation type: {relation_type}")
        else:
            raise ValueError(f"Unexpected dataset name: {dataset_name}")

    def load_dmv(self, file_name="dmv.csv"):
        csv_file = os.path.join(dataset_dir, file_name)
        col_names = [
            "Record Type",
            "Registration Class",
            "State",
            "County",
            "Body Type",
            "Fuel Type",
            "Reg Valid Date",
            "Color",
            "Scofflaw Indicator",
            "Suspension Indicator",
            "Revocation Indicator",
        ]
        # Note: other columns are converted to objects/strings automatically.  We
        # don't need to specify a type-cast for those because the desired order
        # there is the same as the default str-ordering (lexicographical).
        type_casts = {"Reg Valid Date": np.datetime64}
        table = cardest.common.CsvTable(
            "dmv-universal",
            csv_file,
            col_names,
            self.fact_threshold,
            self.fact_bits,
            self.device,
            type_casts=type_casts,
        )
        return cardest.common.DB(name="dmv", vtable=table, tables={})

    def load_dmv_1(self, file_name="dmv.csv"):
        csv_file = os.path.join(dataset_dir, file_name)
        col_names = [
            # "Record Type",
            "Registration Class",
            "State",
            "County",
            "Body Type",
            "Fuel Type",
            "Reg Valid Date",
            "Color",
            "Scofflaw Indicator",
            "Suspension Indicator",
            "Revocation Indicator",
        ]
        # Note: other columns are converted to objects/strings automatically.  We
        # don't need to specify a type-cast for those because the desired order
        # there is the same as the default str-ordering (lexicographical).
        type_casts = {"Reg Valid Date": np.datetime64}
        table = cardest.common.CsvTable(
            "dmv",
            csv_file,
            col_names,
            self.fact_threshold,
            self.fact_bits,
            self.device,
            type_casts,
        )
        return cardest.common.DB(name="dmv", vtable=table, tables={})

    def load_dmv_2(self, file_name="dmv.csv"):
        csv_file = os.path.join(dataset_dir, file_name)
        col_names = [
            # "Record Type",
            "Registration Class",
            "State",
            # "County",
            "Body Type",
            "Fuel Type",
            "Reg Valid Date",
            "Color",
            "Scofflaw Indicator",
            "Suspension Indicator",
            "Revocation Indicator",
        ]
        # Note: other columns are converted to objects/strings automatically.  We
        # don't need to specify a type-cast for those because the desired order
        # there is the same as the default str-ordering (lexicographical).
        type_casts = {"Reg Valid Date": np.datetime64}
        table = cardest.common.CsvTable(
            "dmv",
            csv_file,
            col_names,
            self.fact_threshold,
            self.fact_bits,
            self.device,
            type_casts,
        )
        return cardest.common.DB(name="dmv", vtable=table, tables={})

    def load_dmv_5(self, file_name="dmv.csv"):
        csv_file = os.path.join(dataset_dir, file_name)
        col_names = [
            # "Record Type",
            # "Registration Class",
            # "State",
            # "County",
            # "Body Type",
            "Fuel Type",
            "Reg Valid Date",
            "Color",
            "Scofflaw Indicator",
            "Suspension Indicator",
            "Revocation Indicator",
        ]
        # Note: other columns are converted to objects/strings automatically.  We
        # don't need to specify a type-cast for those because the desired order
        # there is the same as the default str-ordering (lexicographical).
        type_casts = {"Reg Valid Date": np.datetime64}
        table = cardest.common.CsvTable(
            "dmv",
            csv_file,
            col_names,
            self.fact_threshold,
            self.fact_bits,
            self.device,
            type_casts=type_casts,
        )
        return cardest.common.DB(name="dmv", vtable=table, tables={})

    def load_flight_delays(self, file_name="flight-delays.csv"):
        csv_file = os.path.join(dataset_dir, file_name)
        col_names = [
            "YEAR_DATE",
            "UNIQUE_CARRIER",
            "ORIGIN",
            "ORIGIN_STATE_ABR",
            "DEST",
            "DEST_STATE_ABR",
            "DEP_DELAY",
            "TAXI_OUT",
            "TAXI_IN",
            "ARR_DELAY",
            "AIR_TIME",
            "DISTANCE",
        ]
        type_casts = {"YEAR_DATE": pa.float32()}
        table = cardest.common.CsvTable(
            "flight_delays",
            csv_file,
            col_names,
            self.fact_threshold,
            self.fact_bits,
            self.device,
            type_casts,
        )
        return cardest.common.DB(
            name="FlightDelays", vtable=table, tables={table.name: table}
        )

    def load_imdb(self, file_name="imdb.csv"):
        csv_file = os.path.join(dataset_dir, file_name)

        db_col_names = {
            "kind_type": [
                "id",
                "kind",
            ],
            "title": [
                "id",
                "title",
                "imdb_index",
                "kind_id",
                "production_year",
                "phonetic_code",
                "season_nr",
                "episode_nr",
                "series_years",
            ],
            "movie_companies": [
                "movie_id",
                "company_id",
                "company_type_id",
                "note",
            ],
            "company_name": [
                "id",
                "name",
                "country_code",
            ],
            "company_type": [
                "id",
                "kind",
            ],
            "movie_info": [
                "movie_id",
                "info_type_id",
                "info",
                "note",
            ],
            "movie_info_idx": [
                "movie_id",
                "info_type_id",
                "info",
            ],
            "movie_keyword": [
                "movie_id",
                "keyword_id",
            ],
            "keyword": [
                "id",
                "keyword",
                "phonetic_code",
            ],
            "cast_info": [
                "person_id",
                "movie_id",
                "person_role_id",
                "note",
                "nr_order",
                "role_id",
            ],
            "movie_link": [
                "movie_id",
                "linked_movie_id",
                "link_type_id",
            ],
            "link_type": [
                "id",
                "link",
            ],
            "aka_title": [
                "movie_id",
            ],
            "complete_cast": [
                # "id",
                "movie_id",
                "subject_id",
                "status_id",
            ],
            "comp_cast_type": [
                "id",
                "kind",
            ],
            "info_type": [
                "id",
                "info",
            ],
        }

        type_casts_list = {
            "movie_info": {"note": pa.string()},
            "movie_info_idx": {"info": pa.string()},
        }

        table_names = list(db_col_names.keys())
        tables = {}
        distinct_vals_dict = {}
        for table_name in table_names:
            path = os.path.join("datasets", "imdb", f"{table_name}.csv")
            table = cardest.common.CsvTable(
                table_name,
                path,
                db_col_names[table_name],
                self.fact_threshold,
                self.fact_bits,
                self.device,
                type_casts=type_casts_list.get(table_name, {}),
                hold_data=False,
            )
            tables[table_name] = table

            for col in table.cols:
                distinct_vals_dict[f"{table_name}.{col.name}"] = col.distinct_vals

        vtable = cardest.common.CsvTable(
            "imdb-universal",
            csv_file,
            None,  # col_names,
            self.fact_threshold,
            self.fact_bits,
            self.device,
            type_casts=dict(
                sum(
                    [
                        [(f"{t}.{c}", ty) for c, ty in cs.items()]
                        for t, cs in type_casts_list.items()
                    ],
                    [],
                )
            ),
            distinct_vals_dict=distinct_vals_dict,
            n_rows=11244784701195,
        )
        return cardest.common.DB(name="imdb", vtable=vtable, tables=tables)

    # Can process only JOB-light
    def load_imdb_job_light(self, file_name="imdb-job-light.csv"):
        csv_file = os.path.join(dataset_dir, file_name)

        db_col_names = {
            "title": [
                "id",
                # "title",
                # "imdb_index",
                "kind_id",
                "production_year",
                # "imdb_id",
                # "phonetic_code",
                # "episode_of_id",
                # "season_nr",
                # "episode_nr",
                # "series_years",
                # "md5sum",
            ],
            "movie_companies": [
                # "id",
                "movie_id",
                "company_id",
                "company_type_id",
                # "note",
            ],
            "movie_info": [
                # "id", #
                "movie_id",
                "info_type_id",
                # "info",
                # "note",
            ],
            "movie_info_idx": [
                # "id",
                "movie_id",
                "info_type_id",
                # "info",
                # "note",
            ],
            "movie_keyword": [
                # "id",
                "movie_id",
                "keyword_id",
            ],
            "cast_info": [
                # "id",
                # "person_id",
                "movie_id",
                # "person_role_id",
                # "note",
                # "nr_order",
                "role_id",
            ],
        }

        type_casts_list = {
            "movie_info": {"note": pa.string()},
            "movie_info_idx": {"info": pa.string()},
        }

        table_names = list(db_col_names.keys())
        tables = {}
        distinct_vals_dict = {}
        for table_name in table_names:
            path = os.path.join("datasets", "imdb", f"{table_name}.csv")
            table = cardest.common.CsvTable(
                table_name,
                path,
                db_col_names[table_name],
                self.fact_threshold,
                self.fact_bits,
                self.device,
                type_casts=type_casts_list.get(table_name, {}),
                hold_data=False,
            )
            tables[table_name] = table

            for col in table.cols:
                distinct_vals_dict[f"{table_name}.{col.name}"] = col.distinct_vals

        vtable = cardest.common.CsvTable(
            "imdb-job-light-universal",
            csv_file,
            None,  # col_names,
            self.fact_threshold,
            self.fact_bits,
            self.device,
            type_casts=dict(
                sum(
                    [
                        [(f"{t}.{c}", ty) for c, ty in cs.items()]
                        for t, cs in type_casts_list.items()
                    ],
                    [],
                )
            ),
            distinct_vals_dict=distinct_vals_dict,
            n_rows=2128877229383,
        )
        return cardest.common.DB(name="imdb-job-light", vtable=vtable, tables=tables)
