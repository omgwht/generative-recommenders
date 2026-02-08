# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

import abc
from collections import defaultdict
from datetime import datetime, timezone
import json
import logging
import os
import sys
import tarfile
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class DataProcessor:
    """
    This preprocessor does not remap item_ids. This is intended so that we can easily join other
    side-information based on item_ids later.
    """

    def __init__(
        self,
        prefix: str,
        expected_num_unique_items: Optional[int],
        expected_max_item_id: Optional[int],
    ) -> None:
        self._prefix: str = prefix
        self._expected_num_unique_items = expected_num_unique_items
        self._expected_max_item_id = expected_max_item_id

    @abc.abstractmethod
    def expected_num_unique_items(self) -> Optional[int]:
        return self._expected_num_unique_items

    @abc.abstractmethod
    def expected_max_item_id(self) -> Optional[int]:
        return self._expected_max_item_id

    @abc.abstractmethod
    def processed_item_csv(self) -> str:
        pass

    def output_format_csv(self) -> str:
        return f"tmp/{self._prefix}/sasrec_format.csv"

    def sasrec_format_csv_by_user_train(self) -> str:
        return f"tmp/{self._prefix}/sasrec_format_by_user_train.csv"

    def sasrec_format_csv_by_user_valid(self) -> str:
        return f"tmp/{self._prefix}/sasrec_format_by_user_valid.csv"

    def sasrec_format_csv_by_user_test(self) -> str:
        return f"tmp/{self._prefix}/sasrec_format_by_user_test.csv"

    def to_seq_data(
        self,
        ratings_data: pd.DataFrame,
        user_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if user_data is not None:
            ratings_data_transformed = ratings_data.join(
                user_data.set_index("user_id"), on="user_id"
            )
        else:
            ratings_data_transformed = ratings_data
        ratings_data_transformed.item_ids = ratings_data_transformed.item_ids.apply(
            lambda x: ",".join([str(v) for v in x])
        )
        ratings_data_transformed.ratings = ratings_data_transformed.ratings.apply(
            lambda x: ",".join([str(v) for v in x])
        )
        ratings_data_transformed.timestamps = ratings_data_transformed.timestamps.apply(
            lambda x: ",".join([str(v) for v in x])
        )
        ratings_data_transformed.rename(
            columns={
                "item_ids": "sequence_item_ids",
                "ratings": "sequence_ratings",
                "timestamps": "sequence_timestamps",
            },
            inplace=True,
        )
        return ratings_data_transformed

    def file_exists(self, name: str) -> bool:
        return os.path.isfile("%s/%s" % (os.getcwd(), name))


class MovielensSyntheticDataProcessor(DataProcessor):
    def __init__(
        self,
        prefix: str,
        expected_num_unique_items: Optional[int] = None,
        expected_max_item_id: Optional[int] = None,
    ) -> None:
        super().__init__(prefix, expected_num_unique_items, expected_max_item_id)

    def preprocess_rating(self) -> None:
        return


class MovielensDataProcessor(DataProcessor):
    def __init__(
        self,
        download_path: str,
        saved_name: str,
        prefix: str,
        convert_timestamp: bool,
        expected_num_unique_items: Optional[int] = None,
        expected_max_item_id: Optional[int] = None,
    ) -> None:
        super().__init__(prefix, expected_num_unique_items, expected_max_item_id)
        self._download_path = download_path
        self._saved_name = saved_name
        self._convert_timestamp: bool = convert_timestamp

    def download(self) -> None:
        if not self.file_exists(self._saved_name):
            urlretrieve(self._download_path, self._saved_name)
        if self._saved_name[-4:] == ".zip":
            ZipFile(self._saved_name, "r").extractall(path="tmp/")
        else:
            with tarfile.open(self._saved_name, "r:*") as tar_ref:
                tar_ref.extractall("tmp/")

    def processed_item_csv(self) -> str:
        return f"tmp/processed/{self._prefix}/movies.csv"

    def sasrec_format_csv_by_user_train(self) -> str:
        return f"tmp/{self._prefix}/sasrec_format_by_user_train.csv"

    def sasrec_format_csv_by_user_test(self) -> str:
        return f"tmp/{self._prefix}/sasrec_format_by_user_test.csv"

    def preprocess_rating(self) -> int:
        self.download()

        if self._prefix == "ml-1m":
            users = pd.read_csv(
                f"tmp/{self._prefix}/users.dat",
                sep="::",
                names=["user_id", "sex", "age_group", "occupation", "zip_code"],
            )
            ratings = pd.read_csv(
                f"tmp/{self._prefix}/ratings.dat",
                sep="::",
                names=["user_id", "movie_id", "rating", "unix_timestamp"],
            )
            movies = pd.read_csv(
                f"tmp/{self._prefix}/movies.dat",
                sep="::",
                names=["movie_id", "title", "genres"],
                encoding="iso-8859-1",
            )
        elif self._prefix == "ml-20m":
            # ml-20m
            # ml-20m doesn't have user data.
            users = None
            # ratings: userId,movieId,rating,timestamp
            ratings = pd.read_csv(
                f"tmp/{self._prefix}/ratings.csv",
                sep=",",
            )
            ratings.rename(
                columns={
                    "userId": "user_id",
                    "movieId": "movie_id",
                    "timestamp": "unix_timestamp",
                },
                inplace=True,
            )
            # movieId,title,genres
            # 1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
            # 2,Jumanji (1995),Adventure|Children|Fantasy
            movies = pd.read_csv(
                f"tmp/{self._prefix}/movies.csv",
                sep=",",
                encoding="iso-8859-1",
            )
            movies.rename(columns={"movieId": "movie_id"}, inplace=True)
        else:
            assert self._prefix == "ml-20mx16x32"
            # ml-1b
            user_ids = []
            movie_ids = []
            for i in range(16):
                train_file = f"tmp/{self._prefix}/trainx16x32_{i}.npz"
                with np.load(train_file) as data:
                    user_ids.extend([x[0] for x in data["arr_0"]])
                    movie_ids.extend([x[1] for x in data["arr_0"]])
            ratings = pd.DataFrame(
                data={
                    "user_id": user_ids,
                    "movie_id": movie_ids,
                    "rating": user_ids,  # placeholder
                    "unix_timestamp": movie_ids,  # placeholder
                }
            )
            users = None
            movies = None

        if movies is not None:
            # ML-1M and ML-20M only
            movies["year"] = movies["title"].apply(lambda x: x[-5:-1])
            movies["cleaned_title"] = movies["title"].apply(lambda x: x[:-7])
            # movies.year = pd.Categorical(movies.year)
            # movies["year"] = movies.year.cat.codes

        if users is not None:
            ## Users (ml-1m only)
            users.sex = pd.Categorical(users.sex)
            users["sex"] = users.sex.cat.codes

            users.age_group = pd.Categorical(users.age_group)
            users["age_group"] = users.age_group.cat.codes

            users.occupation = pd.Categorical(users.occupation)
            users["occupation"] = users.occupation.cat.codes

            users.zip_code = pd.Categorical(users.zip_code)
            users["zip_code"] = users.zip_code.cat.codes

        # Normalize movie ids to speed up training
        print(
            f"{self._prefix} #item before normalize: {len(set(ratings['movie_id'].values))}"
        )
        print(
            f"{self._prefix} max item id before normalize: {max(set(ratings['movie_id'].values))}"
        )
        # print(f"ratings.movie_id.cat.categories={ratings.movie_id.cat.categories}; {type(ratings.movie_id.cat.categories)}")
        # print(f"ratings.movie_id.cat.codes={ratings.movie_id.cat.codes}; {type(ratings.movie_id.cat.codes)}")
        # print(movie_id_to_cat)
        # ratings["movie_id"] = ratings.movie_id.cat.codes
        # print(f"{self._prefix} #item after normalize: {len(set(ratings['movie_id'].values))}")
        # print(f"{self._prefix} max item id after normalize: {max(set(ratings['movie_id'].values))}")
        # movies["remapped_id"] = movies["movie_id"].apply(lambda x: movie_id_to_cat[x])

        if self._convert_timestamp:
            ratings["unix_timestamp"] = pd.to_datetime(
                ratings["unix_timestamp"], unit="s"
            )

        # Save primary csv's
        if not os.path.exists(f"tmp/processed/{self._prefix}"):
            os.makedirs(f"tmp/processed/{self._prefix}")
        if users is not None:
            users.to_csv(f"tmp/processed/{self._prefix}/users.csv", index=False)
        if movies is not None:
            movies.to_csv(f"tmp/processed/{self._prefix}/movies.csv", index=False)
        ratings.to_csv(f"tmp/processed/{self._prefix}/ratings.csv", index=False)

        num_unique_users = len(set(ratings["user_id"].values))
        num_unique_items = len(set(ratings["movie_id"].values))

        # SASRec version
        ratings_group = ratings.sort_values(by=["unix_timestamp"]).groupby("user_id")
        seq_ratings_data = pd.DataFrame(
            data={
                "user_id": list(ratings_group.groups.keys()),
                "item_ids": list(ratings_group.movie_id.apply(list)),
                "ratings": list(ratings_group.rating.apply(list)),
                "timestamps": list(ratings_group.unix_timestamp.apply(list)),
            }
        )

        result = pd.DataFrame([[]])
        for col in ["item_ids"]:
            result[col + "_mean"] = seq_ratings_data[col].apply(len).mean()
            result[col + "_min"] = seq_ratings_data[col].apply(len).min()
            result[col + "_max"] = seq_ratings_data[col].apply(len).max()
        print(self._prefix)
        print(result)

        seq_ratings_data = self.to_seq_data(seq_ratings_data, users)
        seq_ratings_data.sample(frac=1).reset_index().to_csv(
            self.output_format_csv(), index=False, sep=","
        )

        # Split by user ids (not tested yet)
        user_id_split = int(num_unique_users * 0.9)
        seq_ratings_data_train = seq_ratings_data[
            seq_ratings_data["user_id"] <= user_id_split
        ]
        seq_ratings_data_train.sample(frac=1).reset_index().to_csv(
            self.sasrec_format_csv_by_user_train(),
            index=False,
            sep=",",
        )
        seq_ratings_data_test = seq_ratings_data[
            seq_ratings_data["user_id"] > user_id_split
        ]
        seq_ratings_data_test.sample(frac=1).reset_index().to_csv(
            self.sasrec_format_csv_by_user_test(), index=False, sep=","
        )
        print(
            f"{self._prefix}: train num user: {len(set(seq_ratings_data_train['user_id'].values))}"
        )
        print(
            f"{self._prefix}: test num user: {len(set(seq_ratings_data_test['user_id'].values))}"
        )

        # print(seq_ratings_data)
        if self.expected_num_unique_items() is not None:
            assert self.expected_num_unique_items() == num_unique_items, (
                f"Expected items: {self.expected_num_unique_items()}, got: {num_unique_items}"
            )

        return num_unique_items


class AmazonDataProcessor(DataProcessor):
    def __init__(
        self,
        download_path: str,
        saved_name: str,
        prefix: str,
        expected_num_unique_items: Optional[int],
    ) -> None:
        super().__init__(
            prefix,
            expected_num_unique_items=expected_num_unique_items,
            expected_max_item_id=None,
        )
        self._download_path = download_path
        self._saved_name = saved_name
        self._prefix = prefix

    def download(self) -> None:
        if not self.file_exists(self._saved_name):
            urlretrieve(self._download_path, self._saved_name)

    def preprocess_rating(self) -> int:
        self.download()

        ratings = pd.read_csv(
            self._saved_name,
            sep=",",
            names=["user_id", "item_id", "rating", "timestamp"],
        )
        print(f"{self._prefix} #data points before filter: {ratings.shape[0]}")
        print(
            f"{self._prefix} #user before filter: {len(set(ratings['user_id'].values))}"
        )
        print(
            f"{self._prefix} #item before filter: {len(set(ratings['item_id'].values))}"
        )

        # filter users and items with presence < 5
        item_id_count = (
            ratings["item_id"]
            .value_counts()
            .rename_axis("unique_values")
            .reset_index(name="item_count")
        )
        user_id_count = (
            ratings["user_id"]
            .value_counts()
            .rename_axis("unique_values")
            .reset_index(name="user_count")
        )
        ratings = ratings.join(item_id_count.set_index("unique_values"), on="item_id")
        ratings = ratings.join(user_id_count.set_index("unique_values"), on="user_id")
        ratings = ratings[ratings["item_count"] >= 5]
        ratings = ratings[ratings["user_count"] >= 5]
        print(f"{self._prefix} #data points after filter: {ratings.shape[0]}")

        # categorize user id and item id
        ratings["item_id"] = pd.Categorical(ratings["item_id"])
        ratings["item_id"] = ratings["item_id"].cat.codes
        ratings["user_id"] = pd.Categorical(ratings["user_id"])
        ratings["user_id"] = ratings["user_id"].cat.codes
        print(
            f"{self._prefix} #user after filter: {len(set(ratings['user_id'].values))}"
        )
        print(
            f"{self._prefix} #item ater filter: {len(set(ratings['item_id'].values))}"
        )

        num_unique_items = len(set(ratings["item_id"].values))

        # SASRec version
        ratings_group = ratings.sort_values(by=["timestamp"]).groupby("user_id")

        seq_ratings_data = pd.DataFrame(
            data={
                "user_id": list(ratings_group.groups.keys()),
                "item_ids": list(ratings_group.item_id.apply(list)),
                "ratings": list(ratings_group.rating.apply(list)),
                "timestamps": list(ratings_group.timestamp.apply(list)),
            }
        )

        seq_ratings_data = seq_ratings_data[
            seq_ratings_data["item_ids"].apply(len) >= 5
        ]

        result = pd.DataFrame([[]])
        for col in ["item_ids"]:
            result[col + "_mean"] = seq_ratings_data[col].apply(len).mean()
            result[col + "_min"] = seq_ratings_data[col].apply(len).min()
            result[col + "_max"] = seq_ratings_data[col].apply(len).max()
        print(self._prefix)
        print(result)

        if not os.path.exists(f"tmp/{self._prefix}"):
            os.makedirs(f"tmp/{self._prefix}")

        seq_ratings_data = self.to_seq_data(seq_ratings_data)
        seq_ratings_data.sample(frac=1).reset_index().to_csv(
            self.output_format_csv(), index=False, sep=","
        )

        if self.expected_num_unique_items() is not None:
            assert self.expected_num_unique_items() == num_unique_items, (
                f"expected: {self.expected_num_unique_items()}, actual: {num_unique_items}"
            )
            logging.info(f"{self.expected_num_unique_items()} unique items.")

        return num_unique_items


class EpinionsDataProcessor(DataProcessor):
    def __init__(
        self,
        prefix: str,
        data_dir: str,
        expected_num_unique_items: Optional[int] = None,
    ) -> None:
        super().__init__(
            prefix=prefix,
            expected_num_unique_items=expected_num_unique_items,
            expected_max_item_id=None,
        )
        self._data_dir: str = data_dir

    def processed_item_csv(self) -> str:
        return f"tmp/processed/{self._prefix}/items.csv"

    def _load_id_map(self, filename: str) -> Dict[int, str]:
        mapped_to_original: Dict[int, str] = {}
        with open(
            os.path.join(self._data_dir, filename),
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as f:
            # skip header
            next(f)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                mapped_to_original[int(parts[-1])] = parts[0]
        return mapped_to_original

    def _try_parse_record(
        self,
        merged_record: str,
    ) -> Optional[Tuple[str, str, int, float, str]]:
        parts = merged_record.split(maxsplit=2)
        if len(parts) < 2:
            return None
        item_id = parts[0]
        user_id = parts[1]
        rest = parts[2] if len(parts) == 3 else ""
        tokens = rest.split()
        for i in range(len(tokens) - 1):
            timestamp_token = tokens[i].rstrip(",")
            stars_token = tokens[i + 1].rstrip(",")
            if not timestamp_token.isdigit():
                continue
            if len(timestamp_token) < 9 or len(timestamp_token) > 11:
                continue
            try:
                stars = float(stars_token)
            except ValueError:
                continue
            if stars < 0.0 or stars > 5.0:
                continue
            words = " ".join(tokens[i + 2 :])
            return item_id, user_id, int(timestamp_token), stars, words
        return None

    def _looks_like_record_start(self, stripped_line: str) -> bool:
        tokens = stripped_line.split()
        if len(tokens) < 2:
            return False
        # Continuation lines frequently start with timestamp.
        if tokens[0].isdigit():
            return False
        # Full records usually contain "... <timestamp> <stars> ..." near the front.
        max_idx = min(len(tokens) - 1, 8)
        for i in range(2, max_idx):
            timestamp_token = tokens[i].rstrip(",")
            stars_token = tokens[i + 1].rstrip(",")
            if not timestamp_token.isdigit():
                continue
            if len(timestamp_token) < 9 or len(timestamp_token) > 11:
                continue
            try:
                stars = float(stars_token)
            except ValueError:
                continue
            if 0.0 <= stars <= 5.0:
                return True
        # Some records are split after "item user paid", with timestamp on next line.
        return len(tokens) <= 3

    def _build_pair_to_events(
        self,
        valid_item_ids: set[str],
        valid_user_ids: set[str],
    ) -> Dict[Tuple[str, str], List[Dict[str, Union[int, float, str]]]]:
        pair_to_events: Dict[
            Tuple[str, str], List[Dict[str, Union[int, float, str]]]
        ] = defaultdict(list)
        current_record = ""
        current_start_line = -1
        parse_failed = 0

        def flush_record() -> None:
            nonlocal current_record
            nonlocal current_start_line
            nonlocal parse_failed
            if not current_record:
                return
            parsed = self._try_parse_record(current_record)
            if parsed is None:
                parse_failed += 1
            else:
                item_id, user_id, timestamp, stars, words = parsed
                if item_id in valid_item_ids and user_id in valid_user_ids:
                    pair_to_events[(user_id, item_id)].append(
                        {
                            "timestamp": timestamp,
                            "stars": stars,
                            "words": words,
                            "raw_line": current_start_line,
                        }
                    )
            current_record = ""
            current_start_line = -1

        with open(
            os.path.join(self._data_dir, "epinions.txt"),
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as f:
            # skip header
            next(f)
            for line_no, raw_line in enumerate(f, start=2):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                is_record_start = self._looks_like_record_start(stripped)
                if is_record_start:
                    flush_record()
                    current_record = stripped
                    current_start_line = line_no
                else:
                    if current_record:
                        current_record = current_record + " " + stripped
                    else:
                        # orphan continuation line; ignored safely
                        continue
            flush_record()

        for key in pair_to_events:
            pair_to_events[key].sort(
                key=lambda x: (int(x["timestamp"]), int(x["raw_line"]))
            )
        logging.info(f"{self._prefix}: failed to parse {parse_failed} raw records.")
        return pair_to_events

    def _read_split_interactions(
        self,
        split_name: str,
        user_map: Dict[int, str],
        item_map: Dict[int, str],
    ) -> List[Dict[str, Union[int, str]]]:
        rows: List[Dict[str, Union[int, str]]] = []
        with open(
            os.path.join(self._data_dir, f"{split_name}.txt"),
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                user_id_str, item_id_str, label_str = line.split()
                mapped_user_id = int(user_id_str)
                mapped_item_id = int(item_id_str)
                if mapped_user_id not in user_map:
                    raise ValueError(f"Missing user map for mapped id {mapped_user_id}")
                if mapped_item_id not in item_map:
                    raise ValueError(f"Missing item map for mapped id {mapped_item_id}")
                rows.append(
                    {
                        "mapped_user_id": mapped_user_id,
                        "mapped_item_id": mapped_item_id,
                        "label": int(float(label_str)),
                        "original_user_id": user_map[mapped_user_id],
                        "original_item_id": item_map[mapped_item_id],
                    }
                )
        return rows

    def _get_sorted_events_by_user(
        self, df: pd.DataFrame
    ) -> Dict[int, List[Dict[str, int]]]:
        if df.empty:
            return {}
        df = df.sort_values(
            by=["mapped_user_id", "timestamp", "raw_line", "mapped_item_id"]
        )
        events_by_user: Dict[int, List[Dict[str, int]]] = {}
        for user_id, user_df in df.groupby("mapped_user_id", sort=True):
            events_by_user[int(user_id)] = [
                {
                    "mapped_item_id": int(row["mapped_item_id"]),
                    "stars": int(round(float(row["stars"]))),
                    "timestamp": int(row["timestamp"]),
                }
                for _, row in user_df.iterrows()
            ]
        return events_by_user

    def _event_sequence_to_row(
        self, user_id: int, events: List[Dict[str, int]]
    ) -> Dict[str, Union[int, str]]:
        return {
            "user_id": user_id,
            "sequence_item_ids": ",".join([str(x["mapped_item_id"]) for x in events]),
            "sequence_ratings": ",".join([str(x["stars"]) for x in events]),
            "sequence_timestamps": ",".join([str(x["timestamp"]) for x in events]),
        }

    def _build_train_sequence_df(self, train_df: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, Union[int, str]]] = []
        for user_id, events in self._get_sorted_events_by_user(train_df).items():
            history: List[Dict[str, int]] = []
            for event in events:
                if len(history) > 0:
                    rows.append(self._event_sequence_to_row(user_id, history + [event]))
                history.append(event)
        return pd.DataFrame(
            rows,
            columns=[
                "user_id",
                "sequence_item_ids",
                "sequence_ratings",
                "sequence_timestamps",
            ],
        )

    def _build_eval_sequence_df(
        self,
        history_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: List[Dict[str, Union[int, str]]] = []
        history_events_by_user = self._get_sorted_events_by_user(history_df)
        target_events_by_user = self._get_sorted_events_by_user(target_df)
        for user_id, target_events in target_events_by_user.items():
            history_events = history_events_by_user.get(user_id, [])
            if len(history_events) == 0:
                continue
            for target_event in target_events:
                rows.append(
                    self._event_sequence_to_row(
                        user_id=user_id,
                        events=history_events + [target_event],
                    )
                )
        return pd.DataFrame(
            rows,
            columns=[
                "user_id",
                "sequence_item_ids",
                "sequence_ratings",
                "sequence_timestamps",
            ],
        )

    def preprocess_rating(self) -> int:
        user_map = self._load_id_map("user_id_map.txt")
        item_map = self._load_id_map("item_id_map.txt")

        pair_to_events = self._build_pair_to_events(
            valid_item_ids=set(item_map.values()),
            valid_user_ids=set(user_map.values()),
        )

        split_rows = {
            split: self._read_split_interactions(split, user_map, item_map)
            for split in ["train", "valid", "test"]
        }

        pair_offsets: Dict[Tuple[str, str], int] = defaultdict(int)
        split_to_enriched_df: Dict[str, pd.DataFrame] = {}
        for split in ["train", "valid", "test"]:
            enriched_rows = []
            for row in split_rows[split]:
                pair_key = (
                    str(row["original_user_id"]),
                    str(row["original_item_id"]),
                )
                offset = pair_offsets[pair_key]
                if pair_key not in pair_to_events or offset >= len(pair_to_events[pair_key]):
                    raise ValueError(
                        f"Missing raw interaction for split={split}, pair={pair_key}, offset={offset}."
                    )
                raw_event = pair_to_events[pair_key][offset]
                pair_offsets[pair_key] += 1
                enriched_rows.append(
                    {
                        "mapped_user_id": int(row["mapped_user_id"]),
                        "mapped_item_id": int(row["mapped_item_id"]),
                        "label": int(row["label"]),
                        "original_user_id": str(row["original_user_id"]),
                        "original_item_id": str(row["original_item_id"]),
                        "timestamp": int(raw_event["timestamp"]),
                        "stars": float(raw_event["stars"]),
                        "words": str(raw_event["words"]),
                        "raw_line": int(raw_event["raw_line"]),
                    }
                )
            split_to_enriched_df[split] = pd.DataFrame(enriched_rows)

        os.makedirs(f"tmp/{self._prefix}", exist_ok=True)
        os.makedirs(f"tmp/processed/{self._prefix}", exist_ok=True)

        pd.DataFrame(
            {
                "original_id": list(item_map.values()),
                "mapped_id": list(item_map.keys()),
            }
        ).sort_values(by=["mapped_id"]).to_csv(self.processed_item_csv(), index=False)

        for split in ["train", "valid", "test"]:
            split_to_enriched_df[split].to_csv(
                f"tmp/processed/{self._prefix}/interactions_{split}.csv",
                index=False,
            )

        seq_train = self._build_train_sequence_df(split_to_enriched_df["train"])
        seq_valid = self._build_eval_sequence_df(
            history_df=split_to_enriched_df["train"],
            target_df=split_to_enriched_df["valid"],
        )
        seq_test = self._build_eval_sequence_df(
            history_df=split_to_enriched_df["train"],
            target_df=split_to_enriched_df["test"],
        )

        seq_train.sample(frac=1).reset_index(drop=True).to_csv(
            self.sasrec_format_csv_by_user_train(), index=False
        )
        seq_valid.sample(frac=1).reset_index(drop=True).to_csv(
            self.sasrec_format_csv_by_user_valid(), index=False
        )
        seq_test.sample(frac=1).reset_index(drop=True).to_csv(
            self.sasrec_format_csv_by_user_test(), index=False
        )
        # Keep default output path for compatibility with existing tooling.
        seq_train.sample(frac=1).reset_index(drop=True).to_csv(
            self.output_format_csv(), index=False
        )

        num_unique_items = len(item_map)
        if self.expected_num_unique_items() is not None:
            assert num_unique_items == self.expected_num_unique_items(), (
                f"expected: {self.expected_num_unique_items()}, actual: {num_unique_items}"
            )

        logging.info(
            f"{self._prefix}: train/valid/test rows="
            f"{len(split_to_enriched_df['train'])}/"
            f"{len(split_to_enriched_df['valid'])}/"
            f"{len(split_to_enriched_df['test'])}, "
            f"#items={num_unique_items}."
        )
        return num_unique_items


class YelpDataProcessor(DataProcessor):
    def __init__(
        self,
        prefix: str,
        data_dir: str,
        expected_num_unique_items: Optional[int] = None,
    ) -> None:
        super().__init__(
            prefix=prefix,
            expected_num_unique_items=expected_num_unique_items,
            expected_max_item_id=None,
        )
        self._data_dir: str = data_dir

    def processed_item_csv(self) -> str:
        return f"tmp/processed/{self._prefix}/items.csv"

    def _load_jsonl_id_map(
        self, filename: str, mapped_id_key: str, original_id_key: str
    ) -> Dict[int, str]:
        mapped_to_original: Dict[int, str] = {}
        with open(
            os.path.join(self._data_dir, filename),
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                mapped_to_original[int(row[mapped_id_key])] = str(row[original_id_key])
        return mapped_to_original

    def _read_split_interactions(
        self,
        split_name: str,
        user_map: Dict[int, str],
        item_map: Dict[int, str],
    ) -> List[Dict[str, Union[int, str]]]:
        rows: List[Dict[str, Union[int, str]]] = []
        with open(
            os.path.join(self._data_dir, f"{split_name}.txt"),
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                user_id_str, item_id_str, label_str = line.split()
                mapped_user_id = int(user_id_str)
                mapped_item_id = int(item_id_str)
                if mapped_user_id not in user_map:
                    raise ValueError(f"Missing user map for mapped id {mapped_user_id}")
                if mapped_item_id not in item_map:
                    raise ValueError(f"Missing item map for mapped id {mapped_item_id}")
                rows.append(
                    {
                        "mapped_user_id": mapped_user_id,
                        "mapped_item_id": mapped_item_id,
                        "label": int(float(label_str)),
                        "original_user_id": user_map[mapped_user_id],
                        "original_item_id": item_map[mapped_item_id],
                    }
                )
        return rows

    def _to_unix_timestamp(self, dt_str: str) -> int:
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        return int(dt.replace(tzinfo=timezone.utc).timestamp())

    def _scan_reviews_for_pairs(
        self,
        required_pairs: set[Tuple[str, str]],
    ) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[Tuple[str, str], int]]:
        selected_event_by_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
        match_count_by_pair: Dict[Tuple[str, str], int] = defaultdict(int)
        review_file = os.path.join(self._data_dir, "yelp_academic_dataset_review.json")
        with open(review_file, "r", encoding="utf-8", errors="ignore") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                pair_key = (str(row["user_id"]), str(row["business_id"]))
                if pair_key not in required_pairs:
                    continue
                stars = float(row["stars"])
                date_str = str(row["date"])
                event = {
                    "review_id": str(row["review_id"]),
                    "stars": stars,
                    "date": date_str,
                    "timestamp": self._to_unix_timestamp(date_str),
                    "raw_line": line_no,
                }
                match_count_by_pair[pair_key] += 1
                if pair_key not in selected_event_by_pair:
                    selected_event_by_pair[pair_key] = event
                else:
                    prev = selected_event_by_pair[pair_key]
                    # Keep the earliest interaction for duplicated (user, item) pairs.
                    if (event["timestamp"], event["raw_line"]) < (
                        prev["timestamp"],
                        prev["raw_line"],
                    ):
                        selected_event_by_pair[pair_key] = event
        return selected_event_by_pair, dict(match_count_by_pair)

    def _build_split_df(
        self,
        split_name: str,
        split_rows: List[Dict[str, Union[int, str]]],
        selected_event_by_pair: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> pd.DataFrame:
        enriched_rows = []
        for row in split_rows:
            pair_key = (str(row["original_user_id"]), str(row["original_item_id"]))
            event = selected_event_by_pair.get(pair_key)
            if event is None:
                raise ValueError(
                    f"Missing review event for split={split_name}, pair={pair_key}."
                )
            enriched_rows.append(
                {
                    "mapped_user_id": int(row["mapped_user_id"]),
                    "mapped_item_id": int(row["mapped_item_id"]),
                    "label": int(row["label"]),
                    "original_user_id": str(row["original_user_id"]),
                    "original_item_id": str(row["original_item_id"]),
                    "review_id": str(event["review_id"]),
                    "date": str(event["date"]),
                    "timestamp": int(event["timestamp"]),
                    "stars": float(event["stars"]),
                    "raw_line": int(event["raw_line"]),
                }
            )
        return pd.DataFrame(enriched_rows)

    def _get_sorted_events_by_user(
        self, df: pd.DataFrame
    ) -> Dict[int, List[Dict[str, int]]]:
        if df.empty:
            return {}
        df = df.sort_values(
            by=["mapped_user_id", "timestamp", "raw_line", "mapped_item_id"]
        )
        events_by_user: Dict[int, List[Dict[str, int]]] = {}
        for user_id, user_df in df.groupby("mapped_user_id", sort=True):
            events_by_user[int(user_id)] = [
                {
                    "mapped_item_id": int(row["mapped_item_id"]),
                    "stars": int(round(float(row["stars"]))),
                    "timestamp": int(row["timestamp"]),
                }
                for _, row in user_df.iterrows()
            ]
        return events_by_user

    def _event_sequence_to_row(
        self, user_id: int, events: List[Dict[str, int]]
    ) -> Dict[str, Union[int, str]]:
        return {
            "user_id": user_id,
            "sequence_item_ids": ",".join([str(x["mapped_item_id"]) for x in events]),
            "sequence_ratings": ",".join([str(x["stars"]) for x in events]),
            "sequence_timestamps": ",".join([str(x["timestamp"]) for x in events]),
        }

    def _build_train_sequence_df(self, train_df: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, Union[int, str]]] = []
        for user_id, events in self._get_sorted_events_by_user(train_df).items():
            history: List[Dict[str, int]] = []
            for event in events:
                if len(history) > 0:
                    rows.append(self._event_sequence_to_row(user_id, history + [event]))
                history.append(event)
        return pd.DataFrame(
            rows,
            columns=[
                "user_id",
                "sequence_item_ids",
                "sequence_ratings",
                "sequence_timestamps",
            ],
        )

    def _build_eval_sequence_df(
        self,
        history_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: List[Dict[str, Union[int, str]]] = []
        history_events_by_user = self._get_sorted_events_by_user(history_df)
        target_events_by_user = self._get_sorted_events_by_user(target_df)
        for user_id, target_events in target_events_by_user.items():
            history_events = history_events_by_user.get(user_id, [])
            if len(history_events) == 0:
                continue
            for target_event in target_events:
                rows.append(
                    self._event_sequence_to_row(
                        user_id=user_id,
                        events=history_events + [target_event],
                    )
                )
        return pd.DataFrame(
            rows,
            columns=[
                "user_id",
                "sequence_item_ids",
                "sequence_ratings",
                "sequence_timestamps",
            ],
        )

    def preprocess_rating(self) -> int:
        user_map = self._load_jsonl_id_map(
            filename="yelp_user.json",
            mapped_id_key="uid",
            original_id_key="user_id",
        )
        item_map = self._load_jsonl_id_map(
            filename="yelp_item.json",
            mapped_id_key="iid",
            original_id_key="business_id",
        )

        split_rows = {
            split: self._read_split_interactions(split, user_map, item_map)
            for split in ["train", "valid", "test"]
        }
        required_pairs = {
            (str(row["original_user_id"]), str(row["original_item_id"]))
            for split in ["train", "valid", "test"]
            for row in split_rows[split]
        }

        selected_event_by_pair, match_count_by_pair = self._scan_reviews_for_pairs(
            required_pairs=required_pairs
        )
        missing_pairs = [pair for pair in required_pairs if pair not in selected_event_by_pair]
        if len(missing_pairs) > 0:
            raise ValueError(
                f"{self._prefix}: {len(missing_pairs)} required (user,item) pairs are missing in review file."
            )
        multi_review_pairs = sum(1 for _, c in match_count_by_pair.items() if c > 1)
        logging.info(
            f"{self._prefix}: matched {len(selected_event_by_pair)} pairs from reviews; "
            f"{multi_review_pairs} pairs have duplicated raw reviews."
        )

        split_to_df = {
            split: self._build_split_df(
                split_name=split,
                split_rows=split_rows[split],
                selected_event_by_pair=selected_event_by_pair,
            )
            for split in ["train", "valid", "test"]
        }

        os.makedirs(f"tmp/{self._prefix}", exist_ok=True)
        os.makedirs(f"tmp/processed/{self._prefix}", exist_ok=True)

        pd.DataFrame(
            {
                "original_id": list(item_map.values()),
                "mapped_id": list(item_map.keys()),
            }
        ).sort_values(by=["mapped_id"]).to_csv(self.processed_item_csv(), index=False)

        for split in ["train", "valid", "test"]:
            split_to_df[split].to_csv(
                f"tmp/processed/{self._prefix}/interactions_{split}.csv",
                index=False,
            )

        seq_train = self._build_train_sequence_df(split_to_df["train"])
        seq_valid = self._build_eval_sequence_df(
            history_df=split_to_df["train"],
            target_df=split_to_df["valid"],
        )
        seq_test = self._build_eval_sequence_df(
            history_df=split_to_df["train"],
            target_df=split_to_df["test"],
        )

        seq_train.sample(frac=1).reset_index(drop=True).to_csv(
            self.sasrec_format_csv_by_user_train(), index=False
        )
        seq_valid.sample(frac=1).reset_index(drop=True).to_csv(
            self.sasrec_format_csv_by_user_valid(), index=False
        )
        seq_test.sample(frac=1).reset_index(drop=True).to_csv(
            self.sasrec_format_csv_by_user_test(), index=False
        )
        # Keep default output path for compatibility with existing tooling.
        seq_train.sample(frac=1).reset_index(drop=True).to_csv(
            self.output_format_csv(), index=False
        )

        num_unique_items = len(item_map)
        if self.expected_num_unique_items() is not None:
            assert num_unique_items == self.expected_num_unique_items(), (
                f"expected: {self.expected_num_unique_items()}, actual: {num_unique_items}"
            )

        logging.info(
            f"{self._prefix}: train/valid/test rows="
            f"{len(split_to_df['train'])}/"
            f"{len(split_to_df['valid'])}/"
            f"{len(split_to_df['test'])}, "
            f"#items={num_unique_items}."
        )
        return num_unique_items


def get_common_preprocessors() -> Dict[
    str,
    Union[
        AmazonDataProcessor,
        EpinionsDataProcessor,
        YelpDataProcessor,
        MovielensDataProcessor,
        MovielensSyntheticDataProcessor,
    ],
]:
    ml_1m_dp = MovielensDataProcessor(  # pyre-ignore [45]
        "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "tmp/movielens1m.zip",
        prefix="ml-1m",
        convert_timestamp=False,
        expected_num_unique_items=3706,
        expected_max_item_id=3952,
    )
    ml_20m_dp = MovielensDataProcessor(  # pyre-ignore [45]
        "http://files.grouplens.org/datasets/movielens/ml-20m.zip",
        "tmp/movielens20m.zip",
        prefix="ml-20m",
        convert_timestamp=False,
        expected_num_unique_items=26744,
        expected_max_item_id=131262,
    )
    ml_1b_dp = MovielensDataProcessor(  # pyre-ignore [45]
        "https://files.grouplens.org/datasets/movielens/ml-20mx16x32.tar",
        "tmp/movielens1b.tar",
        prefix="ml-20mx16x32",
        convert_timestamp=False,
    )
    ml_3b_dp = MovielensSyntheticDataProcessor(  # pyre-ignore [45]
        prefix="ml-3b",
        expected_num_unique_items=26743 * 32,
        expected_max_item_id=26743 * 32,
    )
    amzn_books_dp = AmazonDataProcessor(  # pyre-ignore [45]
        "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv",
        "tmp/ratings_Books.csv",
        prefix="amzn_books",
        expected_num_unique_items=695762,
    )
    epinions_dp = EpinionsDataProcessor(  # pyre-ignore [45]
        prefix="epinions",
        data_dir="data/epinions",
        expected_num_unique_items=12440,
    )
    yelp_dp = YelpDataProcessor(  # pyre-ignore [45]
        prefix="yelp",
        data_dir="/home/linjx/code/SELFRec-main/dataset/yelp",
        expected_num_unique_items=11010,
    )
    return {
        "ml-1m": ml_1m_dp,
        "ml-20m": ml_20m_dp,
        "ml-1b": ml_1b_dp,
        "ml-3b": ml_3b_dp,
        "amzn-books": amzn_books_dp,
        "epinions": epinions_dp,
        "yelp": yelp_dp,
    }
