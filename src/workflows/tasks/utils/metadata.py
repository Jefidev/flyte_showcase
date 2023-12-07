from flytekit.types.directory import FlyteDirectory
from sklearn.model_selection import GroupKFold
from flytekit import task
import json
import pandas as pd
import os
import random


def save_split(root: FlyteDirectory, split_name: str, instances: list[str]):
    with open(
        os.path.join(root, "metadata", "splits", split_name + ".json"), "w"
    ) as file:
        json.dump(instances, file, indent=2)


def load_split(root: FlyteDirectory, split_name: str):
    with open(os.path.join(root, "metadata", "splits", split_name + ".json")) as file:
        return json.load(file)


def merge_splits(root: str, splits: list[str], new_split: str):
    instances = []
    for split in splits:
        instances += load_split(root, split)
    instances = list(set(instances))
    save_split(root, new_split, instances)


@task
def create_label_mapping(root: FlyteDirectory) -> None:
    instances = pd.read_csv(os.path.join(root, "instances.csv"), dtype=str)
    unique_signs = instances["sign"].unique().tolist()

    data = {"sign": [], "class": []}

    for idx, sign in enumerate(unique_signs):
        data["sign"].append(sign)
        data["class"].append(idx)

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(root, "metadata", "sign_to_index.csv"), index=False)


@task
def count_sign_occurences(root: FlyteDirectory):
    instances = pd.read_csv(os.path.join(root, "instances.csv"), dtype=str)
    count = instances["sign"].value_counts()
    count.to_csv(
        os.path.join(root, "metadata", "sign_occurences.csv"), index=True, header=True
    )


@task
def create_folds(root: FlyteDirectory, n_splits: int = 5):
    instances = pd.read_csv(os.path.join(root, "instances.csv"), dtype=str)
    groups = instances["signer"].values
    instances["fold"] = -1

    group_k_fold = GroupKFold(n_splits=n_splits)
    for index, (train_index, test_index) in enumerate(
        group_k_fold.split(instances["id"], groups=groups)
    ):
        instances.loc[test_index, "fold"] = index

    for fold_idx in range(n_splits):
        fold_instances = instances.query(f"fold == {fold_idx}")["id"].tolist()
        save_split(root, f"fold_{fold_idx}", fold_instances)


@task
def create_train_test_splits(root: FlyteDirectory):
    split_folds = {
        "test": ["fold_0", "fold_1"],
        "train": ["fold_2", "fold_3", "fold_4"],
    }
    merge_splits(root, split_folds["test"], "test")
    merge_splits(root, split_folds["train"], "train")


@task
def create_mini_sample(root: FlyteDirectory):
    instances = load_split(root, "train")
    random.seed(42)
    instances = random.sample(instances, 10)
    save_split(root, "mini_sample", instances)


@task
def create_all(root: FlyteDirectory, n_splits: int = 5):
    splits = [f"fold_{n}" for n in range(n_splits)]
    merge_splits(root, splits, "all")
