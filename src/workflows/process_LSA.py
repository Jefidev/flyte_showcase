from flytekit import workflow, map_task
from tasks.utils.create_folder import create_folders
from tasks.utils.landmarks import extract_landmark, clean_pose
from tasks.utils.metadata import (
    create_label_mapping,
    count_sign_occurences,
    create_folds,
    create_train_test_splits,
    create_mini_sample,
    create_all,
)
from tasks.lsa.instances import create_instance_csv
import functools


@workflow
def process_LSA(dataset_name: str, out_location: str):
    """A simple Flyte workflow to process the LSA dataset.

    The @workflow decorator allows Flyte to use this function as a Flyte workflow, which
    is executed as an isolated, containerized unit of compute.
    """
    # This is a placeholder for the actual processing code.
    # We'll add the real code in the next step.

    folder = create_folders(dataset_name=dataset_name, out_location=out_location)
    ids = create_instance_csv(directory=folder)

    # Metadata tasks
    promise_label_map = create_label_mapping(root=folder)
    promise_count_occurences = count_sign_occurences(root=folder)

    # Create folds
    promise_create_folds = create_folds(root=folder, n_splits=5)
    promise_create_train_test_splits = create_train_test_splits(root=folder)
    promise_create_mini_sample = create_mini_sample(root=folder)
    promise_all = create_all(root=folder, n_splits=5)

    # Landmarks tasks
    partial_extract_landmark = functools.partial(
        extract_landmark,
        dataset_path=folder,
        skip_existing=True,
    )
    processed_ids = map_task(
        partial_extract_landmark,
        concurrency=4,
    )(video_id=ids)

    partial_clean_pose = functools.partial(clean_pose, output_dir=folder)
    clean_poses_task = map_task(
        partial_clean_pose,
        concurrency=4,
    )(instance_id=processed_ids)

    ids >> promise_label_map
    ids >> promise_count_occurences

    # Handling folds
    (
        ids
        >> promise_create_folds
        >> promise_create_train_test_splits
        >> promise_create_mini_sample
    )
    promise_create_folds >> promise_all


if __name__ == "__main__":
    # This is the entrypoint when the workflow is run locally using the
    # `flytekit` command.
    out_location = "/home/jeromefink/Documents/unamur/signLanguage/Data"
    dataset_name = "lsa64"
    process_LSA(dataset_name=dataset_name, out_location=out_location)
