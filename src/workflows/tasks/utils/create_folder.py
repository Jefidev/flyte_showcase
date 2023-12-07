import os
from flytekit import task
import flytekit
from flytekit.types.directory import FlyteDirectory


@task
def create_folders(dataset_name: str, out_location: str) -> FlyteDirectory:
    """A simple Flyte task to create folders.

    The @task decorator allows Flyte to use this function as a Flyte task, which
    is executed as an isolated, containerized unit of compute.
    """
    work_dir = flytekit.current_context().working_directory
    dataset_dir = os.path.join(work_dir, dataset_name)

    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "metadata"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "metadata", "splits"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "videos"), exist_ok=True)

    for pose_folder in ["pose", "poses_raw"]:
        for body_part in ["face", "right_hand", "left_hand", "pose"]:
            os.makedirs(
                os.path.join(dataset_dir, pose_folder, body_part),
                exist_ok=True,
            )

    if out_location:
        return FlyteDirectory(
            dataset_dir, remote_directory=os.path.join(out_location, dataset_name)
        )
    else:
        return FlyteDirectory(dataset_dir)
