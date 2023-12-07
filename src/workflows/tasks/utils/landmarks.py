from flytekit import task
from flytekit.types.directory import FlyteDirectory
from sign_language_tools.pose.mediapipe.extraction import (
    extract_and_save_landmarks,
)

from sign_language_tools.pose.transform import (
    Compose,
    InterpolateMissing,
    SavitchyGolayFiltering,
)

import gc
import os
from multiprocessing import Pool
import numpy as np
from typing import Optional


@task
def extract_landmark(
    dataset_path: FlyteDirectory,
    video_id: str,
    skip_existing: bool = True,
) -> str:
    video_path = os.path.join(dataset_path, "videos", f"{video_id}.mp4")

    can_skip = all(
        [
            os.path.isfile(
                os.path.join(dataset_path, "poses_raw", lm_set, f"{video_id}.npy")
            )
            for lm_set in ["face", "pose", "left_hand", "right_hand"]
        ]
    )

    if can_skip and skip_existing:
        print(f"Skipped extraction for {video_id}.")
        return video_id

    extract_and_save_landmarks(
        video_path, video_id, os.path.join(dataset_path, "poses_raw")
    )

    return video_id


@task
def clean_pose(output_dir: str, instance_id: str):
    transform = Compose(
        [
            InterpolateMissing(),
            SavitchyGolayFiltering(window_length=7, polynom_order=2),
        ]
    )
    for pose_name in ["face", "pose", "left_hand", "right_hand"]:
        pose = np.load(f"{output_dir}/poses_raw/{pose_name}/{instance_id}.npy")
        pose = transform(pose).astype("float16")
        np.save(f"{output_dir}/poses/{pose_name}/{instance_id}.npy", pose)
