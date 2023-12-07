from flytekit import task
from flytekit.types.directory import FlyteDirectory
import pandas as pd
import os
from tasks.lsa.labels import SIGNS_LABEL


@task
def create_instance_csv(directory: FlyteDirectory) -> list[str]:
    vid_dir = os.path.join(directory, "videos")
    data = {"id": [], "sign": [], "signer": []}

    for file in os.listdir(vid_dir):
        if file.endswith(".mp4"):
            id = file.split(".")[0]
            data["id"].append(id)

            metadata = id.split("_")
            sign_idx = int(metadata[0]) - 1

            data["sign"].append(SIGNS_LABEL[sign_idx])
            data["signer"].append(metadata[1])

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(directory, "instances.csv"), index=False)

    return df["id"].tolist()
