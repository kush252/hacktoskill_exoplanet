import numpy as np
import pandas as pd
from pathlib import Path

DATASET_DIR = Path("dataset")

metadata = pd.read_csv(
    DATASET_DIR / "metadata.csv"
)

def load_star(kepid, label):

    file = (
        DATASET_DIR /
        label.replace(" ", "_") /
        f"{kepid}.npz"
    )

    data = np.load(file)

    return {
        "time": data["time"],
        "flux": data["flux"],
        "flux_err": data["flux_err"]
    }

# Example

row = metadata.iloc[0]

star = load_star(
    row["kepid"],
    row["koi_disposition"]
)

print(star["time"].shape)
print(star["flux"].shape)