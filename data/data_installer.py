import os
import numpy as np
import pandas as pd

from pathlib import Path
from lightkurve import search_lightcurve

# ============================================================
# CONFIG
# ============================================================

KOI_FILE = "data/cum_koi_table.csv"

N_PER_CLASS = 1000
MIN_SEQUENCE_LENGTH = 50000
MIN_OBSERVATION_SPAN = 1000

DATASET_DIR = Path("final_dataset")

CLASSES = [
    "CONFIRMED",
    "FALSE POSITIVE",
    "CANDIDATE"
]

RANDOM_STATE = 42

# ============================================================
# CREATE FOLDERS
# ============================================================

DATASET_DIR.mkdir(exist_ok=True)

for cls in CLASSES:
    folder_name = cls.replace(" ", "_")
    (DATASET_DIR / folder_name).mkdir(exist_ok=True)

# ============================================================
# LOAD KOI TABLE
# ============================================================

print("Loading KOI table...")

koi = pd.read_csv(KOI_FILE)

# ============================================================
# DOWNLOAD FUNCTION
# ============================================================

def download_star(row, max_retries=3):

    kepid = int(row["kepid"])

    label = row["koi_disposition"]

    class_folder = (
        DATASET_DIR /
        label.replace(" ", "_")
    )

    save_file = class_folder / f"{kepid}.npz"

    if save_file.exists():
        return "already"

    for attempt in range(max_retries):

        try:

            result = search_lightcurve(
                f"KIC {kepid}",
                mission="Kepler",
                cadence="long"
            )

            if len(result) == 0:
                return "missing"

            lc_collection = result.download_all()

            if lc_collection is None:
                continue

            lc = (
                lc_collection
                .stitch()
                .remove_nans()
                .normalize()
            )

            time = np.asarray(lc.time.value)

            if len(time) < MIN_SEQUENCE_LENGTH or (time[-1] - time[0]) < MIN_OBSERVATION_SPAN:
                return "rejected"

            flux = np.asarray(lc.flux.value)

            try:
                flux_err = np.asarray(
                    lc.flux_err.value
                )
            except:
                flux_err = np.full_like(
                    flux,
                    np.nan
                )

            np.savez_compressed(
                save_file,
                time=time,
                flux=flux,
                flux_err=flux_err
            )

            return "success"

        except Exception as e:

            print(
                f"KIC {kepid} "
                f"Attempt {attempt+1}/{max_retries} "
                f"Failed: {e}"
            )

    return "failed"


# ============================================================
# DOWNLOAD LOOP
# ============================================================

successful_rows = []

success = 0
already = 0
failed = 0
missing = 0

for cls in CLASSES:

    print("\n" + "=" * 60)
    print(f"Processing {cls}")
    print("=" * 60)

    candidates = (
        koi[koi["koi_disposition"] == cls]
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    class_success = 0

    for idx, row in candidates.iterrows():

        if class_success >= N_PER_CLASS:
            break

        kepid = int(row["kepid"])

        print(
            f"{cls} | "
            f"{class_success+1}/{N_PER_CLASS} | "
            f"KIC {kepid}"
        )

        status = download_star(row)

        if status == "success":

            success += 1
            class_success += 1

            successful_rows.append(row)

        elif status == "already":

            already += 1
            class_success += 1

            successful_rows.append(row)

        elif status == "failed":

            failed += 1

        elif status == "missing":

            missing += 1

        print(
            f"Class Success={class_success}/{N_PER_CLASS}"
        )

    print(
        f"{cls} completed with "
        f"{class_success} samples"
    )


# ============================================================
# CREATE FINAL METADATA
# ============================================================

metadata_columns = [
    "kepid",
    "kepoi_name",
    "koi_disposition",
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_time0bk"
]

metadata = pd.DataFrame(successful_rows)

metadata_columns = [
    c for c in metadata_columns
    if c in metadata.columns
]

metadata = metadata[metadata_columns]

metadata_path = DATASET_DIR / "metadata.csv"

metadata.to_csv(
    metadata_path,
    index=False
)

print("\nMetadata saved ->", metadata_path)

print("\nFinal Dataset Summary")
print(
    metadata["koi_disposition"]
    .value_counts()
)

print("\nFinished")
print(
    f"Success={success}, "
    f"Already={already}, "
    f"Failed={failed}, "
    f"Missing={missing}"
)