"""Download the PTB-XL 500Hz waveform records (records500/, used by load_raw_data) into data/ via wfdb.

Usage:
    python -m utils.download_data
"""

import argparse
import multiprocessing.dummy
import os
import posixpath

import pandas as pd
from tqdm import tqdm
from wfdb.io import download

PTBXL_DB = "ptb-xl"


def download_ptbxl(dest_dir: str = "data") -> None:
    os.makedirs(dest_dir, exist_ok=True)
    info = pd.read_csv(os.path.join(dest_dir, "ptbxl_database.csv"), index_col="ecg_id")
    records = info.filename_hr.tolist()

    db_dir = posixpath.join(PTBXL_DB, download.get_version(PTBXL_DB))
    all_files = [rec + ".hea" for rec in records] + [rec + ".dat" for rec in records]
    dl_inputs = [
        (os.path.split(f)[1], os.path.split(f)[0], db_dir, dest_dir, True, False)
        for f in all_files
    ]
    download.make_local_dirs(dest_dir, dl_inputs, keep_subdirs=True)

    print(
        f"Downloading {len(records)} records500 files from {PTBXL_DB} into {dest_dir!r} ..."
    )
    with multiprocessing.dummy.Pool(processes=4) as pool:
        for _ in tqdm(
            pool.imap_unordered(download.dl_pn_file, dl_inputs), total=len(dl_inputs)
        ):
            pass
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the PTB-XL waveform dataset."
    )
    parser.add_argument(
        "--dest", default="data", help="Destination directory (default: data)"
    )
    args = parser.parse_args()
    download_ptbxl(args.dest)
