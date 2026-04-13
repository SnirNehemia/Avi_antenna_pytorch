import os

import shutil

from pathlib import Path

# --- CONFIGURATION ---

SOURCE_DIR = Path("G:\General_models\output_dipole_no_dielectric")

DEST_LOW = Path("G:\General_models\output_dipole_NoDielectric_WithoutCurrents")

DEST_HIGH = Path("G:\General_models\output_dipole_NoDielectric_WithCurrents")

# The threshold for splitting

THRESHOLD = 10000

# Subfolder names

SUBFOLDERS = ["models", "results", "S11_pictures"]


def setup_dirs():
    """Creates the destination skeleton."""

    for root in [DEST_LOW, DEST_HIGH]:

        for sub in SUBFOLDERS:
            (root / sub).mkdir(parents=True, exist_ok=True)


def split_data():
    # 1. Handle 'models' and 'results' (Folder-based)

    for sub in ["models", "results"]:

        source_path = SOURCE_DIR / sub

        if not source_path.exists(): continue

        for item in source_path.iterdir():

            if item.is_dir() and item.name.isdigit():
                folder_id = int(item.name)

                target_root = DEST_HIGH if folder_id >= THRESHOLD else DEST_LOW

                print(f"Moving folder {item.name} to {target_root / sub}")

                shutil.move(str(item), str(target_root / sub / item.name))

    # 2. Handle 'S11_pictures' (File-based)

    pic_source = SOURCE_DIR / "S11_pictures"

    if pic_source.exists():

        for file in pic_source.glob("S_parameters_*.png"):

            # Extract ID: "S_parameters_10001.png" -> "10001"

            try:

                file_id_str = file.stem.split('_')[-1]

                file_id = int(file_id_str)

                target_root = DEST_HIGH if file_id >= THRESHOLD else DEST_LOW

                print(f"Moving file {file.name} to {target_root / 'S11_pictures'}")

                shutil.move(str(file), str(target_root / "S11_pictures" / file.name))

            except (ValueError, IndexError):

                print(f"Skipping {file.name}: Could not parse ID")


if __name__ == "__main__":
    setup_dirs()

    split_data()

    print("\nProcessing complete!")