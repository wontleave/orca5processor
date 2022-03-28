from orca5_utils import Orca5
from reading_utilis import read_root_folders
from os import listdir, path

import time

full_path = r"E:\TEST\automated_TS"

folders_to_orca5 = {}  # Map each folder path to an Orca5 object or None if it is not possible
orca5_in_pd = []
# Get all the folders within the root
all_folders = []
read_root_folders(full_path, all_folders)

# Determine which folder has the orca output
for folder in all_folders:
    time_start = time.perf_counter()
    print(f"Working on {folder} ....", end="")
    files = listdir(folder)
    output_files = []
    for file in files:
        output_files = [path.join(folder, file) for file in files if
                        "wlm01" in file or
                        "coronab" in file or
                        "lic01" in file or
                        "hpc-mn1" in file
                        ]  # these are the supported extension

    # Each valid output file should correspond to an Orca 5 calculation.
    # Multiple output files can be present in a folder. For instance. OptTS + NumHess + multiple single point
    if len(output_files) == 0:
        folders_to_orca5[folder] = None  # This folder has no valid output file
    else:
        folders_to_orca5[folder] = []
        for out_file in output_files:
            time_temp = time.perf_counter()
            folders_to_orca5[folder].append(Orca5(out_file))
            print(f"{out_file} took {time.perf_counter() - time_temp:.2f} s")
    time_end = time.perf_counter()
    print(f"DONE in {time_end - time_start:.2f} sec")

print()