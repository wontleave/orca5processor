import numpy as np
from reading_utilis import read_root_folders
from orca5_utils import Orca5
from os import listdir, path
"""
Version 1.0.0 -- 20210706 -- Focus on reading single point calculations for PiNN training
"""


class Orca5Processor:
    """
    Read a folder which contains a collection of ORCA5 jobs to create a list of ORCA5 object
    """
    def __init__(self, root_folder_path):
        self.orca5_objs = []
        self.folders_to_orca5 = {}  # Map each folder path to an Orca5 object or None if it is not possible

        # Get all the folders within the root
        all_folders = []
        read_root_folders(root_folder_path, all_folders)

        # Determine which folder has the orca output
        for folder in all_folders:
            files = listdir(folder)
            output_files = []
            for file in files:
                output_files = [path.join(folder, file) for file in files if
                                "wlm01" in file or
                                "coronab" in file or
                                "lic01" in file or
                                "hpc-mn1" in file or
                                "out" in file]  # these are the supported extension

            # Each valid output file should correspond to an Orca 5 calculation.
            # Multiple output files can be present in a folder. For instance. OptTS + NumHess + multiple single point
            if len(output_files) == 0:
                self.folders_to_orca5[folder] = None  # This folder has no valid output file
            else:
                self.folders_to_orca5[folder] = []
                for out_file in output_files:
                    self.folders_to_orca5[folder].append(Orca5(out_file))


if __name__ == "__main__":
    root_ = r"E:\TEST\SP_tests"
    orca5_ojbs = Orca5Processor(root_)