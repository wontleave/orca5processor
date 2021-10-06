import numpy as np
import time
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
    def __init__(self, root_folder_path,
                 post_process_type=None,
                 display_warning=False
                 ):
        self.orca5_objs = []
        self.folders_to_orca5 = {}  # Map each folder path to an Orca5 object or None if it is not possible
        self.orca5_in_pd = []
        # Get all the folders within the root
        all_folders = []
        read_root_folders(root_folder_path, all_folders)

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
                self.folders_to_orca5[folder] = None  # This folder has no valid output file
            else:
                self.folders_to_orca5[folder] = []
                for out_file in output_files:
                    self.folders_to_orca5[folder].append(Orca5(out_file))
            time_end = time.perf_counter()
            print(f"DONE in {time_end-time_start:.2f} sec")
        # Post processing according to the process type indicated
        if display_warning:
            self.display_warning()

        for key in self.folders_to_orca5:
            self.orca5_to_pd(self.folders_to_orca5[key])

        if post_process_type.lower() == "stationary":
            # Check if all the folders have the same number of orca5 object
            # All SP structures must be consistent with the FREQ or OPT structure
            ref_objs = {}  # The key is the root folder, the value is the OPT job_type_obj (only 1)
            for key in self.folders_to_orca5:
                ref_objs[key] = []
                no_of_opt = 0
                temp_obj_lists_ = self.folders_to_orca5[key]
                for obj in temp_obj_lists_:
                    if "OPT" in obj.job_type_objs:
                        ref_objs[key].append(obj)
                    elif "FREQ" in obj.job_type_objs:
                        if obj.job_type_objs["FREQ"].neligible_gradient:
                            ref_objs[key].append(obj)
                        else:
                            orig_cut_off = obj.job_type_objs["SP"].gradients_cut_off
                            loosen_cut_off = orig_cut_off * 10.0
                            diff_wrt_ref = loosen_cut_off - obj.job_type_objs["SP"].gradients
                            if np.any(diff_wrt_ref < 0.0):
                                max_grad = np.max(obj["SP"].gradients)
                                print(f"Max gradient is above loosen cut-off={loosen_cut_off:.5E}: {max_grad}")
                            else:
                                print(f"Loosen cut-off from {orig_cut_off:.2E} to {loosen_cut_off:.2E}"
                                      f" - successful - Adding {obj}")
                                ref_objs[key].append(obj)

                assert len(ref_objs[key]) == 1, f"{key} has {len(ref_objs[key])}. Please check the folder! Terminating"

    def orca5_to_pd(self, orca5_objs):
        """
        Combine information from all the Orca 5 objects into a pd dataframe

        :param orca5_objs:
        :type orca5_objs: [Orca5]
        :return:
        :rtype:
        """
        for item in orca5_objs:
            item.create_labelled_data()

    def display_warning(self):
        print("\n-------------------------------Displaying warnings detected------------------------------------------")
        for key in self.folders_to_orca5:
            print(f"Source folder: {key}")
            for item in self.folders_to_orca5[key]:
                print(f"{item.input_name} --- Warnings: {item.warnings}")
            print()
        print("---------------------------------END of warning(s) section---------------------------------------------")


if __name__ == "__main__":
    root_ = r"E:\TEST\Orca5Processor_tests\YZQ"
    orca5_ojbs = Orca5Processor(root_, display_warning=True, post_process_type="stationary")
