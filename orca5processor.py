import numpy as np
import pandas as pd
import time
import json
import pickle
from typing import Dict, List
from orca5_utils import calc_rmsd_ase_atoms_non_pbs
from reading_utilis import read_root_folders
from orca5_utils import Orca5
from os import listdir, path
from pathlib import Path

"""
Each valid Orca 5 output is converted into an Orca5 object
Version 1.0.0 -- 20210706 -- Focus on reading single point calculations for PiNN training
"""


# TODO Slow reading in some cases!


class Orca5Processor:
    """
    Read a folder which contains a collection of ORCA5 jobs to create a list of ORCA5 object
    """

    def __init__(self, root_folder_path,
                 post_process_type=None,
                 display_warning=False,
                 delete_incomplete_job=False
                 ):
        self.root_folder_path = root_folder_path
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
                    time_temp = time.perf_counter()
                    self.folders_to_orca5[folder].append(Orca5(out_file))
                    print(f"{out_file} took {time.perf_counter() - time_temp:.2f} s")
            time_end = time.perf_counter()
            print(f"DONE in {time_end - time_start:.2f} sec")

        # Remove Orca 5 objects that are flagged as incomplete with option to delete the corresponding output file
        self.remove_incomplete_job(delete_incomplete_job)
        # Post processing according to the process type indicated
        if display_warning:
            self.display_warning()

        if "grad_cut_off" in post_process_type:
            grad_cut_off = post_process_type["grad_cut_off"]
        else:
            grad_cut_off = 1e-5

        for key in post_process_type:
            if key.lower() == "stationary":
                self.process_stationary_pts(post_process_type[key], grad_cut_off=grad_cut_off)
            elif key.lower() == "single point":
                if "to_pinn" in post_process_type[key]:
                    to_pinn = post_process_type[key]["to_pinn"]
                else:
                    to_pinn = None

                if "level_of_theory" in post_process_type[key]:
                    lvl_of_theory = post_process_type[key]["level_of_theory"]
                else:
                    raise ValueError("Please specific the level of theory!")

                self.process_single_pts(post_process_type[key], to_pinn=to_pinn, level_of_theory=lvl_of_theory)

    @staticmethod
    def orca5_to_pd(orca5_objs, temperature=298.15):
        """
        Combine information from all the Orca 5 objects into a pd dataframe
        TODO: Warning to txt file
        :param orca5_objs:
        :type orca5_objs: [Orca5]
        :return:
        :rtype:
        """
        for item in orca5_objs:
            item.create_labelled_data(temperature=temperature)

    def display_warning(self):
        print("\n-------------------------------Displaying warnings detected------------------------------------------")
        for key in self.folders_to_orca5:
            print(f"Source folder: {key}")
            try:
                for item in self.folders_to_orca5[key]:
                    print(f"{item.input_name} --- Warnings: {item.warnings}")
            except TypeError:
                raise TypeError(f"{key} is problematic")
            print()
        print("---------------------------------END of warning(s) section---------------------------------------------")

    def remove_incomplete_job(self, delete_incomplete_job):
        """

        :param delete_incomplete_job: delete the file that corresponds to the incomplete Orca 5 job
        :type delete_incomplete_job: bool
        :return:
        :rtype:
        """

        for key in self.folders_to_orca5:
            idx_of_complete_job = []
            for idx, obj in enumerate(self.folders_to_orca5[key]):
                if not obj.completed_job:
                    if delete_incomplete_job:
                        to_be_removed = Path(obj.output_path)
                        print(f"Deleting {to_be_removed.resolve()}")
                        to_be_removed.unlink()
                else:
                    idx_of_complete_job.append(idx)
            self.folders_to_orca5[key] = [self.folders_to_orca5[key][i] for i in idx_of_complete_job]

    def process_single_pts(self, to_json=None, to_pickle=None, to_pinn=None, level_of_theory=None):
        """

        :param to_json:
        :type to_json:
        :param to_pickle:
        :type to_pickle:
        :param to_pinn: Output the objects into an ASE atoms + energy and/or forces for PiNN training in pickle
        :type to_pinn: List[str]
        :param level_of_theory: specific the level of theory for the single point calculation. All if None
        :type level_of_theory: str
        :return:
        :rtype:
        """
        sp_objs: Dict[str, List[Orca5]] = {}
        for key in self.folders_to_orca5:
            sp_objs[key] = []
            temp_obj_lists_ = self.folders_to_orca5[key]
            for obj in temp_obj_lists_:
                if "OPT" not in obj.job_type_objs and "FREQ" not in obj.job_type_objs:
                    # TODO We will assume that this is a SP for now
                    sp_objs[key].append(obj)

        if to_pinn is not None:
            pinn_tuples = []
            get_forces = False
            if "pickle" in to_pinn:
                output_type = "pickle"
            else:
                raise ValueError("No valid output type!")

            assert level_of_theory is not None, "You need to specific the level of theory!"
            for key in sp_objs:
                for obj_ in sp_objs[key]:
                    if obj_.level_of_theory == level_of_theory:
                        if get_forces:
                            # TODO Engrad is not implemented yet
                            temp_ = obj_.job_type_objs["ENGRAD"]
                            pinn_tuples.append((temp_.geo_from_output, temp_.final_sp_energy, temp_.forces))
                        else:
                            temp_ = obj_.job_type_objs["SP"]
                            pinn_tuples.append((temp_.geo_from_output, temp_.final_sp_energy))

            if output_type == "pickle":
                pickle.dump(pinn_tuples, open(path.join(self.root_folder_path, "sp_objs.pickle"), "wb"))

    def process_stationary_pts(self, temperature=298.15, grad_cut_off=1e-5):
        """
        TODO Streamline gradient check?
        :param temperature:
        :type temperature: float
        :param grad_cut_off:
        :type grad_cut_off: float
        :return:
        :rtype:
        """
        # Check if all the folders have the same number of orca5 object
        ref_objs: Dict[str, List[Orca5]] = {}  # The key is the root folder, the value is the OPT job_type_obj

        self.merge_thermo(temperature=temperature, grad_cut_off=grad_cut_off)

        for key in self.folders_to_orca5:
            ref_objs[key] = []
            no_of_opt = 0
            temp_obj_lists_ = self.folders_to_orca5[key]
            for obj in temp_obj_lists_:
                if "OPT" in obj.job_type_objs and "FREQ" in obj.job_type_objs:
                    if len(ref_objs[key]) == 1:
                        # Check if the OPT object structure is the same as the current one
                        rmsd_ = calc_rmsd_ase_atoms_non_pbs(ref_objs[key][0].geo_from_xyz, obj.geo_from_xyz)
                        if rmsd_ > 1e-5:
                            ref_objs[key].append(obj)
                    else:
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
                                  f" - successful - Adding {obj.input_name}")
                            ref_objs[key].append(obj)
                            obj.job_type_objs["FREQ"].neligible_gradient = True

            assert len(ref_objs[key]) == 1, f"{key} has {len(ref_objs[key])}. Please check the folder! Terminating"

        for key in self.folders_to_orca5:
            self.orca5_to_pd(self.folders_to_orca5[key], temperature=temperature)

        # All SP structures must be consistent with the structure in the reference obj
        sp_objs: Dict[str, List[Orca5]] = {}
        for key in self.folders_to_orca5:
            sp_objs[key] = []
            temp_obj_lists_ = self.folders_to_orca5[key]
            for obj in temp_obj_lists_:
                if "OPT" not in obj.job_type_objs and "FREQ" not in obj.job_type_objs:
                    # TODO We will assume that this is a SP for now
                    rmsd_ = calc_rmsd_ase_atoms_non_pbs(ref_objs[key][0].geo_from_xyz, obj.geo_from_xyz)
                    if rmsd_ < grad_cut_off:
                        sp_objs[key].append(obj)

        # Collect the self.labelled_data and make the pandas df
        labeled_data: Dict[str, Dict[str, str]] = {}
        row_labels = []
        counter = 0
        thermo_corr_labels = np.array(["ZPE", "thermal", "thermal_enthalpy_corr", "final_entropy_term"])
        for key in ref_objs:
            key_from_base = path.basename(key)
            labeled_data[key_from_base] = ref_objs[key][0].labelled_data
            for obj_ in sp_objs[key]:
                thermo_corr_sp = {}
                corr = 0.0
                # For each SP Orca 5 object, we will add the necessary thermochemistry corr to the SP elec energy
                labeled_data[key_from_base] = {**labeled_data[key_from_base], **obj_.labelled_data}
                for item in ref_objs[key][0].labelled_data:
                    opt_theory, thermo_corr_type = item.split("--")
                    if np.any(np.char.equal(thermo_corr_type, thermo_corr_labels)):
                        for sp_key in obj_.labelled_data:
                            try:
                                corr += ref_objs[key][0].labelled_data[item]
                                corr_value = obj_.labelled_data[sp_key] + corr
                                thermo_corr_sp[f"{thermo_corr_type} -- {sp_key}//{opt_theory}"] = corr_value
                            except TypeError:
                                raise TypeError(f"key:{key} item:{item} sp keu:{sp_key} failed. corr={corr}")
                labeled_data[key_from_base] = {**labeled_data[key_from_base], **thermo_corr_sp}
            if counter == 0:
                row_labels = list(labeled_data[key_from_base].keys())
                counter += 1
        combined_df = pd.DataFrame(labeled_data).reindex(row_labels)
        combined_df.to_excel(path.join(self.root_folder_path, "stationary.xlsx"))

    def merge_thermo(self, temperature=298.15, grad_cut_off=1e-5):
        """
        Merge and delete Orca5 that belongs to a printthermochem job with Orca5 from a Freq or Opt+Freq job
        :param temperature: the required temperature
        :type temperature: float
        :param grad_cut_off: A manual gradient cut-off for SP gradient
        :type grad_cut_off: float
        :return:
        :rtype:
        """
        for key in self.folders_to_orca5:
            ref_objs: List[Orca5] = []
            # Find the reference object
            for orca5_obj in self.folders_to_orca5[key]:
                if "FREQ" in orca5_obj.job_type_objs:
                    if "OPT" in orca5_obj.job_type_objs or orca5_obj.job_type_objs["FREQ"].neligible_gradient:
                        if not orca5_obj.method.is_print_thermo:
                            ref_objs.append(orca5_obj)
                    elif not orca5_obj.job_type_objs["FREQ"].neligible_gradient:
                        diff_wrt_ref = grad_cut_off - orca5_obj.job_type_objs["SP"].gradients
                        if np.any(diff_wrt_ref < 0.0):
                            print(f"Max gradient is above loosen cut-off={grad_cut_off:.5E} ... termininating")
                        else:
                            ref_objs.append(orca5_obj)

            assert len(ref_objs) != 0, f"We cannot find a valid Orca 5 object with a valid FREQ " \
                                       f"or OPT+FREQ job for {key}!"
            orca5_obj_to_exclude = []

            for idx, print_thermo_obj in enumerate(self.folders_to_orca5[key]):
                if print_thermo_obj.method.is_print_thermo:
                    orca5_obj_to_exclude.append(idx)
                    for ref_obj in ref_objs:
                        temp_ = print_thermo_obj.job_type_objs["FREQ"]
                        elec_energy = ref_obj.job_type_objs["FREQ"].elec_energy
                        zpe_corr_energy = elec_energy + temp_.thermo_data[temperature]["zero point energy"]
                        total_thermal_energy = zpe_corr_energy + \
                                               temp_.thermo_data[temperature]["total thermal correction"]
                        total_enthalpy = total_thermal_energy + \
                                         temp_.thermo_data[temperature]["thermal Enthalpy correction"]
                        final_gibbs_free_energy = total_enthalpy + \
                                                  temp_.thermo_data[temperature]["total entropy correction"]

                        ref_obj_thermo_data = ref_obj.job_type_objs["FREQ"].thermo_data
                        assert temperature not in ref_obj_thermo_data, f"The requested temperature is already present!"
                        ref_obj_thermo_data[temperature] = {}
                        ref_obj_thermo_data[temperature]["zero point energy"] = \
                            temp_.thermo_data[temperature]["zero point energy"]
                        ref_obj_thermo_data[temperature]["total thermal correction"] = \
                            temp_.thermo_data[temperature]["total thermal correction"]
                        ref_obj_thermo_data[temperature]["thermal Enthalpy correction"] = \
                            temp_.thermo_data[temperature]["thermal Enthalpy correction"]
                        ref_obj_thermo_data[temperature]["total entropy correction"] = \
                            temp_.thermo_data[temperature]["total entropy correction"]

                        ref_obj_thermo_data[temperature]["total thermal energy"] = total_thermal_energy
                        ref_obj_thermo_data[temperature]["total enthalpy"] = total_enthalpy
                        ref_obj_thermo_data[temperature]["final gibbs free energy"] = final_gibbs_free_energy

            req_orca5_objs = []

            for i in range(len(self.folders_to_orca5[key])):
                if i not in orca5_obj_to_exclude:
                    req_orca5_objs.append(self.folders_to_orca5[key][i])

            self.folders_to_orca5[key] = req_orca5_objs


if __name__ == "__main__":
    # root_ = r"E:\TEST\Orca5Processor_tests\YZQ\DEBUG"
    root_ = r"E:\vBoxShared\PiNN_database\TCH_catalysts\YZQ-PNtBU-Steglich-Step1TS"
    # orca5_ojbs = Orca5Processor(root_, display_warning=True,
    #                             post_process_type={"stationary": 253.15, "grad_cut_off": 1e-4},
    #                             delete_incomplete_job=True)

    orca5_objs = Orca5Processor(root_, post_process_type={"single point":
                                                              {"to_pinn": ["pickle", "energy"],
                                                               "level_of_theory": 'CPCM(TOLUENE)/wB97X-V/def2-TZVPP'}})
