import numpy as np
import pandas as pd
import time
import json
import pickle
import argparse
import shutil
import matplotlib.pyplot as plt
from typing import Dict, List
from orca5_utils import calc_rmsd_ase_atoms_non_pbs
from writing_utils import modify_orca_input
from reading_utilis import read_root_folders
from orca5_utils import Orca5
from os import listdir, path
from pathlib import Path
from geo_optimization import Action, analyze_ts
from copy import copy
"""
Each valid Orca 5 output is converted into an Orca5 object
Version 1.0.0 -- 20210706 -- Focus on reading single point calculations for PiNN training
"""


# TODO Slow reading in some cases!
# TODO

class Orca5Processor:
    """
    Read a folder which contains a collection of ORCA5 jobs to create a list of ORCA5 object
    """

    def __init__(self, root_folder_path,
                 post_process_type=None,
                 display_warning=False,
                 delete_incomplete_job=False,
                 warning_txt_file=None,
                 error_handling=None
                 ):

        get_opt_steps = False
        for key in post_process_type:
            if "get_opt_step" in post_process_type[key]:
                if post_process_type[key]["get_opt_step"][0].lower() == "yes":
                    get_opt_steps = True

        self.root_folder_path = root_folder_path
        self.root_folder = Path(root_folder_path)
        self.folders_to_orca5 = {}  # Map each folder path to an Orca5 object or None if it is not possible
        self.orca5_in_pd = []
        # Get all the folders within the root
        all_folders = []
        read_root_folders(root_folder_path, all_folders)

        for key in post_process_type:
            if "grad_cut_off" in post_process_type[key]:
                grad_cut_off = float(post_process_type[key]["grad_cut_off"][0])  # value is a list.
            else:
                grad_cut_off = 1e-5

        # Determine which folder has the orca output

        for folder in all_folders:
            time_start = time.perf_counter()
            print(f"Working on {folder} ....", end="")
            files = listdir(folder)
            output_files = [path.join(folder, file) for file in files if
                            "wlm01" in file or
                            "coronab" in file or
                            "lic01" in file or
                            "hpc-mn1" in file
                            ]  # these are the supported extension

            # Each valid output file should correspond to an Orca 5 calculation.
            # Multiple output files can be present in a folder. For instance. OptTS + NumHess + multiple single point
            if "optts analysis" in post_process_type:
                allow_incomplete = True
            else:
                allow_incomplete = False

            if len(output_files) == 0:
                self.folders_to_orca5[folder] = None  # This folder has no valid output file
            else:
                self.folders_to_orca5[folder] = []
                for out_file in output_files:
                    time_temp = time.perf_counter()
                    self.folders_to_orca5[folder].append(Orca5(out_file, grad_cut_off, get_opt_steps=get_opt_steps,
                                                               allow_incomplete=allow_incomplete))
                    print(f"{out_file} took {time.perf_counter() - time_temp:.2f} s")
            time_end = time.perf_counter()
            print(f"DONE in {time_end - time_start:.2f} sec")

        # Remove Orca 5 objects that are flagged as incomplete with option to delete the corresponding output file
        self.remove_incomplete_job(delete_incomplete_job)

        # Post-processing according to the process type indicated
        if display_warning:
            self.display_warning(warning_txt_path=warning_txt_file)

        for key in post_process_type:
            if "temperature" in post_process_type[key]:
                temperature_ = float(post_process_type[key]["temperature"][0])
            else:
                temperature_ = 298.15

            if key.lower() == "stationary":
                self.process_stationary_pts(temperature_, grad_cut_off=grad_cut_off)
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
            elif key.lower() == "optts analysis":
                self.process_optts(post_process_type[key]["ts_int_coords"])

    @staticmethod
    def orca5_to_pd(orca5_objs_, temperature_=298.15):
        """

        :param orca5_objs_: Combine information from all the Orca 5 objects into a pd dataframe
        :type orca5_objs_:
        :param temperature_:
        :type temperature_:
        :return:
        :rtype:
        """
        for item in orca5_objs_:
            if not item.method.is_print_thermo:
                item.create_labelled_data(temperature=temperature_)

    @staticmethod
    def parse_pp_inp(path_to_pp_inp):
        """

        :param path_to_pp_inp:
        :type path_to_pp_inp: str
        :return:
        :rtype: Dict[str, List[str]]
        """
        with open(path_to_pp_inp) as f:
            lines = f.readlines()

        specifications = {}
        for line in lines:
            try:
                key, values = line.split("=")
            except ValueError:
                raise ValueError(f"{line} is in a readable format: key = x y z")
            values = values.split()

            if key.strip().lower() == "ts_atoms":  # TODO this is no longer in use?
                values = [int(item) for item in values]
            else:
                values = [item.strip() for item in values]
            specifications[key.strip()] = values.copy()
        return specifications

    def display_warning(self, warning_txt_path=False):
        """

        :param warning_txt_path: the path for the text file which the content of the warnings will be written to
        :type warning_txt_path: str
        :return:
        """

        content = "\n------------------------------Displaying warnings detected---------------------------------------"

        for key in self.folders_to_orca5:
            content += f"\nSource folder: {key}"
            try:
                for item in self.folders_to_orca5[key]:
                    content += f"\n{item.input_name} --- Warnings: {item.warnings}"
            except TypeError:
                raise TypeError(f"{key} is problematic")
            print()
        content += "\n--------------------------------END of warning(s) section----------------------------------------"
        content += "\n"
        print(content)
        if warning_txt_path is not None:
            with open(warning_txt_path, "w") as f:
                f.writelines(content)

    def remove_incomplete_job(self, delete_incomplete_job):
        """

        :param delete_incomplete_job: delete the file that corresponds to the incomplete Orca 5 job
        :type delete_incomplete_job: bool
        :return:
        :rtype:
        """

        empty_folders = []
        for key in self.folders_to_orca5:
            idx_of_complete_job = []
            try:
                for idx, obj in enumerate(self.folders_to_orca5[key]):
                    if not obj.completed_job:
                        if delete_incomplete_job:
                            to_be_removed = Path(obj.output_path)
                            print(f"Deleting {to_be_removed.resolve()}")
                            to_be_removed.unlink()
                    else:
                        idx_of_complete_job.append(idx)
                self.folders_to_orca5[key] = [self.folders_to_orca5[key][i] for i in idx_of_complete_job]
            except TypeError:
                print(f"{key} does not contain any Orca 5 objects ... skipping")
                empty_folders.append(key)

        if len(empty_folders) > 0:
            fixed_folders_to_orca5 = {}
            for key in self.folders_to_orca5:
                if key not in empty_folders:
                    fixed_folders_to_orca5[key] = self.folders_to_orca5[key]
            self.folders_to_orca5 = fixed_folders_to_orca5

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
        print_thermo_objs: Dict[str, List[Orca5]] = {}

        # After merge_thermo, all the printthermo obj will be removed
        # self.merge_thermo(temperature=temperature, grad_cut_off=grad_cut_off)

        for key in self.folders_to_orca5:
            ref_objs[key] = []
            print_thermo_objs[key] = []
            no_of_opt = 0
            temp_obj_lists_ = self.folders_to_orca5[key]
            for obj in temp_obj_lists_:
                if obj.method.is_print_thermo:
                    print_thermo_objs[key].append(obj)
                else:
                    if "OPT" in obj.job_type_objs and "FREQ" in obj.job_type_objs:
                        if len(ref_objs[key]) == 1:
                            # Check if the OPT object structure is the same as the current one
                            rmsd_ = calc_rmsd_ase_atoms_non_pbs(ref_objs[key][0].geo_from_xyz, obj.geo_from_xyz)
                            if rmsd_ > 1e-5:
                                ref_objs[key].append(obj)
                        else:
                            ref_objs[key].append(obj)

                    elif "FREQ" in obj.job_type_objs:
                        if obj.job_type_objs["FREQ"].negligible_gradient:
                            ref_objs[key].append(obj)
                        else:
                            loosen_cut_off = grad_cut_off * 2.0
                            diff_wrt_ref = loosen_cut_off - obj.job_type_objs["SP"].gradients
                            if np.any(diff_wrt_ref < 0.0):
                                max_grad = np.max(obj.job_type_objs["SP"].gradients)
                                print(f"Max gradient is above loosen cut-off={loosen_cut_off:.5E}: {max_grad}")
                            else:
                                print(f"Loosen cut-off from {grad_cut_off:.2E} to {loosen_cut_off:.2E}"
                                      f" - successful - Adding {obj.input_name}")
                                ref_objs[key].append(obj)
                                obj.job_type_objs["FREQ"].negligible_gradient = True

            assert len(ref_objs[key]) == 1, f"{key} has {len(ref_objs[key])}. Please check the folder! Terminating"
            # Merge the printthermo job to ref_obj
            if len(print_thermo_objs[key]) > 0:
               self.merge_thermo(print_thermo_objs[key], ref_objs[key][0])

        for key in self.folders_to_orca5:
            self.orca5_to_pd(self.folders_to_orca5[key], temperature_=temperature)

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
                    if item == "N Img Freq" or item == "Negative Freqs":
                        continue
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
        suffix = path.basename(self.root_folder_path)
        combined_df.to_excel(path.join(self.root_folder_path, f"stationary_{suffix}.xlsx"))

    def process_optts(self, int_coords_from_spec):
        """
        TODO We assume that each folder will have only one OPTTS
        :param int_coords_from_spec:
        :return:
        """
        optts_objs: Dict[str, List[Orca5]] = {}  # The key is the root folder, the value is the OPT job_type_obj
        failed_ts_objs = []  # stored the key where there is a failed OPTTS obj
        coords_in_ts_master = {}
        ts_modes_master = {}
        req_int_coords = None

        # Find all the ORCA 5 output that belongs to a optTS job
        for key in self.folders_to_orca5:
            temp_obj_lists_ = self.folders_to_orca5[key]
            assert len(temp_obj_lists_) == 1, f"Sorry, we don't support the presence of " \
                                              f"more than one TS job in a folder"
            for obj in temp_obj_lists_:
                optts_objs[key] = []
                if "TS" in obj.job_types:
                    optts_objs[key].append(obj)

        # Analyse: if last step of opt has the required ts mode we will use this geometry
        for key in optts_objs:
            for obj in optts_objs[key]:
                full_input_path, input_spec = analyze_ts(obj, int_coords_from_spec)
                if full_input_path is None:
                    failed_ts_objs.append(key)
                else:
                    modify_orca_input(full_input_path, **input_spec)

        # remove the *.pbs file
        print()
        for idx, key in enumerate(failed_ts_objs):
            if idx == 0:
                failed_root = Path(self.root_folder.joinpath("FAILED"))
            stem_of_ts_objs = Path(key).stem
            destination = failed_root.joinpath(stem_of_ts_objs)
            print(f"{idx + 1} ... Moving failed job {key} to {destination.resolve()}")
            shutil.move(key, destination.resolve())

    def merge_thermo(self, print_thermo_obj, ref_obj):
        """
        Populate the self.freqs Dict in ref_obj

        TODO: Have not implemented multiple PrintThermo
        TODO: Final single point is not present in PrintThermo
        Merge and delete Orca5 that belongs to a printthermochem job with Orca5 from a Freq or Opt+Freq job
        :param print_thermo_obj: the Orca5 object with the printthermo job
        :type print_thermo_obj: Orca5
        :param ref_obj: the reference obj
        :type ref_obj: Orca5
        :return: None
        :rtype:
        """

        for obj_ in print_thermo_obj:
            obj_.job_type_objs["FREQ"].elec_energy = ref_obj.job_type_objs["FREQ"].elec_energy
            temperature = obj_.job_type_objs["FREQ"].temp
            freq_ = obj_.job_type_objs["FREQ"]  # point to the Freq obj in the current printthermo object
            elec_energy = ref_obj.job_type_objs["FREQ"].elec_energy
            zpe_corr_energy = elec_energy + freq_.zero_pt_energy
            freq_.total_thermal_energy = zpe_corr_energy + freq_.total_thermal_correction
            freq_.total_enthalpy = freq_.total_thermal_energy + freq_.thermal_enthalpy_correction
            freq_.final_gibbs_free_energy = freq_.total_enthalpy + freq_.final_entropy_term

            freq_.thermo_data[temperature] = {"zero point energy": freq_.zero_pt_energy,
                                        "total thermal correction": freq_.total_thermal_correction,
                                        "thermal Enthalpy correction": freq_.thermal_enthalpy_correction,
                                        "total entropy correction": freq_.final_entropy_term,
                                        "zero point corrected energy": zpe_corr_energy,
                                        "total thermal energy": freq_.total_thermal_energy,
                                        "total enthalpy": freq_.total_enthalpy,
                                        "final gibbs free energy": freq_.final_gibbs_free_energy}

            ref_obj.freqs[temperature] = freq_


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", help="Path of the root folder")
    parser.add_argument("-t", "--pptype", help="Post processing type. Supported: \"single point\" \"stationary\" "
                                               "\"optts analysis \"")
    parser.add_argument("-i", "--ppinp", help="path to a text file to specific various options for post processing")
    args = parser.parse_args()

    if args.ppinp is None:
        ppinp_path = path.join(args.root, "ppinp.txt")
        temp_ = Path(ppinp_path)
        assert temp_.is_file(), f"{temp_.resolve()} is not a valid file or doesn't exist!"
    else:
        ppinp_path = args.ppinp

    spec = Orca5Processor.parse_pp_inp(ppinp_path)

    if args.pptype == "stationary":
        warning_path = None
        # We will not get individual optimization step for a stationary post-processing
        if spec["get_opt_step"].lower() == "yes":
            spec["get_opt_step"] = "no"

        if "warning" in spec:
            # spec["warning"] is a list of str
            if spec["warning"][0].lower() == "here":
                warning_path = path.join(args.root, "warning.txt")
            else:
                warning_path = spec["warning"][0]

        orca5_ojbs = Orca5Processor(args.root,
                                    display_warning=True,
                                    post_process_type={"stationary": spec},
                                    delete_incomplete_job=True,
                                    warning_txt_file=warning_path)
    elif args.pptype == "single point":
        orca5_objs = Orca5Processor(args.root, post_process_type={"single point":
                                                              {"to_pinn": ["pickle", "energy"],
                                                               "level_of_theory": 'CPCM(TOLUENE)/wB97X-V/def2-TZVPP'}})

    elif args.pptype == "optts analysis":
        orca5_objs = Orca5Processor(args.root, post_process_type={"optts analysis": spec})
