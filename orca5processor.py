import numpy as np
import pandas as pd
import time
import yaml
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

class Orca5Processor:
    """
    Read a folder which contains a collection of ORCA5 jobs to create a list of ORCA5 object
    TODO singularity_scratch will only affect process_optts post-procesing for now
    """

    def __init__(self, root_folder_path,
                 post_process_type=None,
                 display_warning=False,
                 delete_incomplete_job=False,
                 warning_txt_file=None,
                 error_handling=None,
                 singularity_scratch=None
                 ):
        """

        :param root_folder_path:
        :param post_process_type:
        :param display_warning:
        :param delete_incomplete_job:
        :param warning_txt_file:
        :param error_handling:
        :param singularity_scratch: if this is not None, then ORCA5 is ran from a singularity container and
                                    this variable will be preprended to all filename in the orca input
        :type singularity_scratch: str
        """

        get_opt_steps = False
        for key in post_process_type:
            if "get_opt_step" in post_process_type[key]:
                if post_process_type[key]["get_opt_step"][0].lower() == "yes":
                    get_opt_steps = True

        self.root_folder_path = root_folder_path
        self.root_folder = Path(root_folder_path)
        self.folders_to_orca5 = {}  # Map each folder path to an Orca5 object or None if it is not possible
        self.orca5_in_pd = []
        self.singularity_scratch = singularity_scratch
        # Get all the folders within the root
        all_folders = []
        read_root_folders(root_folder_path, all_folders)

        for key in post_process_type:
            if "grad_cut_off" in post_process_type[key]:
                grad_cut_off = post_process_type[key]["grad_cut_off"]
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
                temperature_ = post_process_type[key]["temperature"]
            else:
                temperature_ = 298.15

            if "calc_relative_en" in post_process_type[key]:
                calc_relative_en = post_process_type[key]["calc_relative_en"]
            else:
                calc_relative_en = False

            if "filter_and_copy" in post_process_type[key]:
                filter_and_copy = post_process_type[key]["filter_and_copy"]
            else:
                filter_and_copy = None

            if "boltzmann_weighted_en" in post_process_type[key]:
                boltzmann_weighted_en = post_process_type[key]["boltzmann_weighted_en"]

            if key.lower() == "stationary":
                self.process_stationary_pts(temperature_, grad_cut_off=grad_cut_off, calc_relative_en=calc_relative_en,
                                            filter_and_copy=filter_and_copy,
                                            boltzmann_weighted_en=boltzmann_weighted_en)
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
        Parse the yaml configuration file

        :param path_to_pp_inp:
        :type path_to_pp_inp: str
        :return:
        :rtype: Dict[str, List[str]]
        """
        with open(path_to_pp_inp) as f:
            specifications = yaml.safe_load(f)

        # for key, value in specifications:

        # specifications = {}
        # for line in lines:
        #     try:
        #         key, values = line.split("=")
        #     except ValueError:
        #         raise ValueError(f"{line} is in a readable format: key = x y z")
        #     values = values.split()
        #
        #     if key.strip().lower() == "ts_atoms":  # TODO this is no longer in use?
        #         values = [int(item) for item in values]
        #     else:
        #         values = [item.strip() for item in values]
        #     specifications[key.strip()] = values.copy()
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

    def process_stationary_pts(self, temperature=298.15, grad_cut_off=1e-5, calc_relative_en=False,
                               boltzmann_weighted_en=False, filter_and_copy=None):
        """
        TODO Streamline gradient check?
        :param temperature:
        :type temperature: float
        :param grad_cut_off:
        :type grad_cut_off: float
        :param calc_relative_en: Flag to control whether the relative energy
                                of each struct is calculated relative to the minimum
        :type: calc_relative_en: bool
        :param boltzmann_weighted_en: calculate the Boltzmann-weighted energy at the given temperature
        :type boltzmann_weighted_en: bool
        :param filter_and_copy: Control whether to select those structures that is less than the given cut-off and
                                copy them to a new folder. Can only be triggered if calc_relative_en is True.
                                It contains the level of theory used for filtering the dataframe and the cut-off
        :type filter_and_copy: Dict[str, float]
        :return:
        :rtype:
        """
        # Check if all the folders have the same number of orca5 object
        ref_objs: Dict[str, List[Orca5]] = {}  # The key is the root folder, the value is the OPT job_type_obj
        print_thermo_objs: Dict[str, List[Orca5]] = {}

        # After merge_thermo, all the printthermo obj will be removed
        # self.merge_thermo(temperature=temperature, grad_cut_off=grad_cut_off)
        cannot_proceed = False
        unacceptable_references = {}
        missing_temp = {}
        gradient_too_large = {}

        for key in self.folders_to_orca5:
            ref_objs[key] = []
            print_thermo_objs[key] = []
            temp_obj_lists_ = self.folders_to_orca5[key]
            for obj in temp_obj_lists_:
                if obj.method.is_print_thermo:
                    print_thermo_objs[key].append(obj)
                else:
                    if "OPT" in obj.job_type_objs and "FREQ" in obj.job_type_objs:
                        if len(ref_objs[key]) == 1:
                            # Check if the OPT object structure is the same as the current one
                            rmsd_ = calc_rmsd_ase_atoms_non_pbs(ref_objs[key][0].geo_from_xyz, obj.geo_from_xyz)
                            if rmsd_ > grad_cut_off:
                                ref_objs[key].append(obj)
                        else:
                            ref_objs[key].append(obj)

                    elif "FREQ" in obj.job_type_objs:
                        if obj.job_type_objs["FREQ"].negligible_gradient:
                            ref_objs[key].append(obj)
                        else:
                            loosen_cut_off = grad_cut_off * 4.0
                            diff_wrt_ref = loosen_cut_off - obj.job_type_objs["SP"].gradients
                            if np.any(diff_wrt_ref < 0.0):
                                max_grad = np.max(obj.job_type_objs["SP"].gradients)
                                print(f"Max gradient is above loosen cut-off={loosen_cut_off:.5E}: {max_grad}")
                                # TODO Considered as failed?
                            else:
                                print(f"Loosen cut-off from {grad_cut_off:.2E} to {loosen_cut_off:.2E}"
                                      f" - successful - Adding {obj.input_name}")
                                ref_objs[key].append(obj)
                                obj.job_type_objs["FREQ"].negligible_gradient = True

            if len(ref_objs[key]) != 1:
                cannot_proceed = True
                unacceptable_references[key] = len(ref_objs[key])

            # Merge the printthermo job to ref_obj
            if len(print_thermo_objs[key]) > 0 and not cannot_proceed:
                Orca5Processor.merge_thermo(print_thermo_objs[key], ref_objs[key][0])

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
                    try:
                        rmsd_ = calc_rmsd_ase_atoms_non_pbs(ref_objs[key][0].geo_from_xyz, obj.geo_from_xyz)
                    except IndexError:
                        if cannot_proceed:
                            continue
                        else:
                            raise IndexError(f"{key=}")
                    if rmsd_ < grad_cut_off:
                        sp_objs[key].append(obj)
                    else:
                        cannot_proceed = True
                        gradient_too_large[key] = rmsd_
                else:
                    for warning in obj.warnings:
                        if "THERMOCHEMISTRY:" in warning and "OPT" not in obj.job_type_objs:
                            cannot_proceed = True
                            missing_temp[key] = temperature

        if not cannot_proceed:
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
                                    if thermo_corr_type.upper() == "ZPE":
                                        _label = "ZPE_corrected elec_energy"
                                    elif thermo_corr_type.upper() == "THERMAL":
                                        _label = "total_thermal_energy"
                                    elif thermo_corr_type.upper() == "THERMAL_ENTHALPY_CORR":
                                        _label = "total_enthalpy"
                                    elif thermo_corr_type.upper() == "FINAL_ENTROPY_TERM":
                                        _label = "final_gibbs_free_energy"

                                    thermo_corr_sp[f"{_label} -- {sp_key}//{opt_theory}"] = corr_value

                                except TypeError:
                                    raise TypeError(f"key:{key} item:{item} sp keu:{sp_key} failed. corr={corr}")

                    labeled_data[key_from_base] = {**labeled_data[key_from_base], **thermo_corr_sp}

                if counter == 0:
                    row_labels = list(labeled_data[key_from_base].keys())
                    counter += 1
                elif counter > 0:
                    row_labels_check = list(labeled_data[key_from_base].keys())
                    # Failsafe for missing SP in some ORCA obj
                    if len(row_labels_check) > len(row_labels):
                        row_labels = row_labels_check.copy()

                # else:
                ## For DEBUG only
                #     temp_labels = list(labeled_data[key_from_base].keys())
                #     for idx, label in enumerate(temp_labels):
                #         if row_labels[idx] == label:
                #             same_label = True
                #         else:
                #             same_label = False
                #         print(f"{idx=} {row_labels[idx]} vs. {label} -- {same_label}")

            combined_df = pd.DataFrame(labeled_data).reindex(row_labels)
            suffix = path.basename(self.root_folder_path)
            has_relative_en = False
            if calc_relative_en:
                # Filter unwanted rows by query
                expr = 'index != "N Img Freq" and index != "Negative Freqs"'
                expr += 'and not index.str.contains("--ZPE")'
                expr += 'and not index.str.contains("--thermal")'
                expr += 'and not index.str.contains("--thermal_enthalpy_corr")'
                expr += 'and not index.str.contains("--final_entropy_term")'

                combined_df["min"] = combined_df.query(expr).min(axis=1)

                # Calculate the relative energy relative to the lowest
                rel_en = None
                col_names = []
                for column in combined_df:
                    if column != "min":
                        col_names.append(column)
                        if rel_en is None:
                            rel_en = (combined_df.query(expr)[column] - combined_df.query(expr)["min"]) * 627.509
                        else:
                            rel_en = pd.concat([rel_en,
                                                (combined_df.query(expr)[column] - combined_df.query(expr)[
                                                    "min"]) * 627.509],
                                               axis=1)

                rel_en.set_axis(col_names, axis=1, inplace=True)

                if boltzmann_weighted_en:
                    boltzmann_partition = -4184.0 * rel_en.astype(float) / (8.31446261815324 * temperature)
                    boltzmann_partition = boltzmann_partition.apply(np.exp)
                    total_pop = boltzmann_partition.apply(np.sum, axis=1)
                    boltzmann_weights = boltzmann_partition.div(total_pop, axis=0)
                    req_columns = combined_df.columns.values[0:-1]
                    boltz_weighted_en = boltzmann_weights * combined_df.loc[boltzmann_weights.index.values, req_columns]
                    boltz_weighted_en = boltz_weighted_en.apply(np.sum, axis=1)
                    boltz_weighted_en.rename("Boltzmann weighted energy", inplace=True)
                    boltzmann_df = pd.concat([boltzmann_weights, boltz_weighted_en.T], axis=1)
                    combined_df = pd.concat([combined_df, rel_en, boltzmann_df], axis=0)
                    has_relative_en = True
                    has_boltzmann_weights = True

                else:
                    combined_df = pd.concat([combined_df, rel_en])
                    has_relative_en = True

            combined_df.to_excel(path.join(self.root_folder_path, f"stationary_{suffix}.xlsx"))
            if filter_and_copy:
                if not has_relative_en:
                    print("Filter_and_copy requested, but ... please calculate relative energies first ...")
                else:
                    # Duplicates will exist. Idx 0: absolute energies Idx 1: relative energies
                    filter_cut_off = filter_and_copy["cut_off"]
                    sliced = combined_df.loc[filter_and_copy["level_of_theory"]].iloc[1]
                    req_folder_names = sliced[sliced < filter_cut_off].index.values
                    print(f"{len(req_folder_names)} are within the required cut-off values of {filter_cut_off} kcal/mol")
                    print(req_folder_names)
                    filtered_path = path.join(self.root_folder_path, "FILTERED")
                    assert not Path(filtered_path).is_dir(), f"Sorry {filtered_path} already exists ... please check"
                    Path(filtered_path).mkdir()

                    for key in ref_objs:
                        name = path.basename(key)
                        if name in req_folder_names:
                            filtered_obj_path = path.join(filtered_path, name)
                            filtered_obj_path_obj = Path(filtered_obj_path)
                            filtered_obj_path_obj.mkdir()
                            xyz_full_path = filtered_obj_path_obj / "structure.xyz"
                            print(f"Writing {xyz_full_path}", end=" --- ")
                            ref_objs[key][0].geo_from_xyz.write(xyz_full_path.resolve())
                            print("DONE!")

        else:

            for idx, key in enumerate(unacceptable_references.keys()):
                if idx == 0:
                    print("Problem(s) found. The following reference(s) is/are unacceptable:")
                print(f"{key} -- {unacceptable_references[key]}")

            for idx, key in enumerate(missing_temp.keys()):
                if idx == 0:
                    print("Problem(s) found. The following calculations are missing the required temperature")
                print(f"{key} -- {missing_temp[key]}")

            for idx, key in enumerate(gradient_too_large.keys()):
                if idx == 0:
                    print("Problem(s) found. The following SP geometry has RMSD that is larger than the reference")
                print(f"{key} -- {gradient_too_large[key]}")

    def process_optts(self, int_coords_from_spec):
        """
        TODO We assume that each folder will have only one OPTTS
        :param int_coords_from_spec:
        :return:
        """
        optts_objs: Dict[str, List[Orca5]] = {}  # The key is the root folder, the value is the OPT job_type_obj
        failed_ts_objs = []  # stored the key where there is a failed OPTTS obj
        converged_ts_objs = []

        # Find all the ORCA 5 output that belongs to a optTS job
        for key in self.folders_to_orca5:
            temp_obj_lists_ = self.folders_to_orca5[key]
            optts_objs[key] = []
            for obj in temp_obj_lists_:
                if "TS" in obj.job_types:
                    optts_objs[key].append(obj)

            assert len(optts_objs[key]) == 1, f"Sorry, we don't support the presence of " \
                                              f"more than one TS job or no TS job in a folder"
        # Analyse: if last step of opt has the required ts mode we will use this geometry
        for key in optts_objs:
            for obj in optts_objs[key]:
                if obj.job_type_objs["OPT"].converged:
                    converged_ts_objs.append(key)
                else:
                    full_input_path, input_spec = analyze_ts(obj, int_coords_from_spec, recalc_hess_min=0)

                    if full_input_path is None:
                        failed_ts_objs.append(key)
                    else:
                        if self.singularity_scratch is not None:
                            input_spec["xyz_path"] = "/" + self.singularity_scratch + "/" + \
                                                     input_spec["xyz_path"]  # linux only
                        modify_orca_input(full_input_path, **input_spec)

        for idx, key in enumerate(converged_ts_objs):
            if idx == 0:
                converged_root = Path(self.root_folder.joinpath("CONVERGED"))
            stem_of_ts_objs = Path(key).stem
            destination = converged_root.joinpath(stem_of_ts_objs)
            print(f"{idx + 1} ... Moving converged job {key} to {destination.resolve()}")
            shutil.move(key, destination.resolve())
        print()
        for idx, key in enumerate(failed_ts_objs):
            if idx == 0:
                failed_root = Path(self.root_folder.joinpath("FAILED"))
            stem_of_ts_objs = Path(key).stem
            destination = failed_root.joinpath(stem_of_ts_objs)
            print(f"{idx + 1} ... Moving failed job {key} to {destination.resolve()}")
            shutil.move(key, destination.resolve())

    @staticmethod
    def merge_thermo(print_thermo_obj, ref_obj):
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
        ppinp_path = path.join(args.root, "ppinp.yaml")
        temp_ = Path(ppinp_path)
        assert temp_.is_file(), f"{temp_.resolve()} is not a valid file or doesn't exist!"
    else:
        ppinp_path = args.ppinp

    spec = Orca5Processor.parse_pp_inp(ppinp_path)
    if "singularity_scratch" in spec:
        singularity_scratch = spec["singularity_scratch"][0]
    else:
        singularity_scratch = None

    if args.pptype == "stationary":
        warning_path = None
        # We will not get individual optimization step for a stationary post-processing
        if "get_opt_step" in spec:
            if spec["get_opt_step"].lower() == "yes":
                print(f"get_opt_step was set to yes ... disabling it ...")
                spec["get_opt_step"] = "no"

        if "warning" in spec:
            # spec["warning"] is a list of str
            if spec["warning"][0].lower() == "here":
                warning_path = path.join(args.root, "warning.txt")
            else:
                warning_path = spec["warning"][0]

        orca5_objs = Orca5Processor(args.root,
                                    display_warning=True,
                                    post_process_type={"stationary": spec},
                                    delete_incomplete_job=True,
                                    warning_txt_file=warning_path,
                                    singularity_scratch=singularity_scratch)

    elif args.pptype == "single point":
        # orca5_objs = Orca5Processor(args.root, post_process_type={"single point":
        #                                                       {"to_pinn": ["pickle", "energy"],
        #                                                        "level_of_theory": 'CPCM(TOLUENE)/wB97X-V/def2-TZVPP'}})

        # TODO flexible parameters input from ppinp
        orca5_objs = Orca5Processor(args.root, post_process_type={"single point":
                                                                      {"to_pinn": None,
                                                                       "level_of_theory": 'CPCM(TOLUENE)/r2scan-3c'}})
        print()
    elif args.pptype == "optts analysis":
        orca5_objs = Orca5Processor(args.root,
                                    post_process_type={"optts analysis": spec},
                                    singularity_scratch=singularity_scratch)
