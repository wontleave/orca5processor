import glob
import subprocess
import time
import numpy as np
from os import path, remove, chdir
from pathlib import Path


class Step:
    def __init__(self, energy_change, rms_gradient, max_gradient, rms_step, max_step, red_int_coords):
        self.energy_change = energy_change
        self.rms_gradient = rms_gradient
        self.max_gradient = max_gradient
        self.rms_step = rms_step
        self.max_step = max_step
        self.red_int_coords = red_int_coords
        self.condensed_data = None  # indices to ts mode for analysis
        self.coords_in_ts = []
        self.int_coord_to_idx = {}

    def find_ts_mode(self):
        """

        """
        condensed_data = {}
        for idx, coord_index in enumerate(self.red_int_coords):
            req_indices = self.red_int_coords[coord_index].get_indices()
            self.int_coord_to_idx[req_indices] = idx + 1
            ts_mode = self.red_int_coords[coord_index].ts_mode
            if ts_mode > 0.0:
                self.coords_in_ts.append(req_indices)
            condensed_data[req_indices] = ts_mode
        self.condensed_data = condensed_data


class Opt:
    def __init__(self):
        self.distance_old = None
        self.distance_new = None
        self.gradient = None
        self.step = None
        self.ts_mode = 0.0
        self.constrained = False


class OptBond(Opt):
    def __init__(self, element1, element2, index1, index2):
        Opt.__init__(self)
        self.element1 = element1.strip()
        self.element2 = element2.strip()
        self.index1 = int(index1)
        self.index2 = int(index2)

    def get_indices(self):
        return self.index1, self.index2


class OptAngle(Opt):
    def __init__(self, element1, element2, element3, index1, index2, index3):
        Opt.__init__(self)
        self.element1 = element1.strip()
        self.element2 = element2.strip()
        self.element3 = element3.strip()
        self.index1 = int(index1)
        self.index2 = int(index2)
        self.index3 = int(index3)

    def get_indices(self):
        return self.index1, self.index2, self.index3


class OptTorsion(Opt):
    def __init__(self, element1, element2, element3, element4,
                 index1, index2, index3, index4, part=None):
        """

        :param element1:
        :param element2:
        :param element3:
        :param element4:
        :param index1:
        :param index2:
        :param index3:
        :param index4:
        :param part: some dihedrals are splitted into 2 parts
        """
        Opt.__init__(self)
        self.element1 = element1.strip()
        self.element2 = element2.strip()
        self.element3 = element3.strip()
        self.element4 = element4.strip()
        self.part = part
        self.index1 = int(index1)
        self.index2 = int(index2)
        self.index3 = int(index3)
        self.index4 = int(index4)

    def get_indices(self):
        return self.index1, self.index2, self.index3, self.index4


class Action:
    """
    Class to control what to do for the next cycle of TS search
    Current support action:
        1) use last geometry - changes in internal coordinates which are involved in TS is decreasing and
           their TS modes > 0
        2) use x th geoemetry - if TS mode disappears. Will also decrease recalc_hess by 50%
        3) exclude - no viable step to continue the optTS search
    """
    def __init__(self):
        self.exclude = False
        self.partial_req_ts_mode = False
        self.geo_to_use = -1  # Last geometry by default

    @staticmethod
    def delete_pbs(root_path):
        pbs_full_path = path.join(root_path, "*.pbs")
        files_to_delete = glob.glob(pbs_full_path)
        for file in files_to_delete:
            remove(file)


def analyze_ts(orca5_obj, int_coords_from_spec):
    """

    :param orca5_obj:
    :type orca5_obj: Orca5
    :param int_coords_from_spec: the spec.txt file will provide the required internal coordinates for the TS
    :type int_coords_from_spec:
    :return:
    """
    coords_in_ts_master = {}
    ts_modes_master = {}
    req_int_coords = None

    for idx, step in enumerate(orca5_obj.job_type_objs["OPT"].opt_steps):
        step.find_ts_mode()
        for i in step.coords_in_ts:
            value = step.red_int_coords[step.int_coord_to_idx[i]].distance_old
            ts_mode = step.condensed_data[i]
            if i not in coords_in_ts_master:
                coords_in_ts_master[i] = []
                ts_modes_master[i] = []
            else:
                coords_in_ts_master[i].append(value)
                ts_modes_master[i].append(ts_mode)

    orca5_obj.job_type_objs["OPT"].steps_to_pd(int_coords_from_spec)
    if req_int_coords is None:
        req_int_coords = orca5_obj.job_type_objs["OPT"].req_int_coords_idx
    ts_mode_only_df = orca5_obj.job_type_objs["OPT"].req_data_pd.xs("TS mode", level=1, axis=1)
    all_req_int_coords_ts_mode_at_step = ts_mode_only_df.all(axis=1)
    any_req_int_coords_ts_mode_at_step = ts_mode_only_df.any(axis=1)

    action = Action()

    need_partial = True
    for i in reversed(all_req_int_coords_ts_mode_at_step.index.values):
        if all_req_int_coords_ts_mode_at_step[i]:
            action.geo_to_use = i
            need_partial = False
            break

    if need_partial:
        for i in reversed(any_req_int_coords_ts_mode_at_step.index.values):
            if any_req_int_coords_ts_mode_at_step[i]:
                action.geo_to_use = i
                action.partial_req_ts_mode = True
                break

    if action.geo_to_use < 0:
        action.exclude = True

    orca5_obj.job_type_objs["OPT"].action = action

    for key in coords_in_ts_master:
        show = False
        if key in req_int_coords:
            show = True
        if show:
            freq, bin = np.histogram(coords_in_ts_master[key], bins="auto")
            max_freq_idx = np.argmax(freq)
            left = bin[max_freq_idx]
            right = bin[max_freq_idx + 1]
            mid_pt = (left + right) / 2
            print(f"{key} -- {freq[max_freq_idx]} -- between {left:.4f} and {right:.4f} -- midpoint = {mid_pt:.4f}")

    req_obj = orca5_obj.job_type_objs["OPT"].action
    values = orca5_obj.job_type_objs["OPT"].req_data_pd.xs("value", level=1, axis=1)
    orca5_obj.job_type_objs["OPT"].values_diff = values.diff().abs()

    if req_obj.exclude:
        print(f"FAILED No possible candidates")
        for int_coords in values:
            print(f"{int_coords} = {values[int_coords].min()} and {values[int_coords].max()}", end="  ")
        return None, None
    else:
        # Set up the trajectory file
        if "QMMM" in orca5_obj.job_types:
            trj_filename = orca5_obj.root_name + ".activeRegion_trj.xyz"
        else:
            trj_filename = orca5_obj.root_name + "_trj.xyz"
        trj_full_path = path.join(orca5_obj.root_path, trj_filename)
        orca5_obj.job_type_objs["OPT"].get_opt_trj(trj_full_path)

        print(orca5_obj.job_type_objs["OPT"].req_data_pd)
        # TODO get recalc_hess of previous iteration
        # TODO adapt recalc_hess for partial internal coordinates
        if req_obj.partial_req_ts_mode:
            print(f"Not all internal coordinate(s) have TS mode -- Geometry {req_obj.geo_to_use} will be used")
            input_path = orca5_obj.root_name + ".inp"
            full_input_path = path.join(orca5_obj.root_path, input_path)
            xyz_name = orca5_obj.root_name + ".xyz"
            xyz_full_path = path.join(orca5_obj.root_path, xyz_name)
            orca5_obj.job_type_objs["OPT"].opt_trj[req_obj.geo_to_use - 1].write(xyz_full_path)
            return full_input_path, {"recalc_hess": 5, "xyz_path": xyz_name}
        else:
            print(f"All internal coordinates have TS mode -- Geometry {req_obj.geo_to_use} will be used")

            xyz_name = orca5_obj.root_name + ".xyz"
            xyz_full_path = path.join(orca5_obj.root_path, xyz_name)
            orca5_obj.job_type_objs["OPT"].opt_trj[req_obj.geo_to_use - 1].write(xyz_full_path)

            # Determine by how many steps recalc_hess will be increased from the current
            recalc_hess = req_obj.geo_to_use

            if (orca5_obj.job_type_objs["OPT"].values_diff.iloc[req_obj.geo_to_use - 1] < 0.1).all():
                recalc_hess += 10
            elif (orca5_obj.job_type_objs["OPT"].values_diff.iloc[req_obj.geo_to_use - 1] < 0.05).all():
                recalc_hess += 20
            elif (orca5_obj.job_type_objs["OPT"].values_diff.iloc[req_obj.geo_to_use - 1] < 0.01).all():
                recalc_hess += 100

            input_path = orca5_obj.root_name + ".inp"
            full_input_path = path.join(orca5_obj.root_path, input_path)
            return full_input_path, {"recalc_hess": recalc_hess, "xyz_path": xyz_name}

