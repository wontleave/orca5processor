import glob
import subprocess
import time
from os import path, remove, chdir
from pathlib import Path


class Step:
    def __init__(self, rms_gradient, max_gradient, rms_step, max_step, red_int_coords):
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


