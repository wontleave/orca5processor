import re
from rmsd import reorder_inertia_hungarian, rmsd
from os import path
from typing import Dict, List
import numpy as np
import pandas as pd
import time
from ase import Atoms
from ase.io import read
from pathlib import Path
from geo_optimization import OptBond, OptAngle, OptTorsion, Step

# Explanation on the use of case:
# self.keywords key: lowercase. self.jobtype key: UPPERCASE


class Orca5:
    """
    Read a folder that contains data generated from Orca 5.0 and create the relevant ASE objects
    Supported extensions for ORCA5 output are out, wlm01, coronab, lic01 and hpc-mn1
    Molecules are created as ASE Atoms object
    !!!Maybe compatible with ORCA 4, but use at your own risk
    """

    def __init__(self, orca_out_file, grad_cut_off,
                 property_path=None,
                 engrad_path=None,
                 output_path=None,
                 safety_check=False,
                 allow_incomplete=False,
                 get_opt_steps=False):

        # Get the root name of the orca_out_file
        self.root_name = None
        self.root_path = path.dirname(orca_out_file)
        self.property_path = property_path  # Extension is .txt. NOT USED FOR NOW 20210930
        self.engrad_path = engrad_path  # Extension is .engrad
        self.output_path = orca_out_file  # Extension can be .out, .coronab, .wml01, .lic001
        self.job_types = []  # If job_types is None after processing the given folder
        self.level_of_theory = None  # e.g CPCM(solvent)/B97-3c, CPCM(solvent)/wB97X-V/def2-TZVPP//PBEh-3c
        self.labelled_data: Dict[str, str] = {}  # Used to create a pandas for writing to excel
        self.warnings = []  # large gradient, incomplete job
        # Each job type will have its own object
        # SP: Scf
        # ENGRAD: Scf
        # OPT: Scf + Geom + Freq(optional, depends on whether there is a frequency calc after a successful opt)
        # OPTTS: Scf + Geom + Freq(optional, depends on whether there is a frequency calc after a successful opt)
        # Freq and PrintThermo: Scf + Freq
        # IRC: Scf + IRC
        self.job_type_objs = {}
        self.method: Method
        self.method = None  # All job type must have a Method object
        self.basis: Basis
        self.basis = None  # All job type must have a Basis object
        self.geo_from_output: Atoms
        self.geo_from_output = None  # All properties are derived from this structure.
        self.geo_from_xyz: Atoms
        self.geo_from_xyz = None  # The actual xyzfile used for the Orca 5 calculation

        self.freqs = {}  # A dict of Freq objects with temperature as the key(s)

        # Variables to check for certain condition
        self.completed_job = False  # A valid output file needs to have ORCA TERMINATED NORMALLY

        # Only property file -> single point, spectroscopic properties? Elec energy here doesn't include SRM
        with open(orca_out_file, "r") as f:
            lines = f.readlines()

        # Check if the ORCA 5 output is valid. If not we will not proceed further unless allow_incomplete is true
        if "ORCA TERMINATED NORMALLY" in lines[-2] or allow_incomplete:
            self.completed_job = True

        if self.completed_job:
            self.input_section = {"start": 0, "end": 0}
            self.find_input_sections(lines)

            # keywords used in the Orca 5 calculation are stored in <<self.keywords>>
            # self.coord_spec is a dict that contains
            # - "coord_type": xyz, xyzfile - "coord_path" - "charge" - "multiplicity
            # self.input_name is obtained from the input section "NAME = .... "
            self.keywords, self.coord_spec, self.input_name = \
                get_orca5_keywords(lines[self.input_section["start"]: self.input_section["end"]])

            self.root_name, _ = path.splitext(self.input_name)  # Join with .engrad to get the engrad if needed

            self.determine_jobtype()
            # TODO Only optTS is allowed for get_opt_steps = True
            # if get_opt_steps and "TS" not in self.job_types:
            #     print("WARNING: Only optTS is allowed for get_opt_steps = True ... setting get_opt_steps to False! ")
            #     get_opt_steps = False
            self.parse_orca5_output(lines[self.input_section["end"]:], grad_cut_off, get_opt_steps)
            self.get_level_of_theory()

            if self.coord_spec["coord_type"] == "xyzfile":
                found_xyzfile = False
                temp_path = path.join(self.root_path, self.coord_spec["coord_path"])
                from_root = None
                if Path(temp_path).is_file() and "OPT" not in self.job_types:
                    found_xyzfile = True
                    self.geo_from_xyz = read(temp_path)
                else:
                    from_root = path.join(self.root_path, self.input_name)
                    from_root = Path(from_root)
                    from_root = from_root.with_suffix(".xyz")
                    if from_root.is_file():
                        self.geo_from_xyz = read(from_root.resolve())
                        found_xyzfile = True
                assert found_xyzfile, f"Sorry, we can't find {temp_path} or {from_root.resolve()}"

            # compare the geometry from output and from xyzfile if necessary
            if self.geo_from_xyz is not None and \
                    not self.method.is_qmmm and \
                    not self.method.is_print_thermo and\
                    not allow_incomplete:
                # TODO check the QM atoms with the corresponding atom in the xyzfile
                rmsd_ = calc_rmsd_ase_atoms_non_pbs(self.geo_from_xyz, self.geo_from_output)
                if rmsd_ > 1e-5:
                    self.warnings.append(f"Difference between geometry from output and xyzfile. RMSD = {rmsd_}")
        else:
            self.warnings.append("INCOMPLETE ORCA5 job detect!")

    def parse_orca5_output(self, lines_, grad_cut_off, get_opt_steps):
        """
        job_types will determine what information to extract
        SP: only final single point energy
        :param lines_:
        :param grad_cut_off:
        :param get_opt_steps: if this is true, then each individual steps of opt will be read
        :return:
        """

        for job_type in self.job_types:
            if job_type == "SP":
                self.job_type_objs["SP"].get_final_sp_energy(lines_)
            elif job_type == "OPT":
                self.job_type_objs["OPT"].get_keywords(self.keywords["geom"].keywords)
                if get_opt_steps:
                    self.job_type_objs["OPT"].get_opt_steps(lines_)
                else:
                    self.job_type_objs["OPT"].get_opt_geo(lines_)
                # TODO read from xyz as an option?
            elif job_type in ("FREQ", "ANFREQ", "NUMFREQ"):
                if self.method.is_print_thermo:
                    self.job_type_objs["FREQ"].get_thermochemistry(lines_, print_thermo=True)
                else:
                    self.job_type_objs["FREQ"].get_thermochemistry(lines_)
                    self.freqs[self.job_type_objs["FREQ"].temp] = self.job_type_objs["FREQ"]

                if "OPT" not in self.job_types and "TS" not in self.job_types:

                    # Read the engrad to determine if we have a minimum.
                    # 2 possibilities: to obtain the engrad. From the xyz filename or from the freq job filename.
                    temp_path = path.join(self.root_path, self.coord_spec["coord_path"])
                    engrad_from_coord = Path(temp_path)
                    engrad_from_coord = engrad_from_coord.with_suffix(".engrad")
                    if engrad_from_coord.is_file():
                        self.job_type_objs["SP"].read_gradient(engrad_from_coord.resolve())
                        diff_wrt_ref = grad_cut_off - self.job_type_objs["SP"].gradients
                        if np.any(diff_wrt_ref < 0.0):
                            max_grad = np.max(self.job_type_objs["SP"].gradients)
                            self.warnings.append(f"Max gradient is above {grad_cut_off}: {max_grad}")
                        else:
                            self.job_type_objs["FREQ"].negligible_gradient = True
                    else:
                        temp_path = path.join(self.root_path, self.input_name)
                        engrad_from_inp = Path(temp_path)
                        engrad_from_inp = engrad_from_inp.with_suffix(".engrad")
                        if engrad_from_inp.is_file():
                            self.job_type_objs["SP"].read_gradient(engrad_from_inp.resolve())
                            diff_wrt_ref = grad_cut_off - self.job_type_objs["SP"].gradients
                            if np.any(diff_wrt_ref < 0.0):
                                max_grad = np.max(self.job_type_objs["SP"].gradients)
                                self.warnings.append(f"Max gradient is above {grad_cut_off}: {max_grad}")
                            else:
                                self.job_type_objs["FREQ"].negligible_gradient = True
                        else:
                            self.warnings.append("No engrad can be found for a frequency calculation")

            elif job_type == "CPCM":
                self.job_type_objs["CPCM"].get_cpcm_details(lines_)
            # TODO read QMMM details
            # elif job_type == "QMMM":
            #     print()

        if "OPT" not in self.job_types:
            if "FREQ" in self.job_types:
                self.geo_from_output = self.job_type_objs["FREQ"].geo_from_output
                self.job_type_objs["SP"].geo_from_output = self.job_type_objs["FREQ"].geo_from_output
            elif "SP" in self.job_types:
                self.geo_from_output = self.job_type_objs["SP"].geo_from_output
        else:
            self.geo_from_output = self.job_type_objs["OPT"].geo_from_output
            self.job_type_objs["SP"].geo_from_output = self.job_type_objs["OPT"].geo_from_output

    def find_input_sections(self, lines_):
        """

        :param lines_:
        :type lines_: [str]
        :return:
        :rtype:
        """
        separator_count = 0
        for idx, line_ in enumerate(lines_):
            if "INPUT FILE" in line_:
                self.input_section["start"] = idx + 2
            elif "****END OF INPUT****" in line_:
                self.input_section["end"] = idx

    def determine_jobtype(self):
        """
        Go through the simple keywords to determine the jobtype.
        Fill in parameters of the detailed keywords from the simple keyword
        :return:
        :rtype:
        """
        qmmm_lvl_of_theory = None

        if "OPT" in self.keywords["simple"]:
            self.job_types.append("OPT")
        elif "OPTTS" in self.keywords["simple"]:
            self.job_types.append("OPT")
            self.job_types.append("TS")

        if "FREQ" in self.keywords["simple"] or "PRINTTHERMOCHEM" in self.keywords["simple"]:
            self.job_types.append("FREQ")
        elif "ANFREQ" in self.keywords["simple"]:
            self.job_types.append("ANFREQ")
        elif "NUMFREQ" in self.keywords["simple"]:
            self.job_types.append("NUMFREQ")

        if "IRC" in self.keywords["simple"]:
            self.job_types.append("IRC")
        elif "ENGRAD" in self.keywords["simple"]:
            self.job_types.append("ENGRAD")

        if "QM/XTB" in self.keywords["simple"] or "QM/QM2" in self.keywords["simple"]:
            self.job_types.append("QMMM")
            qmmm_lvl_of_theory = "QM/XTB"

        # CPCM section
        solvent = None
        req_kw = None
        sanity_check = 0
        for kw in self.keywords["simple"]:
            req_kw = re.search("CPCM", kw)
            if req_kw is not None:
                self.job_types.append("CPCM")
                solvent = req_kw.string.split("(")[-1].strip(")")
                sanity_check += 1

            req_kw = re.search("ALPB", kw)
            if req_kw is not None:
                self.job_types.append("ALPB")
                solvent = req_kw.string.split("(")[-1].strip(")")
                sanity_check += 1

        assert sanity_check <= 1, f"You have {sanity_check} CPCM simple keyword in the ORCA5 input: "

        # This happens when there is no !CPCM(solvent) but there is %CPCM .... end
        if "cpcm" in self.keywords and "CPCM" not in self.job_types:
            self.job_types.append("CPCM")

        # Check if it is a single point
        if len(self.job_types) == 0:
            is_single_point = False
            for item in dft_simple_keywords:
                if item.upper() in self.keywords["simple"]:
                    is_single_point = True
                    break
            if not is_single_point:
                for item in general_keywords:
                    if item.upper() in self.keywords["simple"]:
                        is_single_point = True
                        break
                if "method" in self.keywords:
                    # Maybe a custom single point
                    if self.keywords["method"].keywords["functional"] is not None or \
                            self.keywords["method"].keywords["exchange"] is not None or \
                            self.keywords["method"].keywords["correlation"] is not None:
                        is_single_point = True

            if not is_single_point:
                self.job_types = None
            else:
                self.job_types.append("SP")

        # Create or copy the required job_type objs
        if self.job_types is not None:

            # All job type will contain a method, basis and Scf object
            if "method" not in self.keywords.keys():
                self.keywords["method"] = Method()

            self.method = self.keywords["method"]

            if "basis" not in self.keywords.keys():
                self.basis = Basis()

            self.basis.process_simple_keywords(self.keywords["simple"])

            if "SP" not in self.job_types:
                self.job_types.append("SP")
            if "scf" not in self.keywords.keys():
                self.job_type_objs["SP"] = Scf()
            else:
                self.job_type_objs["SP"] = self.keywords["scf"]

            for kw in scf_conv_simple_keywords:
                if kw.upper() in self.keywords["simple"]:
                    self.job_type_objs["SP"].keywords["convergence"] = kw

            for kw in dft_simple_keywords:
                if kw.upper() in self.keywords["simple"]:
                    self.method.keywords["functional"] = kw
                    self.method.keywords["method"] = "dft"

            # Other job types are specified here
            for item in self.job_types:
                # Detailed keywords used in ORCA 5 % ... end are in lowercase! e.g. self.method.keywords["runtyp"]
                # each key in self.keywords is in uppercase. Each of them correspond to an object in self.job_type_objs
                if item == "SP":
                    self.method = self.keywords["method"]
                    self.method.keywords["runtyp"] = "SP"

                elif item == "OPT":
                    self.method = self.keywords["method"]
                    self.method.keywords["runtyp"] = "OPT"

                    if "GEOM" not in self.keywords:
                        self.job_type_objs["OPT"] = Geom()
                    else:
                        self.job_type_objs["OPT"] = self.keywords["geom"]

                elif item in ('FREQ', "ANFREQ", "NUMFREQ"):
                    if "OPT" not in self.job_types:
                        self.method = self.keywords["method"]
                        self.method.keywords["runtyp"] = "FREQ"

                    if "freq" not in self.keywords:
                        self.keywords["freq"] = Freq()
                        if item == "ANFREQ":
                            self.keywords["freq"].keywords["anfreq"] = True
                        elif item == "NUMFREQ":
                            self.keywords["freq"].keywords["numfreq"] = True
                        self.job_type_objs["FREQ"] = self.keywords["freq"]
                    else:
                        self.job_type_objs["FREQ"] = self.keywords["freq"]

                    if "PRINTTHERMOCHEM" in self.keywords["simple"]:
                        self.method.is_print_thermo = True

                elif item == "CPCM":
                    if "cpcm" not in self.keywords:
                        self.keywords["cpcm"] = Cpcm(solvent)

                    self.job_type_objs["CPCM"] = self.keywords["cpcm"]
                    self.method.cpcm = self.keywords["cpcm"]
                    if self.job_type_objs["CPCM"].solvent == "":  # If %cpcm is detected solvent will be set to ""
                        self.job_type_objs["CPCM"].solvent = solvent

                elif item == "ALPB":
                    if "cpcm" not in self.keywords:
                        self.keywords["cpcm"] = Cpcm(solvent, name="ALPB")
                    self.job_type_objs["CPCM"] = self.keywords["cpcm"]
                    self.job_types.append("CPCM")
                    self.method.cpcm = self.keywords["cpcm"]
                    self.method.solvation = "ALPB"

                elif item == "QMMM":
                    if "qmmm" not in self.keywords:
                        self.keywords["qmmm"] = Qmmm()
                    self.job_type_objs["QMMM"] = self.keywords["qmmm"]
                    self.method.is_qmmm = True
                    if qmmm_lvl_of_theory is not None:
                        self.job_type_objs["QMMM"].lvl_of_theory = qmmm_lvl_of_theory

    def get_level_of_theory(self):
        """
        Only support some DFT functional now
        :return:
        :rtype:
        """
        level_of_theory = ""
        has_dft_exchange = False
        has_dft_correlation = False

        if self.method.is_print_thermo:
            print_thermo_spec_path = path.join(self.root_path, "PRINT_THERMO_SUPP.txt")
            print_thermo_spec = Path(print_thermo_spec_path)
            assert print_thermo_spec.is_file(), f"Cannot find {print_thermo_spec_path}. A PRINTTHERMO Job needs this!"
            lines = print_thermo_spec.read_text().split("\n")
            for line in lines:
                if "level_of_theory" in line:
                    level_of_theory = line.split()[-1]

        else:
            if self.method.keywords["functional"] is not None:
                level_of_theory += self.method.keywords["functional"]
            if self.method.keywords["exchange"] is not None:
                level_of_theory += self.method.keywords["exchange"].split("_")[-1]
                has_dft_exchange = True
            if self.method.keywords["correlation"] is not None:
                level_of_theory += self.method.keywords["correlation"].split("_")[-1]
                has_dft_correlation = True

            if has_dft_exchange and has_dft_correlation and "mp2" in self.keywords:
                # Some custom double-hybrid functional
                if self.keywords["mp2"].keywords["doscs"].lower() == "true":
                    level_of_theory = "DSD-" + level_of_theory
                if self.keywords["mp2"].keywords["dlpno"].lower() == "true":
                    level_of_theory = "DLPNO-" + level_of_theory

            if "3c" not in level_of_theory and self.basis.keywords["basis"] is not None:
                basis = self.basis.keywords["basis"]
                level_of_theory += f"/{basis}"

            if "QMMM" in self.job_types:
                if self.keywords["qmmm"].lvl_of_theory != "":
                    high_level, low_level = self.keywords["qmmm"].lvl_of_theory.split("/")
                    level_of_theory += f":{low_level.upper()}"

            if "CPCM" in self.job_types:
                if self.keywords["cpcm"].keywords["smd"].lower() == "true":
                    solvent = self.keywords["cpcm"].keywords["smdsolvent"]
                    level_of_theory = f"SMD({(solvent.upper())})/" + level_of_theory
                else:
                    solvent = self.keywords["cpcm"].solvent
                    if self.keywords["cpcm"].name == "ALPB":
                        fepstype = "ALPB"
                    elif self.keywords["cpcm"].keywords["cds_cpcm"] == 2:
                        fepstype = "CPCM2"
                    else:
                        fepstype = self.keywords["cpcm"].keywords["fepstype"]
                    level_of_theory = f"{fepstype}({(solvent)})/" + level_of_theory

        self.level_of_theory = level_of_theory

    def create_labelled_data(self, temperature=298.15):
        """

        :return:
        :rtype:
        """
        # Create the thermochemistry correction from an optimized geometry first
        thermochemistry = {}
        single_point = {}

        if temperature in self.freqs.keys():
            req_freq = self.freqs[temperature]
            if "OPT" in self.job_type_objs.keys() or req_freq.negligible_gradient:
                if temperature in req_freq.thermo_data:
                    temp_ = req_freq.thermo_data[temperature]
                    thermochemistry[self.level_of_theory + "--ZPE"] = temp_["zero point energy"]
                    thermochemistry[self.level_of_theory + "--thermal"] = temp_["total thermal correction"]
                    thermochemistry[self.level_of_theory + "--thermal_enthalpy_corr"] = \
                        temp_["thermal Enthalpy correction"]
                    thermochemistry[self.level_of_theory + "--final_entropy_term"] = temp_["total entropy correction"]
                    thermochemistry[self.level_of_theory + "--elec_energy"] = \
                        self.job_type_objs["FREQ"].elec_energy
                    thermochemistry[self.level_of_theory + "--total_thermal_energy"] = temp_["total thermal energy"]
                    thermochemistry[self.level_of_theory + "--total_enthalpy"] = temp_["total enthalpy"]
                    thermochemistry[self.level_of_theory + "--final_gibbs_free_energy"] = \
                        temp_["final gibbs free energy"]

                    freqs_ = np.array(self.job_type_objs["FREQ"].frequencies)
                    negative_freqs = freqs_[freqs_ < 0.0]
                    thermochemistry["N Img Freq"] = np.count_nonzero(freqs_ < 0.0)
                    thermochemistry["Negative Freqs"] = str(negative_freqs.tolist())
                    self.labelled_data = {**self.labelled_data, **thermochemistry}
                else:
                    temp_ = self.job_type_objs["FREQ"].thermo_data
                    raise ValueError(f"We cannot find {temperature} in {temp_}")
        else:
            self.warnings.append(f"THERMOCHEMISTRY: {temperature} cannot be found!")

        if "SP" in self.job_type_objs.keys():
            if "OPT" not in self.job_type_objs.keys() and "FREQ" not in self.job_type_objs.keys():
                single_point[self.level_of_theory] = self.job_type_objs["SP"].final_sp_energy
                self.labelled_data = {**self.labelled_data, **single_point}


# ORCA5 simple keywords
scf_conv_simple_keywords = ("NORMALSCF", "LOOSESCF", "SLOPPYSCF", "STRONGSCF", "TIGHTSCF", "VERYTIGHTSCF",
                            "EXTREMESCF", "SCFCONV")

geom_conv_simple_keywords = ("VERYTIGHTOPT", "TIGHTOPT", "NORMALOPT", "LOOSEOPT")

grid_simple_keywords = ("DEFGRID", "NOFINALGRIDX")

runtypes_simple_keywords = ("ENERGY", "SP", "OPT", "ZOPT", "COPT", "GDIIS-COPT", "GDIIS-ZOPT", "ENGRAD", "NUMGRAD",
                            "NUMFREQ", "NUMNACME", "MD", "CIM")

general_keywords = ("HF", "DFT", "FOD")

# ORCA5 DFT functional
dft_simple_keywords = ("PBEh-3c", "r2scan-3c", "B97-3c",
                       "B3LYP", "M06-2X", "wB97X-V", "wB97X-D4", "wB97M-V", "wB97M-D4")

qmmm_simple_keywords = ("QM/XTB")

# ORCA5 basis sets keywords
basis_set_keywords = {"basis_set_keywords": ("def2-SVP", "def2-SV(P)", "def2-TZVP", "def2-TZVP(-f)",
                                             "def2-TZVPP", "def2-QZVP", "def2-QZVPP"),
                      "aux_basis_coulomb_keywords": ("def2/J", "SARC/J", "x2c/J"),
                      "aux_basis_coulomb_ex_keywords": ("def2/JK", "def2/JKsmall"),
                      "aux_basis_correlation_keywords": ("def2-SVP/C", "def2-TZVP/C", "def2-TZVPP/C", "def2-QZVPP/C",
                                                         "def2-SVPD/C", "def2-TZVPD/C", "def2-TZVPPD/C",
                                                         "def2-QZVPPD/C"),
                      "aux_general_keywords": "autoaux"
                      }

# ORCA5 input block
pal_block = {}

# ORCA4 simple keywords
grid_simple_keywords = ("GRID", "GRIDX", "NOFINALGRIDX", "NOFINALGRID")

# Classes related to Detailed Keywords---------------------------------------------------------------------------------


class Method:
    def __init__(self, run_type=None, cpcm=None):
        """
        Store information pertinent to the level of theory
        All job type must have a method object
        :param run_type: The type of Orca 5 jobs: SP, ENGRAD, OPT
        :type run_type: str
        :param cpcm: a Cpcm object
        :type cpcm: Cpcm
        """
        self.name = "method"
        self.keywords = {"runtyp": run_type,
                         "functional": None, "exchange": None, "correlation": None,
                         "scalmp2c": None, "scalldac": None, "scalhfx": None,
                         "scalggac": None, "scaldfx": None,
                         "method": None,
                         "mayer_bondorderthres": None,
                         "ri": None,
                         "d3s6": None, "d3a2": None, "d3s8": None, "d3a1": None
                         }

        self.single_value_str = ("exchange", "correlation", "runtyp", "functional", "method", "ri")
        self.single_value_float = ("mayer_bondorderthres", "scalmp2c", "scalldac", "scalhfx", "scalggac", "scaldfx",
                                   "d3s6", "d3a2", "d3s8", "d3a1")
        self.single_value_int = ()
        self.multi_values = ()  # Multi_values keywords required an "end" to terminate
        self.is_qmmm = False
        self.is_print_thermo = False
        self.solvation: str = ""


class Basis:
    def __init__(self, basis=None, auxj=None, auxjk=None, auxc=None, cabs=None):
        """
        Store information on the basis set used in a calculation.
        It is essential.
        :param basis:
        :type basis:
        :param auxJ:
        :type auxJ:
        :param auxJK:
        :type auxJK:
        :param auxC:
        :type auxC:
        :param cabs:
        :type cabs:
        """
        self.keywords = {"basis": basis,
                         "auxj": auxj, "auxjk": auxjk, "auxc": auxc, "cabs": cabs,
                         "decontractbas": False, "decontractauxj": False, "decontractauxjk": False,
                         "decontractauxc": False, "decontractcabs": False, "decontract": False
                         }

    def process_simple_keywords(self, simple_kw):
        for kw in basis_set_keywords["basis_set_keywords"]:
            if kw.upper() in simple_kw:
                self.keywords["basis"] = kw

        # Auxillary basis
        is_autoaux = False
        for kw in basis_set_keywords["aux_general_keywords"]:
            if kw.upper() in simple_kw:
                self.keywords["auxj"] = kw
                self.keywords["auxjk"] = kw
                self.keywords["auxc"] = kw
                is_autoaux = True

        if not is_autoaux:
            for kw in basis_set_keywords["aux_basis_coulomb_keywords"]:
                if kw.upper() in simple_kw:
                    self.keywords["auxj"] = kw

            for kw in basis_set_keywords["aux_basis_coulomb_ex_keywords"]:
                if kw.upper() in simple_kw:
                    self.keywords["auxjk"] = kw

            for kw in basis_set_keywords["aux_basis_correlation_keywords"]:
                if kw.upper() in simple_kw:
                    self.keywords["auxc"] = kw


class Cpcm:
    def __init__(self, solvent, name="cpcm"):
        """
        Solvation will be covered under Method
        ALPB is currently under Cpcm
        """
        self.name = name
        self.detail_read = False  # CPCM details appear at every SCF cycle. This flag causes it to be read only once
        self.solvent = solvent  # The CPCM solvent
        self.keywords = {"epsilon": None, "refrac": None, "rsolv": None, "rmin": None, "pmin": None,
                         "surfacetype": None, "fepstype": None, "xfeps": None,
                         "ndiv": None, "num_leb": None,
                         "smd": "False", "smdsolvent": None,
                         "cds_cpcm": None
                         }

        self.single_value_str = ("surfacetype", "fepstype", "smd", "smdsolvent")
        self.single_value_float = ("epsilon", "refrac", "rmin", "pmin", "rsolv", "xfeps")
        self.single_value_int = ("ndiv", "num_leb", "cds_cpcm")
        self.multi_values = ()  # Multi_values keywords required an "end" to terminate

    def get_cpcm_details(self, lines_):
        """

        :param lines_:
        :type lines_:
        :return:
        :rtype:
        """

        if not self.detail_read:
            read_cpcm = False
            for line in lines_:
                if "CPCM SOLVATION MODEL" in line:
                    read_cpcm = True
                elif read_cpcm:
                    if "Epsilon                                         ..." in line:
                        self.keywords["epsilon"] = float(line.split()[-1])
                    elif "Refrac                                          ..." in line:
                        self.keywords["refrac"] = float(line.split()[-1])
                    elif "Rsolv                                           ..." in line:
                        self.keywords["rsolv"] = float(line.split()[-1])
                    elif "Surface type                                    ..." in line:
                        self.keywords["surfacetype"] = line.split("... ")[-1].strip()
                    elif "Epsilon function type                           ..." in line:
                        self.keywords["fepstype"] = line.split()[-1].strip()


class Geom:
    def __init__(self):
        """
        TODO indicate if the geometry optimization is a TS search or not
        """
        self.name = "geom"
        self.keywords = {"maxiter": None,
                         "gdiismaxeq": None,
                         "numhess": None,
                         "calc_hess": None,
                         "recalc_hess": None,
                         "trust": None,
                         "numhess_centraldiff": None,
                         "scan": [],
                         "modify_internal": [],
                         "constraints": []}

        self.single_value_str = ("calc_hess, numhess", "correlation", "numhess_centraldiff")
        self.single_value_float = ("trust")
        self.single_value_int = ("maxiter", "gdiismaxeq", "recalc_hess")
        # Multi_values keywords required an "end" to terminate
        self.multi_values = ("scan", "modify_internal", "constraints")

        # ASE Atoms object of the optimized geometry. Both has to be equal
        self.geo_from_output = None
        self.geo_from_xyz = None
        self.int_coords_at_steps = None
        self.is_optts = False

        # Geometry optmization analysis variables
        self.converged = False
        self.opt_steps = None
        self.req_int_coords_idx = []
        self.req_data_pd = None  # Multi-index dataframe row-step index columns:
        # 0 - required_index, level 1 - value, TS mode
        self.values_diff = None  # The difference between the i and i+1 step in the optimization divide by the distance
        self.action = None  # An Action object to indicate what to do with this job
        self.opt_trj: List[Atoms] = None

    def get_keywords(self, keywords):
        """
        Sort the keywords from the output file and assign them to the appropriate keywords in GEOM

        :param keywords: a dictionary of ORCA5 keywords from the output file
        :type  keywords: dict[str, str]
        :return:
        """
        for key in keywords:
            if key in self.keywords:
                if key in self.single_value_int:
                    if keywords[key] is not None:
                        self.keywords[key] = int(keywords[key])
                else:
                    # TODO multi values keywords are not supported
                    self.keywords[key] = keywords[key]

    def get_opt_geo(self, lines_):
        """
        Read the xyz coordinates for the Orca 5 output text.
        TODO Read the convergence Criteria every step
        TODO Read the CPCM volume and surface area of the optimized structure
        :param lines_:
        :type lines_:
        :return:
        :rtype:
        """
        start_to_read_xyz = 0
        n_atoms = 0
        element = None
        x = None
        y = None
        z = None
        xyz = ""
        start_to_read_failed_attempt = 0
        elements_seq = ""
        cart_coords = []

        req_idx = -1

        # In case of a OPT+FREQ, the
        for i in range(len(lines_) - 1, 0, -1):
            if "THE OPTIMIZATION HAS CONVERGED" in lines_[i]:
                req_idx = i
                break

        assert req_idx > 0, f"Invalid Orca 5 FREQ output file detected! or incomplete geometry optimization!"

        for line in lines_[req_idx:]:
            # Read cartesian coordinates if start_to_read_xyz == 2
            if start_to_read_xyz == 2:
                try:
                    element, x, y, z = line.split()
                    elements_seq += element
                    x = float(x)
                    y = float(y)
                    z = float(z)
                    cart_coords.append((x, y, z))
                    n_atoms += 1
                except ValueError:
                    if start_to_read_failed_attempt == 2:
                        break
                    else:
                        start_to_read_failed_attempt += 1
                    continue
            if "THE OPTIMIZATION HAS CONVERGED" in line:
                start_to_read_xyz += 1
                continue
            if "CARTESIAN COORDINATES (ANGSTROEM)" in line and start_to_read_xyz:
                start_to_read_xyz += 1

        self.geo_from_output = Atoms(elements_seq, positions=cart_coords)

    def get_opt_steps(self, lines_):
        steps = []
        converged = False
        red_int_coords_at_steps = {}
        energy_change: float = None
        rms_gradient: float = None
        max_gradient: float = None
        rms_step: float = None
        max_step: float = None

        # Flags to control whether to start reading redundant internal coordinates
        in_geo_cycle = False
        in_red_coords_section = False
        read_red_coords = False

        for line in lines_:
            if "GEOMETRY OPTIMIZATION CYCLE" in line:
                current_step = int(line.split()[-2])
                in_geo_cycle = True
            elif "Energy change" in line and "..." not in line:
                energy_change = float(line.split()[2])
            elif "RMS gradient " in line and "..." not in line:
                rms_gradient = float(line.split()[2])
            elif "MAX gradient" in line and "..." not in line:
                max_gradient = float(line.split()[2])
            elif "RMS step" in line:
                rms_step = float(line.split()[2])
            elif "MAX step" in line:
                max_step = float(line.split()[2])
            elif "THE OPTIMIZATION HAS CONVERGED" in line:
                converged = True
                break
            elif "Redundant Internal Coordinates" in line and in_geo_cycle:
                in_red_coords_section = True
                continue

            if "----------------------------------------------------------------------------" in line \
                    and in_red_coords_section:
                if read_red_coords:
                    in_geo_cycle = False
                    read_red_coords = False
                    in_red_coords_section = False
                    steps.append(Step(energy_change, rms_gradient, max_gradient, rms_step, max_step,
                                      red_int_coords_at_steps))
                    red_int_coords_at_steps = {}
                    rms_gradient: float = None
                    max_gradient: float = None
                    rms_step: float = None
                    max_step: float = None
                    continue
                else:
                    read_red_coords = True
                    continue

            if in_red_coords_section:
                if read_red_coords:
                    # Two letters elements such as Br, Cl, Na, etc will affect the spiltting.
                    temp_ = line.split("(")
                    index = int(temp_[0].split(".")[0])
                    int_coords_type = temp_[0][-1]
                    temp_ = temp_[1].split(")")
                    if int_coords_type == "B":
                        left, right = temp_[0].split(",")
                        red_int_coords_at_steps[index] = OptBond(left[0:2], right[0:2],
                                                                 left[2:], right[2:])
                    elif int_coords_type == "A":
                        atom1, atom2, atom3 = temp_[0].split(",")
                        red_int_coords_at_steps[index] = OptAngle(atom1[0:2], atom2[0:2], atom3[0:2],
                                                                  atom1[2:], atom2[2:], atom3[2:])

                    elif int_coords_type == "D":
                        atom1, atom2, atom3, atom4 = temp_[0].split(",")
                        red_int_coords_at_steps[index] = OptTorsion(atom1[0:2], atom2[0:2], atom3[0:2], atom4[0:2],
                                                                    atom1[2:], atom2[2:], atom3[2:], atom4[2:])

                    elif int_coords_type == "L":
                        try:
                            atom1, atom2, atom3, atom4, part = temp_[0].split(",")
                            red_int_coords_at_steps[index] = OptTorsion(atom1[0:2], atom2[0:2], atom3[0:2], atom4[0:2],
                                                                        atom1[2:], atom2[2:], atom3[2:], atom4[2:],
                                                                        part=part)
                        except ValueError:
                            raise ValueError(f"{temp_[0]}")

                    temp_ = temp_[1].split()
                    # temp _ 0: Value
                    #        1: dE/q
                    #        2: Step
                    #        3: New-Value
                    #        4  omp.(TS mode)

                    red_int_coords_at_steps[index].distance_old = float(temp_[0])
                    red_int_coords_at_steps[index].gradient = float(temp_[1])
                    red_int_coords_at_steps[index].step = float(temp_[2])
                    red_int_coords_at_steps[index].distance_new = float(temp_[3])

                    try:
                        if temp_[4] == "C":
                            red_int_coords_at_steps[index].constrained = True
                        else:
                            red_int_coords_at_steps[index].ts_mode = float(temp_[4])
                    except IndexError:
                        continue

        self.opt_steps = steps
        self.converged = converged

    def get_req_int_coords_idx(self, int_coords_from_spec):
        for coords in int_coords_from_spec:
            num_dash = coords.count("-")
            if num_dash == 1:
                atom1, atom2 = coords.split("-")
                self.req_int_coords_idx.append((int(atom1), int(atom2)))
            elif num_dash == 2:
                atom1, atom2, atom3 = coords.split("-")
                self.req_int_coords_idx.append((int(atom1), int(atom2), int(atom3)))
            elif num_dash == 3:
                atom1, atom2, atom3, atom4 = coords.split("-")
                self.req_int_coords_idx.append((int(atom1), int(atom2), int(atom3), int(atom4)))
            else:
                raise ValueError(f"{num_dash} \"-\" was found. This is not a valid internal coordinate")

    def steps_to_pd(self, int_coords_from_spec):
        """
        Create a multi-index dataframe
        :param int_coords_from_spec: used to determine if an internal coordinate is to be included as given in ppinp.txt
        :type int_coords_from_spec: List[str]
        :return:
        """
        self.get_req_int_coords_idx(int_coords_from_spec)
        int_coords_detail = ["value", "TS mode"]
        row_idx = list(range(1, len(self.opt_steps)+1))
        index = pd.MultiIndex.from_product([self.req_int_coords_idx, int_coords_detail],
                                            names=["required int coords", "details"])

        ordered_data = []

        for step in self.opt_steps:
            data = []
            for req_idx in self.req_int_coords_idx:
                idx = step.int_coord_to_idx[req_idx]
                data.append(step.red_int_coords[idx].distance_old)
                data.append(step.red_int_coords[idx].ts_mode)
            ordered_data.append(data.copy())

        self.req_data_pd = pd.DataFrame(ordered_data, index=row_idx, columns=index)

    def get_opt_trj(self, trj_full_path, final_xyzfile_path):
        """

        :param trj_full_path:
        :param final_xyzfile_path:
        :return: list[Atoms]
        """
        initial_to_second_last = read(trj_full_path, index=':')
        last = read(final_xyzfile_path)
        # include the last geometry
        self.opt_trj = initial_to_second_last + [last]


class Mtr:
    def __init__(self):
        self.name = "mtr"
        self.keywords = {"hessname": None,
                         "modetype": None,
                         "mlist": [],
                         "rsteps": [],
                         "lsteps": [],
                         "enstep": []}

        self.single_value_str = ("hessname", "modetype")
        self.single_value_float = ()
        self.single_value_int = ()
        # Multi_values keywords required an "end" to terminate
        self.multi_values = ("mlist", "rsteps", "lsteps", "enstep")


class Freq:
    def __init__(self):
        self.name = "freq"
        self.keywords = {"temp": None,
                         "t": [],
                         "numfreq": None,
                         "anfreq": None,
                         "scalfreq": 1.0,
                         "centradiff": None,
                         "restart": None,
                         "dx": None,
                         "increment": None,
                         "hybrid_hess": [],
                         "quasirrho": None,
                         "cutoffreq": None}

        self.single_value_str = ("numfreq", "anfreq", "centradiff", "restart")
        self.single_value_float = ("temp", "scalfreq", "dx", "increment", "cutoffreq", "quasirrho")
        self.single_value_int = ()
        # Multi_values keywords required an "end" to terminate
        self.multi_values = ("hybrid_hess", "t")

        # Calculation's result
        self.thermo_data: Dict[float, Dict[str, float]] = {}
        self.temp = None
        self.pressure = None
        self.total_mass = None
        self.frequencies = []
        self.elec_energy = None
        self.total_thermal_correction = None
        self.zero_pt_energy = None
        self.total_thermal_energy = None  # electronic energy + total thermal correction + ZPE
        self.thermal_enthalpy_correction = None
        self.total_enthalpy = None
        self.final_entropy_term = None
        self.final_gibbs_free_energy = None
        # If an OPT object is not present but the gradient is less than 5e-6 we consider this frequency to be part of
        # a converged geometry optimization
        self.negligible_gradient = False
        self.geo_from_output = None

    def get_thermochemistry(self, lines_, with_opt=False, print_thermo=False):
        """
        TODO: Handle printthermo
        :param lines_:
        :type lines_:
        :param with_opt: indicate whether a Freq job is part of an opt or optTS job, If true, the Freq information will
        only be obtained after the geometry optimization has converged.
        :param print_thermo: if this is a printthermo job, there will be no information on geometry
        :type with_opt: bool
        :return:
        :rtype:
        """

        start_to_read_freq = False
        start_to_read_thermo = False
        freq_read = False
        thermo_check = 0  # Check the number of thermochemistry data read
        # 1: found the line "CARTESIAN COORDINATES (ANGSTROEM)"
        # 2: found the ilne : ------------------
        start_geometry_from_single_point = 0

        elements_seq = ""
        coordinates = []

        if with_opt:
            successful_opt = False
        else:
            successful_opt = True
            geometry_from_single_point = True

        req_idx = -1

        # In case of a OPT+FREQ, the
        for i in range(len(lines_) - 1, 0, -1):
            if print_thermo and "ORCA THERMOCHEMISTRY ANALYSIS" in lines_[i]:
                req_idx = i
                break
            elif "THE OPTIMIZATION HAS CONVERGED" in lines_[i]:
                req_idx = i
                break

        if req_idx < 0:
            for i in range(len(lines_) - 1, 0, -1):
                if "CARTESIAN COORDINATES (ANGSTROEM)" in lines_[i]:
                    # This is not a OPT+FREQ
                    req_idx = i
                    break

        assert req_idx > 0, f"Invalid Orca 5 FREQ output file detected!"

        for idx, line in enumerate(lines_[req_idx:]):

            if "THE OPTIMIZATION HAS CONVERGED" in line:
                successful_opt = True
                continue
            if "VIBRATIONAL FREQUENCIES" in line and successful_opt:
                start_to_read_freq = True
                continue
            if "THERMOCHEMISTRY AT" in line and successful_opt:
                start_to_read_thermo = True
                continue
            if "CARTESIAN COORDINATES (ANGSTROEM)" in line:
                start_geometry_from_single_point += 1
                continue
            if thermo_check == 11:
                start_to_read_thermo = False

            if start_to_read_freq and "cm**-1" in line:
                try:
                    temp = line.split()
                    if len(temp) == 3:
                        self.frequencies.append(float(temp[-2]))
                    elif len(temp) == 5:
                        self.frequencies.append(float(temp[1]))
                except ValueError:
                    start_to_read_freq = False
                    freq_read = True

            elif start_to_read_thermo:
                if "Temperature         ..." in line:
                    try:
                        self.temp = float(line.split()[2])
                        thermo_check += 1
                    except ValueError:
                        raise ValueError(f"{line} -- does not contain the temperature at the expected index")
                elif "Pressure            ..." in line:
                    try:
                        self.pressure = float(line.split()[2])
                        thermo_check += 1
                    except ValueError:
                        raise ValueError(f"{line} -- does not contain the pressure at the expected index")
                elif "Total Mass          ..." in line:
                    try:
                        self.total_mass = float(line.split()[-2])
                        thermo_check += 1
                    except ValueError:
                        raise ValueError(f"{line} -- does not contain the total mass at the expected index")
                elif "Electronic energy                ..." in line:
                    try:
                        self.elec_energy = float(line.split()[-2])
                        thermo_check += 1
                    except ValueError:
                        raise ValueError(f"{line} -- does not contain the electronic energy at the expected index")
                elif "Zero point energy                ..." in line:
                    try:
                        self.zero_pt_energy = float(line.split()[-4])
                        thermo_check += 1
                    except ValueError:
                        raise ValueError(f"{line} -- does not contain the zero point energy at the expected index")
                elif "Total thermal correction" in line:
                    try:
                        self.total_thermal_correction = float(line.split()[-4])
                        thermo_check += 1
                    except ValueError:
                        raise ValueError(f"{line} -- does not contain the total thermal correction"
                                         f" at the expected index")
                elif "Total thermal energy" in line:
                    try:
                        self.total_thermal_energy = float(line.split()[-2])
                        thermo_check += 1
                    except ValueError:
                        raise ValueError(f"{line} -- does not contain the total thermal energy"
                                         f" at the expected index")
                elif "Thermal Enthalpy correction       ..." in line:
                    try:
                        self.thermal_enthalpy_correction = float(line.split()[-4])
                        thermo_check += 1
                    except ValueError:
                        raise ValueError(f"{line} -- does not contain the thermal enthalpy correction"
                                         f" at the expected index")
                elif "Total Enthalpy                    ..." in line:
                    try:
                        self.total_enthalpy = float(line.split()[-2])
                        thermo_check += 1
                    except ValueError:
                        raise ValueError(f"{line} -- does not contain the total enthalpy"
                                         f" at the expected index")
                elif "Total entropy correction          ..." in line:
                    # The final entropy term in ORCA 5 needs to be multiply by -1
                    try:
                        self.final_entropy_term = float(line.split()[-4])
                        thermo_check += 1
                    except ValueError:
                        raise ValueError(f"{line} -- does not contain the final entropy term"
                                         f" at the expected index")
                elif "Final Gibbs free energy         ..." in line:
                    try:
                        self.final_gibbs_free_energy = float(line.split()[-2])
                        thermo_check += 1
                    except ValueError:
                        raise ValueError(f"{line} -- does not contain the final Gibbs free energy"
                                         f" at the expected index")

            elif start_geometry_from_single_point:
                if start_geometry_from_single_point == 1 and "---------------------------------" in line:
                    start_geometry_from_single_point += 1
                    continue
                elif start_geometry_from_single_point == 2:
                    try:
                        element, x, y, z = line.split()
                        elements_seq += element
                        coordinates.append([float(x), float(y), float(z)])
                    except ValueError:
                        start_geometry_from_single_point = 0
                        self.geo_from_output = Atoms(elements_seq, positions=np.array(coordinates))
                        continue

            elif thermo_check == 11:
                thermo_check += 1
                self.thermo_data[self.temp] = {"zero point energy": self.zero_pt_energy,
                                               "total thermal correction": self.total_thermal_correction,
                                               "thermal Enthalpy correction": self.thermal_enthalpy_correction,
                                               "total entropy correction": self.final_entropy_term,
                                               "zero point corrected energy": self.elec_energy + self.zero_pt_energy,
                                               "total thermal energy": self.total_thermal_energy,
                                               "total enthalpy": self.total_enthalpy,
                                               "final gibbs free energy": self.final_gibbs_free_energy}


class Scf:
    def __init__(self):
        """
        This class will contain the keywords that specify a SCF calculation.
        It will contain the results of a SCF calculation:
        Final single point energy in a SP. Final single point of the optimized structure in OPT and OPTTS.

        """
        self.name = "scf"
        self.keywords = {"convergence": None,
                         "autotrah": None,
                         "stabperform": None,
                         "tole": None,
                         "tolg": None,
                         "sthresh": 1E-8}

        self.single_value_str = ("convergence", "autotrah", "stabperform")
        self.single_value_float = ("tole", "tolg", "sthresh")
        self.single_value_int = ()
        # Multi_values keywords required an "end" to terminate
        self.multi_values = ()

        self.gradients_cut_off = 1e-5
        self.gradients = None
        self.final_sp_energy = None
        self.geo_from_output = None

    def get_final_sp_energy(self, lines):
        """

        :return:
        :rtype:
        """
        start_geometry_from_single_point = 0
        elements_seq = ""
        coordinates = []

        for line in lines:
            if "CARTESIAN COORDINATES (ANGSTROEM)" in line:
                start_geometry_from_single_point += 1

            if "FINAL SINGLE POINT ENERGY" in line:
                try:
                    self.final_sp_energy = float(line.split()[-1])
                except ValueError:
                    raise ValueError(f"Cannot cast {line.split()[-1]} as a float ! check your output!")
            elif start_geometry_from_single_point:
                if start_geometry_from_single_point == 1 and "---------------------------------" in line:
                    start_geometry_from_single_point += 1
                    continue
                elif start_geometry_from_single_point == 2:
                    try:
                        element, x, y, z = line.split()
                        elements_seq += element
                        coordinates.append([float(x), float(y), float(z)])
                    except ValueError:
                        start_geometry_from_single_point = 0
                        self.geo_from_output = Atoms(elements_seq, positions=np.array(coordinates))
                        continue

    def read_gradient(self, path_to_engrad):
        """
        Start to read based on the number of #: we expect 8

        :param path_to_engrad: the full path to the engrad file
        :type path_to_engrad: str
        :return: None
        :rtype: None
        """
        with open(path_to_engrad, "r") as f:
            lines_ = f.readlines()

        start_to_read = 8
        coords_count = 0
        n_atoms = 0
        n_cart_forces = 0
        gradients = None
        for line in lines_:
            if start_to_read == -2 and coords_count < n_cart_forces:
                gradients[coords_count] = float(line)
                coords_count += 1
            elif start_to_read == 5:
                n_atoms = int(line)
                n_cart_forces = 3 * n_atoms
                assert gradients is None, "There is something wrong with your engrad file!!! Gradient is " \
                                          "initialized more than once"
                gradients = np.zeros(n_cart_forces)
                start_to_read -= 1
            elif start_to_read == -3:
                self.gradients = gradients.reshape(-1, 3)
                break

            if "#" in line:
                start_to_read -= 1


class Qmmm:
    def __init__(self, qm_spec=None, qm_charge=0, qm_multiplicity=1, total_charge=0, total_multiplicity=1):
        self.name = "qmmm"
        self.lvl_of_theory: str = ""
        self.keywords = {"qmatoms:": None, "charge_total": total_charge, "multi_total": total_multiplicity,
                         "qm2custommethod": None, "qm2custombasis": None, "qm2customfile": None}

        self.single_value_str = ("qm2custommethod", "qm2custombasis", "qm2customfile")
        self.single_value_float = ()
        self.single_value_int = ("charge_total", "multi_total")
        # Multi_values keywords required an "end" to terminate
        self.multi_values = ("qmatoms")

        if qm_spec is not None:
            self.keywords["qmatoms"] = qm_spec
            self.keywords["charge_total"] = total_charge
            self.keywords["multi_total"] = total_multiplicity

    def write_qmmm_inp(self, inp_path):
        """

        :return:
        :rtype:
        """
        qm_atoms = self.keywords["qmatoms"]
        total_charge = self.keywords["charge_total"]
        mult_total = self.keywords["multi_total"]
        to_be_written = f"%qmmm\n    QMAtoms {qm_atoms} end\n    " \
                        f"Charge_Total {total_charge}\n    " \
                        f"mult_total {mult_total}\n" \
                        f"end"

        with open(inp_path, "w") as f:
            f.writelines(to_be_written)


class Mp2:
    def __init__(self):
        self.name = "mp2"
        self.keywords = {"doscs": None, "ps": None, "pt": None, "dlpno": None, "ri": None}

        self.single_value_str = ("doscs", "dlpno", "qm2customfile", "ri")
        self.single_value_float = ("ps", "pt")
        self.single_value_int = ()
        # Multi_values keywords required an "end" to terminate
        self.multi_values = ()

# END of classes related to detailed keyword---------------------------------------------------------------------------


# Independent function collections
def reorder_atoms(atoms_p, atoms_q):
    """

    :param atoms_p:
    :type atoms_p: Atoms
    :param atoms_q:
    :type atoms_q: Atoms
    :return:
    :rtype:
    """
    p_elements = np.array(atoms_p.get_chemical_symbols())
    q_elements = np.array(atoms_q.get_chemical_symbols())
    p_coords = np.array(atoms_p.get_positions())
    q_coords = np.array(atoms_q.get_positions())
    assert p_coords.shape == q_coords.shape, f"Sorry, no point comparing two molecules of different number of atoms. " \
                                             f"atoms_p:{p_coords.shape} and atoms_q:{q_coords.shape}"

    q_review = reorder_inertia_hungarian(p_elements, q_elements, p_coords, q_coords)
    return q_review


def calc_rmsd_ase_atoms_non_pbs(atoms_p, atoms_q):
    """
    Compare two ASE Atoms object and calculate its rmsd
    :param atoms_p:
    :type atoms_p: Atoms
    :param atoms_q:
    :type atoms_q: Atoms
    :return:
    :rtype:
    """
    p_elements = np.array(atoms_p.get_chemical_symbols())
    q_elements = np.array(atoms_q.get_chemical_symbols())
    p_coords = np.array(atoms_p.get_positions())
    q_coords = np.array(atoms_q.get_positions())
    assert p_coords.shape == q_coords.shape, f"Sorry, no point comparing two molecules of different number of atoms. " \
                                             f"atoms_p:{p_coords.shape} and atoms_q:{q_coords.shape}"

    q_review = reorder_inertia_hungarian(p_elements, q_elements, p_coords, q_coords)
    q_coords = q_coords[q_review]
    return rmsd(p_coords, q_coords)


def calc_rmsd_ase_multi_atoms(some_atoms_objs: [List]):
    """

    :param some_atoms_objs:
    :type some_atoms_objs:
    :return: a matrix of RMSD
    :rtype: np.array
    """
    n_objs = len(some_atoms_objs)
    req_indices = np.triu_indices(n_objs, 1)
    rmsd_matrix = np.zeros((n_objs, n_objs))

    for i, j in np.transpose(req_indices):
        rmsd_matrix[i, j] = calc_rmsd_ase_atoms_non_pbs(some_atoms_objs[i], some_atoms_objs[j])

    return rmsd_matrix


def get_orca5_keywords(lines_):
    """
    Parse the INPUT FILES section of an Orca 5 output to get all the simple and detailed keywords which determine
    a calculation specification. This will also read NAME, mutliplicity, charge, coordinates type and coordinates name
    TODO Only support xyzfile currently
    :param lines_: a list of strings
    :type lines_: [str]
    :return: a dictionary with keys ("simple", ... various method names "scf", "mtr", "freq" , etc), a dictionary with
    coordinate type, path, multiplicity and charge and name of the input file
    :rtype: dict, dict, str
    """

    # Supported detailed keywords. NOTE: use lower-case for key (outermost).
    # TODO: May translate into classes in the future
    keywords = {"simple": []}
    name = None
    charge = None
    multiplicity = None
    coord_type = None
    coord_path = None
    idx_to_skip = []
    sub_method_end = False  # For keywords that require multiple values
    current_detailed_kw = None

    flatten_lines = []
    for line_ in lines_:
        if ">" in line_ and "#" not in line_:
            temp = line_.strip("\n").split(">")
            temp = temp[-1].split()
        elif "NAME" in line_:
            temp = line_.strip("\n").split("=")
        elif "#" in line_:
            continue  # exclude line with #. We assume that # will only appear at the start but no check is performed
        else:
            temp = []

        flatten_lines += temp

    start_to_read_simple_kw = False
    current_detailed_kw = None

    idx_to_exclude = []
    for idx, item in enumerate(flatten_lines):
        if idx in idx_to_exclude:
            continue
        elif current_detailed_kw is not None:
            if item.lower() not in keywords[current_detailed_kw].keywords or item.lower() == "end":
                current_detailed_kw = None

        if "!" in item:
            # Simple keywords
            if current_detailed_kw is not None:
                current_detailed_kw = None
            start_to_read_simple_kw = True
            keywords["simple"].append(item.strip("!").upper())

        elif "%" in item:
            start_to_read_simple_kw = False
            # keywords dictionary keys are in lowercase
            if current_detailed_kw is None:
                current_detailed_kw = item.strip("%\n").lower()
                if current_detailed_kw == "method":
                    keywords[current_detailed_kw] = Method()
                elif current_detailed_kw == "geom":
                    keywords[current_detailed_kw] = Geom()
                elif current_detailed_kw == "mtr":
                    keywords[current_detailed_kw] = Mtr()
                elif current_detailed_kw == "freq":
                    keywords[current_detailed_kw] = Freq()
                elif current_detailed_kw == "scf":
                    keywords[current_detailed_kw] = Scf()
                elif current_detailed_kw == "cpcm":
                    if current_detailed_kw not in keywords:
                        keywords[current_detailed_kw] = Cpcm("")
                elif current_detailed_kw == "mp2":
                    keywords[current_detailed_kw] = Mp2()
                else:
                    current_detailed_kw = None
        elif start_to_read_simple_kw:
            keywords["simple"].append(item.strip("!"))

        elif current_detailed_kw is not None:
            # We expect that item to be lowercase for this section
            item_ = item.lower()
            type_casted = False
            if item_ in keywords[current_detailed_kw].keywords.keys():
                if item_ not in keywords[current_detailed_kw].multi_values:
                    idx_to_exclude = [idx + 1]
                    value = flatten_lines[idx + 1]

                    if item_ in keywords[current_detailed_kw].single_value_float:
                        value = float(value)
                        type_casted = True
                    elif item_ in keywords[current_detailed_kw].single_value_int:
                        value = int(value)
                        type_casted = True

                    if type_casted:
                        keywords[current_detailed_kw].keywords[item_] = value
                    else:
                        keywords[current_detailed_kw].keywords[item_] = value.strip("\"\'\n")

                else:
                    value_idx = idx + 1
                    idx_to_exclude = [value_idx]
                    while flatten_lines[value_idx].lower() != "end":
                        keywords[current_detailed_kw].keywords[item_].append(flatten_lines[value_idx])
                        value_idx += 1
                        idx_to_exclude.append(value_idx)

        elif "NAME" in item:
            name = flatten_lines[idx + 1].strip()
            idx_to_exclude = [idx + 1]
        elif "*" in item:  # * marks the beginning of coordinates specification
            coord_type = flatten_lines[idx + 1]
            multiplicity = int(flatten_lines[idx + 2])
            charge = int(flatten_lines[idx + 3])
            if coord_type == "xyzfile":
                coord_path = flatten_lines[idx + 4]
            else:
                coord_path = None
            idx_to_exclude = [idx + 1, idx + 2, idx + 3, idx + 4]

    # Change all simple keywords to lowercase
    keywords["simple"] = [item.upper() for item in keywords["simple"]]

    # basename accounts for the relative path used in Singularity container
    return keywords, {"coord_type": coord_type, "coord_path": path.basename(coord_path),
                      "multiplicity": multiplicity, "charge": charge}, path.basename(name)


if __name__ == "__main__":
    path_ = r"/home/wontleave/calc/autoopt/new_test/methylI_F/run.optTS.r2scan-3c.coronab"
    orca5 = Orca5(path_, 1e-5, allow_incomplete=True, get_opt_steps=True)
    print()