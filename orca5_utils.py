import re
import struct
import tempfile
from os import path
from ase import Atoms
import pathlib


# Classes related to Detailed Keywords
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
                         "scalmp2c": None, "scalldac": None,"scalhfx": None,
                         "method": None,
                         "mayer_bondorderthres": None,
                         "ri": None
                        }

        self.single_value_str = ("exchange", "correlation", "runtyp", "functional", "method", "ri")
        self.single_value_float = ("mayer_bondorderthres", "scalmp2c", "scalldac", "scalhfx")
        self.single_value_int = ()
        self.multi_values = ()  # Multi_values keywords required an "end" to terminate

        self.cpcm = cpcm


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
    def __init__(self, solvent):
        """
        Solvation will be covered under Method
        """
        self.name = "cpcm"
        self.detail_read = False  # CPCM details appear at every SCF cycle. This flag causes it to be read only once
        self.solvent = solvent  # The CPCM solvent
        self.keywords = {"epsilon": None, "refrac": None, "rsolv": None, "rmin": None, "pmin": None,
                         "surfacetype": None, "fepstype": None, "xfeps": None,
                         "ndiv": None, "num_leb": None,
                         "smd": None, "smdsolvent": None
                        }

        self.single_value_str = ("surfacetype", "fepstype", "smd", "smdsolvent")
        self.single_value_float = ("epsilon", "refrac", "rmin", "pmin", "rsolv", "xfeps")
        self.single_value_int = ("ndiv", "num_leb")
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
                        self.keywords["surfacetype"] = float(line.split("... ")[-1])
                    elif "Epsilon function type                           ..." in line:
                        self.keywords["fepstype"] = float(line.split()[-1])


class Geom:
    def __init__(self):
        """

        """
        self.name = "geom"
        self.keywords = {"maxiter": None,
                         "gdiismaxeq": None,
                         "numhess": None,
                         "recalc_hess": None,
                         "trust": None,
                         "scan": [],
                         "modify_internals": [],
                         "constraints": []}

        self.single_value_str = ("numhess", "correlation")
        self.single_value_float = ("trust")
        self.single_value_int = ("maxiter", "gdiismaxeq", "recalc_hess")
        # Multi_values keywords required an "end" to terminate
        self.multi_values = ("scan", "modify_internals", "constraints")

        # ASE Atoms object of the optimized geometry. Both has to be equal
        self.opt_geo_from_output = None
        self.opt_geo_from_xyz = None

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

        for line in lines_:
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
                        start_to_read_xyz += 1
                    else:
                        start_to_read_failed_attempt += 1
                    continue
            if "THE OPTIMIZATION HAS CONVERGED" in line:
                start_to_read_xyz += 1
                continue
            if "CARTESIAN COORDINATES (ANGSTROEM)" in line and start_to_read_xyz:
                start_to_read_xyz += 1

        self.opt_geo_from_output = Atoms(elements_seq, positions=cart_coords)


class Mtr:
    def __init__(self):
        self.name ="mtr"
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
        self.keywords ={"temp": None,
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

    def get_thermochemistry(self, lines_, with_opt=False):
        """

        :param lines_:
        :type lines_:
        :param with_opt: indicate whether a Freq job is part of an opt or optTS job, If true, the Freq information will
        only be obtained after the geometry optimization has converged.
        :type with_opt: bool
        :return:
        :rtype:
        """

        start_to_read_freq = False
        start_to_read_thermo = False
        freq_read = False

        if with_opt:
            successful_opt = False
        else:
            successful_opt = True

        for line in lines_:
            if "THE OPTIMIZATION HAS CONVERGED" in line:
                successful_opt = True
                continue
            if "VIBRATIONAL FREQUENCIES" in line and successful_opt:
                start_to_read_freq = True
                continue
            if "THERMOCHEMISTRY AT" in line and successful_opt:
                start_to_read_thermo = True
                continue

            if start_to_read_freq and "cm**-1" in line:
                try:
                    _, frequency, _ = line.split()
                    self.frequencies.append(float(frequency))
                except ValueError:
                    start_to_read_freq = False
                    freq_read = True
            elif start_to_read_thermo:
                if "Temperature         ..." in line:
                    try:
                        self.temp = float(line.split()[2])
                    except ValueError:
                        raise ValueError(f"{line} -- does not contain the temperature at the expected index")
                elif "Pressure            ..." in line:
                    try:
                        self.pressure = float(line.split()[2])
                    except ValueError:
                        raise ValueError(f"{line} -- does not contain the pressure at the expected index")
                elif "Total Mass          ..." in line:
                    try:
                        self.total_mass = float(line.split()[-2])
                    except ValueError:
                        raise ValueError(f"{line} -- does not contain the total mass at the expected index")
                elif "Electronic energy                ..." in line:
                    try:
                        self.elec_energy = float(line.split()[-2])
                    except ValueError:
                        raise ValueError(f"{line} -- does not contain the electronic energy at the expected index")
                elif "Zero point energy                ..." in line:
                    try:
                        self.zero_pt_energy = float(line.split()[-4])
                    except ValueError:
                        raise ValueError(f"{line} -- does not contain the zero point energy at the expected index")
                elif "Total thermal correction" in line:
                    try:
                        self.total_thermal_correction = float(line.split()[-4])
                    except ValueError:
                        raise ValueError(f"{line} -- does not contain the total thermal correction"
                                         f" at the expected index")
                elif "Total thermal energy" in line:
                    try:
                        self.total_thermal_energy = float(line.split()[-2])
                    except ValueError:
                        raise ValueError(f"{line} -- does not contain the total thermal energy"
                                         f" at the expected index")
                elif "Thermal Enthalpy correction       ..." in line:
                    try:
                        self.thermal_enthalpy_correction = float(line.split()[-4])
                    except ValueError:
                        raise ValueError(f"{line} -- does not contain the thermal enthalpy correction"
                                         f" at the expected index")
                elif "Total Enthalpy                    ..." in line:
                    try:
                        self.total_enthalpy = float(line.split()[-2])
                    except ValueError:
                        raise ValueError(f"{line} -- does not contain the total enthalpy"
                                         f" at the expected index")
                elif "Final entropy term                ..." in line:
                    try:
                        self.final_entropy_term = float(line.split()[-4])
                    except ValueError:
                        raise ValueError(f"{line} -- does not contain the final entropy term"
                                         f" at the expected index")
                elif "Final Gibbs free energy         ..." in line:
                    try:
                        self.final_gibbs_free_energy = float(line.split()[-2])
                    except ValueError:
                        raise ValueError(f"{line} -- does not contain the final Gibbs free energy"
                                         f" at the expected index")


class Scf:
    def __init__(self):
        """
        This class will contain the keywords that specify a SCF calculation.
        It will contain the results of a SCF calculation:
        Final single point energy in a SP. Final single point of the optimized structure in OPT and OPTTS.

        """
        self.name ="scf"
        self.keywords = {"convergence": None,
                         "autotrah": None,
                         "stabperform": None,
                         "tole": None,
                         "tolg": None,
                         "sthresh": 1E-8}
        self.all_keys = [key.lower() for key in self.keywords.keys()]
        self.single_value_str = ("convergence", "autotrah", "stabperform")
        self.single_value_float = ("tole", "tolg", "sthresh")
        self.single_value_int = ()
        # Multi_values keywords required an "end" to terminate
        self.multi_values = ()

        self.final_sp_energy = None

    def get_final_sp_energy(self, lines):
        """

        :return:
        :rtype:
        """
        final_sp_expr = re.compile("FINAL SINGLE POINT ENERGY")
        for line in lines:
            if "FINAL SINGLE POINT ENERGY" in line:
                try:
                    self.final_sp_energy = float(line.split()[-1])
                except ValueError:
                    raise ValueError(f"Cannot cast {line.split()[-1]} as a float ! check your output!")


# END of classes related to detailed keyword
class Orca5:
    """
    Read a folder that contains data generated from Orca 5.0 and create the relevant ASE objects
    Supported extensions for ORCA5 output are out, wlm01, coronab, lic01 and hpc-mn1
    Molecules are created as ASE Atoms object
    !!!Maybe compatible with ORCA 4, but use at your own risk
    """
    def __init__(self, orca_out_file,
                 property_path=None,
                 engrad_path=None,
                 output_path=None,
                 safety_check=False):

        # Get the root name of the orca_out_file
        self.root_name = None

        self.property_path = property_path  # Extension is .txt. NOT USED FOR NOW 20210930
        self.engrad_path = engrad_path  # Extension is .engrad
        self.output_path = orca_out_file  # Extension can be .out, .coronab, .wml01, .lic001
        self.job_types = []  # If job_types is None after processing the given folder
        # Each job type will have its own object
        # SP: Scf
        # ENGRAD: Scf
        # OPT: Scf + Geom + Freq(optional, depends on whether there is a frequency calc after a sucessful opt)
        # OPTTS: Scf + Geom + Freq(optional, depends on whether there is a frequency calc after a sucessful opt)
        # Freq: Scf + Freq
        # IRC: Scf + IRC
        self.job_type_objs = {}
        self.method = None  # All job type must have a Method object
        self.basis = None  # All job type must have a Basis object
        # an incomplete/unsupported job is assumed

        # Only property file -> single point, spectroscopic properties? Elec energy here doesn't include SRM
        with open(orca_out_file, "r") as f:
            lines = f.readlines()

        self.input_section = {"start": 0, "end": 0}
        self.find_input_sections(lines)

        # keywords used in the Orca 5 calculation
        self.keywords, self.coord_spec, self.input_name = \
            get_orca5_keywords(lines[self.input_section["start"]: self.input_section["end"]])

        self.root_name, _ = path.splitext(self.input_name)
        self.determine_jobtype()
        self.parse_orca5_output(lines[self.input_section["end"]:])

    def parse_orca5_output(self, lines_):
        """
        job_types will determine what information to extract
        SP: only final single point energy
        Opt:
        :return:
        :rtype:
        """
        for job_type in self.job_types:
            if job_type == "SP":
                self.job_type_objs["SP"].get_final_sp_energy(lines_)
            elif job_type == "OPT":
                self.job_type_objs["OPT"].get_opt_geo(lines_)
                # TODO read from xyz as an option?
            elif job_type in ("FREQ", "ANFREQ", "NUMFREQ"):
                self.job_type_objs["FREQ"].get_thermochemistry(lines_)
            elif job_type == "CPCM":
                self.job_type_objs["CPCM"].get_cpcm_details(lines_)

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

        if "OPT" in self.keywords["simple"]:
            self.job_types.append("OPT")
        elif "OPTTS" in self.keywords["simple"]:
            self.job_types.append("OPTTS")

        if "FREQ" in self.keywords["simple"]:
            self.job_types.append("FREQ")
        elif "ANFREQ"  in self.keywords["simple"]:
            self.job_types.append("ANFREQ")
        elif "NUMFREQ" in self.keywords["simple"]:
            self.job_types.append("NUMFREQ")

        if "IRC" in self.keywords["simple"]:
            self.job_types.append("IRC")
        elif "ENGRAD" in self.keywords["simple"]:
            self.job_types.append("ENGRAD")

        solvent = None
        req_kw = None
        sanity_check = 0
        for kw in self.keywords["simple"]:
            req_kw = re.search("CPCM", kw)
            if req_kw is not None:
                self.job_types.append("CPCM")
                solvent = req_kw.string.split("(")[-1].strip(")")
                sanity_check += 1
        assert sanity_check == 1, f"You have more than one CPCM simple keyword in the ORCA5 input"
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
            if not is_single_point:
                self.job_types = None
            else:
                self.job_types.append("SP")

        # Create or copy the required job_type objs
        if self.job_types is not None:
            if "basis" not in self.keywords.keys():
                self.basis = Basis()
                self.basis.process_simple_keywords(self.keywords["simple"])

            for item in self.job_types:
                if item == "SP":
                    if "method" not in self.keywords.keys():
                        self.method = Method(run_type="SP")
                    else:
                        self.method = self.keywords["method"]
                        self.method.keywords["runtyp"] = "SP"

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

                elif item == "OPT":
                    if "method" not in self.keywords.keys():
                        self.method = Method(run_type="OPT")
                    else:
                        self.method = self.keywords["method"]
                        self.method.keywords["runtyp"] = "OPT"

                    if "geom" not in self.keywords.keys():
                        self.job_type_objs["OPT"] = Geom()
                    else:
                        self.job_type_objs["OPT"] = self.keywords["geom"]

                elif item in ('FREQ', "ANFREQ", "NUMFREQ"):
                    if "freq" not in self.keywords.keys():
                        self.keywords["freq"] = Freq()
                        if item == "ANFREQ":
                            self.keywords["freq"].keywords["anfreq"] = True
                        elif item == "NUMFREQ":
                            self.keywords["freq"].keywords["numfreq"] = True
                        self.job_type_objs["FREQ"] = self.keywords["freq"]
                    else:
                        self.job_type_objs["FREQ"] = self.keywords["freq"]

                elif item == "CPCM":
                    if "cpcm" not in self.keywords.keys():
                        self.keywords["CPCM"] = Cpcm(solvent)

                    if "method" not in self.keywords.keys():
                        self.method = Method(cpcm=self.keywords["cpcm"])
                    else:
                        self.method.cpcm =  self.keywords["cpcm"]



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


# Independent function collections
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
        if "!" in item:
            # Simple keywords
            start_to_read_simple_kw = True
            keywords["simple"].append(item.strip("!").upper())
        elif "%" in item:
            start_to_read_simple_kw = False
            if current_detailed_kw is None:
                current_detailed_kw = item.strip("%").lower()
                if current_detailed_kw == "method":
                    keywords["method"] = Method()
                elif current_detailed_kw == "geom":
                    keywords["geom"] = Geom()
                elif current_detailed_kw == "mtr":
                    keywords["mtr"] = Mtr()
                elif current_detailed_kw == "freq":
                    keywords["freq"] = Freq()
                elif current_detailed_kw == "scf":
                    keywords["scf"] = Scf()
                else:
                    current_detailed_kw = None
        elif start_to_read_simple_kw:
            keywords["simple"].append(item.strip("!"))
        elif current_detailed_kw is not None:
            if item in keywords[current_detailed_kw].keywords.keys():
                if item not in keywords[current_detailed_kw].multi_values:
                    idx_to_exclude = [idx + 1]
                    value = flatten_lines[idx + 1]

                    if item in keywords[current_detailed_kw].single_value_float:
                        value = float(value)
                    elif item in keywords[current_detailed_kw].single_value_int:
                        value = int(value)

                    keywords[current_detailed_kw].keywords[item] = value
                    current_detailed_kw = None
                else:
                    value_idx = idx + 1
                    idx_to_exclude = [value_idx]
                    while flatten_lines[value_idx].lower() != "end":
                        keywords[current_detailed_kw].keywords[current_detailed_kw].append(flatten_lines[value_idx])
                        value_idx += 1
                        idx_to_exclude.append(value_idx)
                    current_detailed_kw = None
        elif "NAME" in item:
            name = flatten_lines[idx + 1]
            idx_to_exclude = [idx + 1]
        elif "*" in item:  # * marks the beginning of coordinates specfification
            coord_type = flatten_lines[idx + 1]
            multiplicity = int(flatten_lines[idx + 2])
            charge = int(flatten_lines[idx + 2])
            coord_path = flatten_lines[idx + 3]
            idx_to_exclude = [idx + 1, idx + 2, idx + 3]

    # Change all simple keywords to lowercase
    keywords["simple"] = [item.upper() for item in keywords["simple"]]
    return keywords, {"coord_type": coord_type, "coord_path": coord_path,
                      "multiplicity": multiplicity, "charge": charge}, name








