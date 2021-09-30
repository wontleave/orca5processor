from os import path


class Orca5:
    """
    Read a folder that contains data generated from Orca 5.0 and create the relevant ASE objects
    Supported extensions for ORCA5 output are out, wlm01, coronab, lic01 and hpc-mn1
    Try to be as close to ASE as possible
    Maybe compatible with ORCA 4, but use at your own risk
    """
    def __init__(self, orca_out_file,
                 property_path=None,
                 engrad_path=None,
                 output_path=None,
                 safety_check=False):

        # Get the root name of the orca_out_file
        self.root_name = path.basename(orca_out_file)

        self.property_path = property_path  # Extension is .txt. NOT USED FOR NOW 20210930
        self.engrad_path = path.join(self.root_name, ".engrad")  # Extension is .engrad
        self.output_path = orca_out_file  # Extension can be .out, .coronab, .wml01, .lic001
        self.job_types = None  # If job_types is None after processing the given folder, an incomplete job is assumed

        # Only property file -> single point, spectroscopic properties? Elec energy here doesn't include SRM
        with open(orca_out_file, "r") as f:
            lines = f.readlines()

        self.input_section = {"start": 0, "end": 0}
        self.find_input_sections(lines)

        # keywords used in the Orca 5 calculation
        self.keywords = get_orca5_keywords(lines[self.input_section["start"]: self.input_section["end"]])

    def parse_orca5_output(self):


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
                      "aux_general_keywords": ("autoaux")
                      }

# ORCA5 input block
pal_block = {}

# ORCA4 simple keywords
grid_simple_keywords = ("GRID", "GRIDX", "NOFINALGRIDX", "NOFINALGRID")

# Independent function collections
def get_orca5_keywords(lines_):
    """
    Parse the INPUT FILES section of an Orca 5 output to get all the simple and detailed keywords which determine
    a calculation specification.

    Currently, we support the following detailed keywords:
    %method: "MAYER_BONDORDERTHRESH"

    :param lines_: a list of strings
    :type lines_: [str]
    :return: a dictionary with keys: "simple", ... various method names "scf", "mtr", "freq" , etc
    :rtype: dict
    """
    keywords = { "simple": []}
    for line_ in lines_:

        detailed_kw_start = False
        detailed_kw_end = False

        if "#" in line_:
            continue
        elif "!" in line_:
            # Simple keywords
            temp_line = line_.split("!")[-1]
            keywords["simple"] += temp_line.split()
        elif "%" in line_:
            detailed_kw_start = True

        if detailed_kw_start:
