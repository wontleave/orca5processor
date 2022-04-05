import subprocess
import time
from os import path, chdir
from pathlib import Path
from orca5_utils import Orca5
from writing_utils import modify_orca_input


class AutoGeoOpt:
    """
    Perform automatic geometry optimization on a single given folder
    Details are read from a spec.txt file in the given folder
    """
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.spec = {}
        self.inputs = {}
        self.time_taken = None
        self.read_spec()

        # Dynamics variables that change during the run
        self.current_input_path: str = None

    def read_spec(self):
        """
        Provides the command to execute ORCA5,
        a list of job sequence:
            [COPT OPTS ANHESS] -> This will provide a constrained optimization, then pass the contrained coordinates
                                    to OPTTS as additional internal coordinates, a AnHess will be run at the end
            [COTP, AnHess-OPTTS AnHess] -> The AnHess for OPTTS is ran as a separate job
        :return:
        """
        full_path = path.join(self.folder_path, "spec.txt")
        with open(full_path) as f:
            lines = f.readlines()

        for line in lines:
            key, value = line.split("=")
            self.spec[key.strip()] = value.strip()

        for key in self.spec:
            if key.upper() == "COPT" or key.upper() == "OPTTS" or key.upper() == "NUMHESS" or key.upper() == "ANHESS":
                self.inputs[key.upper()] = self.spec[key]

    def run_job(self, job_type):
        """
        Execute the calculation
        :param job_type: the key to find the input file from self.inputs
        :type job_type: str
        :return: the full outpath for analysis and the time taken
        :rtype: tuple[str, float]
        """
        t_start = time.perf_counter()
        chdir(self.folder_path)
        self.current_input_path = path.join(self.folder_path, self.inputs[job_type])
        filename = Path(self.inputs[job_type]).stem + ".coronab"
        output_path_ = path.join(self.folder_path, filename)
        command = self.spec["run_command"]
        command = rf"{command}"
        job = subprocess.Popen([command, self.current_input_path],
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, universal_newlines=True)
        out, err = job.communicate()
        with open(output_path_, "w") as f:
            f.writelines(out)
        t_end = time.perf_counter()
        return output_path_, t_end - t_start

    def perform_task(self):
        """

        :return:
        """
        for task in self.spec["job_sequence"]:
            self.run_job(task)
            req_coord_path = Path(auto.current_input_path).stem + ".xyz"
            input_spec = {"xyz_path": req_coord_path}
            modify_orca_input(auto.current_input_path, **input_spec)


if __name__ == "__main__":
    req_folder = r"/home/wontleave/calc/TCH_SCIENCE/TS_S_SN2/0"
    auto = AutoGeoOpt(req_folder)
    current_job_type = "COPT"
    output_path, time_taken = auto.run_job(current_job_type)
    orca5 = Orca5(output_path, 2e-5, allow_incomplete=True, get_opt_steps=True)
    req_coord_path = Path(auto.current_input_path).stem + ".xyz"
    input_spec = {"xyz_path": req_coord_path}
    modify_orca_input(auto.current_input_path, **input_spec)
    run_count = 1
    while not orca5.job_type_objs["OPT"].converged:
        output_path, time_taken = auto.run_job(current_job_type)
        orca5 = Orca5(output_path, 2e-5, allow_incomplete=True, get_opt_steps=True)
        print(f"Run {run_count} took {time_taken:.2f} s")
        run_count += 1
