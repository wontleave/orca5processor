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
            elif key.lower() == "job_sequence":
                self.spec[key] =  self.spec[key].split()
            elif key.lower() == "job_max_try":
                self.spec[key] = [int(item) for item in self.spec[key].split()]

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

    def perform_tasks(self):
        """
        TODO add detail of each task: OPT - convergence criteria, OPTTS - OPT + required int coords
        Perform the task(s) as given by the job_sequence key in spec.txt
        :return:
        """
        for task_idx, task in enumerate(self.spec["job_sequence"]):
            if "HESS" in task:
                total_time_taken = 0.0
                self.run_job(task)
                output_path, time_taken = self.run_job(task)
                total_time_taken += time_taken
                orca5 = Orca5(output_path, 2e-5, allow_incomplete=False, get_opt_steps=False)
                if orca5.completed_job:  # A valid output file needs to have ORCA TERMINATED NORMALLY
                    print(f"{task} is done in {total_time_taken}")
                else:
                    print(f"{task} failed!")
            else:
                print(f"Performing {task} ...")
                total_time_taken = 0.0
                converged = False
                output_path, time_taken = self.run_job(task)
                total_time_taken += time_taken
                orca5 = Orca5(output_path, 2e-5, allow_incomplete=True, get_opt_steps=True)
                req_coord_path = Path(auto.current_input_path).stem + ".xyz"
                input_spec = {"xyz_path": req_coord_path}
                modify_orca_input(auto.current_input_path, **input_spec)
                run_count = 1
                for i in range(self.spec["job_max_try"][task_idx]):
                    output_path, time_taken = auto.run_job(task)
                    orca5 = Orca5(output_path, 2e-5, allow_incomplete=True, get_opt_steps=True)
                    print(f"Run {run_count} took {time_taken:.2f} s")
                    run_count += 1
                    if orca5.job_type_objs["OPT"].converged:
                        converged = True
                        break
                if converged:
                    print(f"---------- {task} completes in {run_count} cycles(s). "
                          f"Total time taken is {total_time_taken:.2f}s")
                else:
                    print(f"---------- {task} failed after {run_count} cycles(s). "
                          f"Total time taken is {total_time_taken:.2f}s")
                    print("We won't proceed to the next task!")
                    break


if __name__ == "__main__":
    req_folder = r"/media/tch/7b8ddedc-3e55-4de2-a2e7-d4aa99ad1f9d/calc/AUTO_TEST/DEBUG/1"
    auto = AutoGeoOpt(req_folder)
    auto.perform_tasks()

