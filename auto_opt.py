import subprocess
import time
import numpy as np
import pandas as pd
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
        self.read_spec()

        # Dynamics variables that change during the run
        self.current_input_path: str = None

        # For creation of multi-index dataframe to manage time taken
        self.time_taken: list[float] = []
        self.job_type_labels: list[str] = []
        self.iter_labels: list[int] = []
        self.times_df = None

        # Determine the state of the job
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
            if key.lower() == "job_sequence":
                self.spec[key] =  self.spec[key].split()
            elif key.lower() == "job_max_try":
                self.spec[key] = [int(item) for item in self.spec[key].split()]

        for job_type in self.spec["job_sequence"]:
            self.inputs[job_type] = self.spec[job_type]

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
        TODO determine if there is a need to include a prefix for the xyzfile (i.e. in singularity container)
        Perform the task(s) as given by the job_sequence key in spec.txt
        :return:
        """
        for task_idx, task in enumerate(self.spec["job_sequence"]):
            print(f"Performing {task} ...")
            # Bookkeeping
            iter_counter = 0
            if task_idx > 0:
                req_coord_path = Path(auto.current_input_path).stem + ".xyz"
                input_spec = {"xyz_path": req_coord_path}
                next_input_path = path.join(self.folder_path, self.inputs[task])
                modify_orca_input(next_input_path, **input_spec)

            if "HESS" in task:
                self.run_job(task)
                output_path, time_taken = self.run_job(task)
                orca5 = Orca5(output_path, 2e-5, allow_incomplete=False, get_opt_steps=False)

                # Bookkeeping
                iter_counter += 1
                self.job_type_labels.append(task)
                self.iter_labels.append(iter_counter)
                self.time_taken.append(time_taken)

                if orca5.completed_job:  # A valid output file needs to have ORCA TERMINATED NORMALLY
                    print(f"{task} is done in {time_taken:.2f}")
                else:
                    print(f"{task} failed!")
            else:
                converged = False
                output_path, time_taken = self.run_job(task)
                orca5 = Orca5(output_path, 2e-5, allow_incomplete=True, get_opt_steps=True)

                # Bookkeeping
                iter_counter += 1
                print(f"Run {iter_counter} took {time_taken:.2f} s")
                self.job_type_labels.append(task)
                self.iter_labels.append(iter_counter)
                self.time_taken.append(time_taken)

                if orca5.job_type_objs["OPT"].converged:
                    print(f"---------- {task} is completed in {iter_counter} cycles(s). "
                          f"Total time taken is {np.sum(self.time_taken):.2f}s")
                    continue
                else:
                    req_coord_path = Path(auto.current_input_path).stem + ".xyz"
                    input_spec = {"xyz_path": req_coord_path}
                    modify_orca_input(auto.current_input_path, **input_spec)

                for i in range(self.spec["job_max_try"][task_idx]):
                    output_path, time_taken = auto.run_job(task)
                    orca5 = Orca5(output_path, 2e-5, allow_incomplete=True, get_opt_steps=True)
                    iter_counter += 1
                    print(f"Run {iter_counter} took {time_taken:.2f} s")
                    self.job_type_labels.append(task)
                    self.iter_labels.append(iter_counter)
                    self.time_taken.append(time_taken)
                    if orca5.job_type_objs["OPT"].converged:
                        print(f"---------- {task} is completed in {iter_counter} cycles(s). "
                              f"Total time taken is {np.sum(self.time_taken):.2f}s")
                        converged = True
                        break

                if not converged:
                    print(f"---------- {task} has failed after {iter_counter} cycles(s). "
                          f"Total time taken is {np.sum(self.time_taken):.2f}s")
                    print("We won't proceed to the next task!")
                    break

        # Create the dataframe for time taken bookkeeping
        tuples = list(zip(*[self.job_type_labels, self.iter_labels]))
        index = pd.MultiIndex.from_tuples(tuples, names=["Job Type", "Iter"])
        self.times_df = pd.DataFrame(self.time_taken, index=index, columns=["Time (s)"])


if __name__ == "__main__":
    req_folder = r"/home/wontleave/calc/autoopt/methylBr_F"
    auto = AutoGeoOpt(req_folder)
    auto.perform_tasks()
    print(auto.times_df)
    print(f"{auto.times_df.sum()}")

