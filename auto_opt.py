import subprocess
import time
import numpy as np
import pandas as pd
import shutil
import paramiko
from os import path, chdir
from pathlib import Path
from orca5_utils import Orca5
from writing_utils import modify_orca_input
from reading_utilis import read_root_folders
from geo_optimization import analyze_ts
from pprint import pprint


class AutoGeoOpt:
    """
    Perform automatic geometry optimization on a single given folder
    Details are read from a spec.txt file in the given folder
    """
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.folder_name = Path(folder_path).stem
        self.spec = {}
        self.inputs = {}
        self.outcomes: dict[str, bool] = {}  # A dict to indicate if the job is successful

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

        COPT, OPT, OPTTS, NUMHESS or ANHESS = the orca input filename

        :return:
        """
        full_path = path.join(self.folder_path, "spec.txt")
        with open(full_path) as f:
            lines = f.readlines()

        for line in lines:
            key, value = line.split("=")
            self.spec[key.strip()] = value.strip()

        for key in self.spec:
            if key.lower() == "job_sequence" or key.lower() == "ts_int_coords":
                self.spec[key] = self.spec[key].split()
            elif key.lower() == "job_max_try":
                self.spec[key] = [int(item) for item in self.spec[key].split()]
            # elif key.lower() == "ts_int_coords":
            #     processed_coords = []
            #     for coords in self.spec[key].split():
            #         coords = coords.split("-")
            #         processed_coords.append(tuple([int(atom) for atom in coords]))
            #
            #     self.spec[key] = processed_coords

        for job_type in self.spec["job_sequence"]:
            self.inputs[job_type] = self.spec[job_type]

    def copy_req_files(self, root_folder):
        """
        In the case if self.identical is True. The required input files and spec.txt will be copied to all subfolders
        :param root_folder: from BatchJobs object
        :type root_folder: str
        :return:
        """
        root = Path(root_folder)
        spec_path = root.joinpath("spec.txt")
        if spec_path.is_file():
            destination_path = (Path(self.folder_path)).joinpath("spec.txt")
            shutil.copyfile(spec_path.resolve(), destination_path)
        else:
            raise Exception("We can't find {spec_path.resolve()}!")

        self.read_spec()
        for key in self.inputs:
            req_input_path = root.joinpath(self.inputs[key])
            if req_input_path.is_file():
                destination_path = (Path(self.folder_path)).joinpath(self.inputs[key])
                shutil.copyfile(req_input_path.resolve(), destination_path)
            else:
                raise Exception(f"We can't find {req_input_path.resolve()}!")

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
            print(f"Performing {task} ...")
            # Bookkeeping
            iter_counter = 0
            converged = False
            if task_idx > 0:
                req_coord_path = Path(self.current_input_path).stem + ".xyz"
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
                    self.outcomes[task] = True
                else:
                    print(f"{task} failed!")
                    self.outcomes[task] = False
            else:
                for i in range(self.spec["job_max_try"][task_idx]):
                    output_path, time_taken = self.run_job(task)
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
                        self.outcomes[task] = True
                        break

                    if i == 0:
                        req_coord_path = Path(self.current_input_path).stem + ".xyz"
                        input_spec = {"xyz_path": req_coord_path}
                        modify_orca_input(self.current_input_path, **input_spec)

                    if task.upper() == "OPTTS":
                        print("Performing OPTTS analysis")
                        _, input_spec = analyze_ts(orca5, self.spec["ts_int_coords"], self.spec["recalc_hess_min"])
                        if input_spec is None:
                            converged = False
                            break
                        else:
                            if "recalc_hess" in input_spec:
                                current_recalc_hess = input_spec["recalc_hess"]
                                print(f"Current recalc_hess is {current_recalc_hess}")
                            elif "maxiter" in input_spec:
                                current_max_iter = input_spec["maxiter"]
                                print(f"Current max iteration is {current_max_iter}")

                            modify_orca_input(self.current_input_path, **input_spec)

                if not converged:
                    print(f"---------- {task} has failed after {iter_counter} cycles(s). "
                          f"Total time taken is {np.sum(self.time_taken):.2f}s")
                    print("We won't proceed to the next task!")
                    self.outcomes[task] = False
                    break

        # Create the dataframe for time taken bookkeeping
        tuples = list(zip(*[self.job_type_labels, self.iter_labels]))
        index = pd.MultiIndex.from_tuples(tuples, names=["Job Type", "Iter"])
        self.times_df = pd.DataFrame(self.time_taken, index=index, columns=["Time (s)"])


class BatchJobs:
    """
    Perform a batch jobs by creating a list of AutoGeoOpt object
    """
    def __init__(self, root_folder, identical=False):
        # Variables from arguments
        self.root_folder = root_folder
        self.identical = identical  # If all the batch jobs have the same sequence and input files
        self.all_folder = []

        read_root_folders(root_folder, self.all_folder)
        self.auto_jobs = [AutoGeoOpt(folder) for folder in self.all_folder]

    def run_batch_jobs(self):
        for idx, job in enumerate(self.auto_jobs):
            print(f"****** Job {idx + 1} -- {job.folder_path} ....")
            if self.identical:
                job.copy_req_files(self.root_folder)
            job.read_spec()
            job.perform_tasks()
            pprint(job.times_df)
            print(f"{job.times_df.sum()}")

    def separate_failed(self):
        """

        :return:
        """
        for job in self.auto_jobs:
            for key in job.outcomes:
                if not job.outcomes[key]:
                    destination = Path(self.root_folder).joinpath(key)
                    if not destination.is_dir():
                        destination.mkdir()
                    full_path = Path(destination).joinpath(job.folder_name)
                    shutil.move(job.folder_path, full_path.resolve())

class RemoteBatchJobs(BatchJobs):
    """
    Uses paramiko to connect to an HPC cluster and manage the sending and processing of jobs
    """
    def __init__(self):

        # Control connection

if __name__ == "__main__":
    # req_folder = r"/home/wontleave/calc/autoopt/new_test"
    # batch = BatchJobs(req_folder, identical=True)
    # batch.run_batch_jobs()
    # batch.separate_failed()

    # Remote Batch jobs test
