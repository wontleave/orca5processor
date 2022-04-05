import os
from os import walk, mkdir, rename
from optparse import OptionParser
import subprocess
import time

parser = OptionParser()
parser.add_option("-f", "--folder", dest="root_folder")
parser.add_option("-j", "--jobname", dest="job_name")

opts, args = parser.parse_args()

allfolders = [x[0] for x in walk(opts.root_folder)]
# remove the parent folder
allfolders.pop(0)
# look for orca input files - assumed to have the extension .inp
# with no corresponding output files - assumed to have the extension .out
required = []

for folder in allfolders:
    _, _, filenames = next(walk(folder), (None, None, []))
    print(folder)
    for filename in filenames:
        if ".inp" in filename and opts.job_name in filename and "multi_sp.inp" not in filename:
            required.append([folder, filename])

# filter according ot filename_of_interest

total = len(required) + 1
progress = 1
for x, y in required:
    fullpath = f"{x}/{y}"
    name = y.split(".")[0] + ".out"

    output = f"{x}/{name}"
    if os.path.isfile(fullpath):
        # submit job
        t_start = time.perf_counter()
        print(f"Running {fullpath}")
        os.chdir(x)
        job = subprocess.Popen([r"/home/user/apps/orca_4_2_1_linux_x86-64_openmpi216/orca", fullpath],
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, universal_newlines=True)
        out, err = job.communicate()
        # with open(name,"w") as f:
        #     f.writelines(out)
        t_end = time.perf_counter()
        print(f"Done it took {t_end - t_start} {progress} out of {total + 1}\n")
        progress += 1
    else:
        print(f"{fullpath} does not exist!\n")
