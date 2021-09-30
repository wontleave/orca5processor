from orca5processor import Orca5Processor
from reading_utilis import read_root_folders

if __name__ == "__main__":
    root_ = r"E:\TEST\SP_tests\wB97X-V_TZ-ALPB_GFN2"

    folders = []
    read_root_folders(root_, folders)
    print()