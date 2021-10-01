from orca5processor import Orca5Processor
from reading_utilis import read_root_folders


class Test:
    def __init__(self):
        self.t = ()


class Test2(Test):
    def __init__(self):
        super().__init__()


if __name__ == "__main__":
    root_ = r"E:\TEST\SP_tests\wB97X-V_TZ-ALPB_GFN2"

    folders = []
    read_root_folders(root_, folders)
    print()