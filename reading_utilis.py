from os import scandir, path


def read_root_folders(root_folder_path, root_folder, sub_folder_level_to_go=0, exclude_folders=("IRC", "SCAN")):
    """

    :param root_folder_path:
    :type root_folder_path: str
    :param root_folder:
    :type root_folder: list
    :param sub_folder_level_to_go: how many levels from the root to consider
    :type sub_folder_level_to_go: int
    :param exclude_folders: if the folder contains these keyword it is excluded
    :type exclude_folders: tuple
    :return: None
    :rtype: None
    """
    with scandir(root_folder_path) as it:
        for entry in it:
            if entry.is_dir():
                if entry.name.upper() not in exclude_folders:
                    root_folder.append(entry.path)
                # Check for sub_folder(s)
                if sub_folder_level_to_go > 0:
                    get_subfolders(entry.path, root_folder, level=sub_folder_level_to_go)


def get_subfolders(_where, list_of_dir, level=1):
    """

    :param _where: the folder where all the subfolders are
    :type _where: str
    :param list_of_dir: the list of folders in where
    :type list_of_dir: list of str
    :param level: how deep to go into where
    :type level: int
    :return:
    """
    if level == 0:
        with scandir(_where) as it:
            for entry in it:
                if entry.is_dir():
                    if "IRC" not in entry.name.upper() and "SCAN" not in entry.name.upper():
                        list_of_dir.append(entry.path)
        return list_of_dir
    elif level > 0:
        _temp = []
        with scandir(_where) as it:
            for entry in it:
                if entry.is_dir():
                    if "IRC" not in entry.name.upper() and "SCAN" not in entry.name.upper():
                        _temp.append(entry.path)
            level -= 1
            list_of_dir += _temp
            for _folder in _temp:
                get_subfolders(_folder, _temp, level)






