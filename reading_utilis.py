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


def process_orca_folder(_folder, custom_excel_filename=None, write_excel=True, embed_chiral=False,
                        read_red_int_coord=False, sub_folder_level=0):
    """
    This function reads all the molecules (assumed to be in a separate folder) in a given folder
    to create instances of Orca5 objects. It combines the single point(s) and their corresponding
    optimized geometry job. The results is then written to an excel spreadsheet.

    Args:
    :param _folder: the main folder in which all molecules relevant to a reaction path resides
    :type _folder: str

    Kwargs:
    :param custom_excel_filename:
    :type custom_excel_filename: str
    :param write_excel: the flag to indicate whether to write the excel file
    :type write_excel: bool
    :param embed_chiral: For conformational analysis this has to be True
    :type embed_chiral: bool
    :param read_red_int_coord: Flag to control whether redundant internal coordinates used in optimization are read
    :type read_red_int_coord: bool
    :param sub_folder_level: controls what level of sub folder relative to the root will be read
    :type sub_folder_level: int

    :return: list of orca4 objects, str
    """

    # read all the folders
    required_folders = []
    orca_objects = []
    excel_filename = path.basename(_folder) + ".xlsx"
    excel_filename = path.join(_folder, excel_filename)

    # Read all the files and folders in the given folder

    if sub_folder_level >= 0:
        with scandir(_folder) as it:
            for entry in it:
                if entry.is_dir():
                    if "IRC" not in entry.name.upper() and "SCAN" not in entry.name.upper():
                        required_folders.append(entry.path)
                    # Check for sub_folder(s)
                    if sub_folder_level > 0:
                        get_subfolders(entry.path, required_folders, level=sub_folder_level)
    else:
        required_folders.append(_folder)

    output_files_count = 0
    incomplete_jobs = []
    incomplete_jobs_count = 0
    for item in required_folders:
        files = listdir(item)
        print(f"In {item}")
        output_files = [path.join(item, file) for file in files if
                        "wlm01" in file or "coronab" in file or "lic01" in file or "hpc-mn1" in file]
        print(f"Found {len(output_files)} potential ORCA output files")
        for file in output_files:
            output_files_count += 1
            print(file)
            orca_out = Orca4(file, embed_chiral=embed_chiral, read_red_int_coord=read_red_int_coord)
            assert orca_out is not None, f"{file} cannot be processed please check!"
            if orca_out.incomplete_job:
                incomplete_jobs.append(item)
                incomplete_jobs_count += 1
            else:
                orca_objects.append(copy.deepcopy(orca_out))
        if output_files_count > 1:
            print("------------------------------------------------------------------------------------*")
            print(f"WARNING! {item} has {output_files_count} potential output files")
            if incomplete_jobs_count > 0:
                print(f"FATAL ERROR!!! {item} has {incomplete_jobs_count} incomplete jobs in !")
                for incomplete_job in incomplete_jobs:
                    print(incomplete_jobs)
            print("*-----------------------------------------------------------------------------------*")

        output_files_count = 0
        incomplete_jobs = []
        incomplete_jobs_count = 0

    assert len(orca_objects) > 0, f"Empty! Check folder or folder path -- {_folder}"
    assert incomplete_jobs_count == 0, "Some incomplete jobs are found, see above"

    # Check if there is any geometry optimization amongst the Orca4 objects
    no_opt = True
    no_freq = True
    # Combine the Orca4 objects with the same name_of_struct
    opt_and_sp_map = {}  # dictionary to map object with opt to their single point.
    opt_and_nmr_map = {}

    for i, orca in enumerate(orca_objects):
        if orca.has_opt:
            opt_and_sp_map[i] = []
            opt_and_nmr_map[i] = []
            no_opt = False
            print(f"Processing {orca.name_of_struct}")
            if orca.has_freq:
                no_freq = False
            for j in range(len(orca_objects)):
                if i != j and orca.name_of_struct == orca_objects[j].name_of_struct:
                    if orca_objects[j].has_additional_single_points:
                        for sp in orca_objects[j].methods:
                            orca.extra_single_points.append(sp)
                            orca.has_additional_single_points = True
                            opt_and_sp_map[i].append(j)
                    if orca_objects[j].has_freq:
                        no_freq = False
                        orca.extra_thermo = copy.deepcopy(orca_objects[j].extra_thermo)
                    if orca_objects[j].has_nmr:
                        orca.nmrs = copy.deepcopy(orca_objects[j].nmrs)
                        opt_and_nmr_map[i].append(j)
                        orca.has_nmr = True

            print(f"{orca.name_of_struct} is successful")

    warnings = []

    # Delete the extra objects. Do not delete if there is no geometry optimization
    if no_opt:
        available_names = []
        new_orca_objs = []
        for item in orca_objects:
            if item.name_of_struct[0] not in available_names:
                available_names.append(item.name_of_struct[0])

        available_names = list(set(available_names))

        for idx, name in enumerate(available_names):
            new_orca_objs.append(Orca4(name, read_output=False))
            for item in orca_objects:
                if item.name_of_struct[0] == name:
                    for method in item.methods:
                        new_orca_objs[idx].extra_single_points.append(copy.deepcopy(method))
                    new_orca_objs[idx].unopt_geometry = item.unopt_geometry

        orca_objects = new_orca_objs
    else:
        # remove Orca objects
        idx_of_obj_for_removal = []
        for key in opt_and_sp_map:
            idx_of_obj_for_removal += opt_and_sp_map[key]

        for key in opt_and_nmr_map:
            idx_of_obj_for_removal += opt_and_nmr_map[key]

        idx_of_obj_for_removal.sort()

        for i in range(len(orca_objects) - 1, -1, -1):
            # if not orca_objects[i].has_opt:  # remove objects with only freq or single point
            #     if orca_objects[i].has_additional_single_points or orca_objects[i].has_freq:
            #         orca_objects.pop(i)
            #     elif len(orca_objects[i].geo_opt) == 1 and orca_objects[i].geo_opt[0] is None:
            #         orca_objects.pop(i)
            # elif not orca_objects[i].has_additional_single_points and orca_objects[i].has_opt:
            #     warnings.append(f"WARNING! {orca_objects[i].name_of_struct[0]} "
            #                     f"does not have any additional single point!")
            if i in idx_of_obj_for_removal:
                orca_objects.pop(i)

    n_orcas_before_sorting = len(orca_objects)
    labels_before_sorting = [x.name_of_struct[0] for x in orca_objects]
    print(f"number of orca objects left is {n_orcas_before_sorting}")
    # Populate the methods with extra single points
    for item in orca_objects:
        for sp in item.extra_single_points:
            item.methods.append(sp)
    # -----------------------------------------------Sorting section --------------------------------------------------
    # Check if name_of_struct can all be casted to an integer. If true, sort the orca_objects
    idx_to_obj = {}
    all_names_are_int = True

    for item in orca_objects:
        try:
            index = int(item.name_of_struct[0])
            idx_to_obj[index] = item
        except ValueError:
            all_names_are_int = False
            break

    # Check if name is of the following format ***CXX where XX is any integer
    conformation_format = True
    for item in orca_objects:
        try:
            temp = item.name_of_struct[0].split("C")
            index = int(temp[-1])
            idx_to_obj[index] = item
        except ValueError:
            conformation_format = False
            break

    # Check if name is of the following format {type of stationary point}(reaction class, step,
    # conformation number, stereochemistry)
    no_of_stereo_format = 0
    no_of_func_format = 0
    sub_section = False
    stereo_labels = {}
    all_indices = []
    steps_info = []
    stereo_info = []
    unique_steps = None

    for item in orca_objects:
        temp = item.name_of_struct[0].split(",")
        if len(temp) > 1:
            try:
                # Works if not stereochemistry label exists
                last_entry = temp[-1].strip(")")
                index = int(last_entry)
                idx_to_obj[index] = item
                all_indices.append(last_entry)
                steps_info.append(temp[-2])
                stereo_info.append(None)
                no_of_func_format += 1
            except ValueError:
                try:
                    stereo = temp[-1].strip(")")
                    if stereo not in stereo_labels:
                        stereo_labels[stereo] = {}
                    index = int(temp[-2])
                    all_indices.append(index)
                    steps_info.append(temp[-3])
                    stereo_info.append(stereo)
                    stereo_labels[stereo][index] = item
                    no_of_stereo_format += 1
                except ValueError:
                    break

    if no_of_func_format > 0 or no_of_stereo_format >= 0:
        # Check for possible duplicates
        len_before = len(all_indices)
        len_after = len(set(all_indices))
        n_unique_steps_info = len(set(steps_info))
        if len_after < len_before:
            print("Duplicates in conformation number detected!")
            if no_of_stereo_format == 0:
                idx_to_obj = {}  # Reset the idx_to_obj dictionary  if duplicate in conformation number is found
                print("Using step info to sort ...")
                unique_steps = OrderedDict.fromkeys(steps_info)
                conformation_format = False
                sub_section = True
                for _key in unique_steps:
                    unique_steps[_key] = {}
                for i, item in enumerate(all_indices):
                    unique_steps[steps_info[i]][item] = orca_objects[i]
            else:
                idx_to_obj = {}  # Reset the idx_to_obj dictionary  if duplicate in conformation number is found
                conformation_format = False
                sub_section = True
                print("Sorting by stereochemistry label, then step info, then conformation number ...")
                stereo_labels = OrderedDict.fromkeys(stereo_info)

                for label in stereo_labels:
                    stereo_labels[label] = OrderedDict.fromkeys(steps_info)
                    for _key in stereo_labels[label]:
                        stereo_labels[label][_key] = {}
                for i, item in enumerate(all_indices):
                    stereo_labels[stereo_info[i]][steps_info[i]][item] = orca_objects[i]

    if conformation_format or all_names_are_int or no_of_func_format > 0 or sub_section:
        if conformation_format:
            print("Names are in conformation format, sorting ... ", end="")
        elif all_names_are_int:
            print("All names of structure are integer, sorting ... ", end="")
        elif no_of_func_format > 0:
            assert no_of_func_format == len(orca_objects), f"FATAL ERROR: Inconsistent naming! " \
                                                           f"no_of_func_format={no_of_func_format} " \
                                                           f"vs. no. of Orca objects={len(orca_objects)}"
            print("All names are in function format, sorting ...", end="")

        if sub_section:
            sorted_orca_obj = []
            if no_of_stereo_format == 0:
                print("Using step info ...", end="")
                temp = []
                for key in unique_steps:
                    sorted_dict = {k: v for k, v in sorted(unique_steps[key].items(), key=lambda _key: _key[0])}
                    temp.append([sorted_dict[k] for k in sorted_dict])

                for item in temp:
                    sorted_orca_obj += item

                orca_objects = sorted_orca_obj.copy()
            elif no_of_stereo_format > 0:
                print("Using step info with stereochemistry information ...", end="")
                temp = []

                def ts_name_to_int(_item):
                    try:
                        value = int(_item[0])
                    except ValueError:
                        try:
                            t = _item[0].split("-")
                            value = int(t[0]) * 10 + int(t[1])
                        except ValueError:
                            raise ValueError(f"{t}")
                    return value

                for k1 in stereo_labels:
                    stereo_labels[k1] = {k: v for k, v in sorted(stereo_labels[k1].items(), key=ts_name_to_int)}
                for k1 in stereo_labels:
                    for k2 in stereo_labels[k1]:
                        sorted_dict = {k: v for k, v in sorted(stereo_labels[k1][k2].items(),
                                                               key=lambda _key: _key[0])}
                        temp.append([sorted_dict[k] for k in sorted_dict])

                for item in temp:
                    sorted_orca_obj += item

        else:
            idx_to_obj = {k: v for k, v in sorted(idx_to_obj.items(), key=lambda _key: _key[0])}
            sorted_orca_obj = [idx_to_obj[key] for key in idx_to_obj]

        orca_objects = sorted_orca_obj
        print("done")

    elif no_of_stereo_format > 0:
        assert no_of_stereo_format == len(orca_objects), "FATAL ERROR: Inconsistent naming!"
        temp = []
        for key in stereo_labels:
            sorted_dict = {k: v for k, v in sorted(stereo_labels[key].items(), key=lambda key: key[0])}
            temp.append([sorted_dict[k] for k in sorted_dict])

        sorted_orca_obj = []
        for item in temp:
            sorted_orca_obj += item

        orca_objects = sorted_orca_obj.copy()

    labels_after_sorting = [x.name_of_struct[0] for x in orca_objects]
    n_orca_after_sorting = len(orca_objects)

    if n_orca_after_sorting != n_orcas_before_sorting:
        print(f"WARNING! {n_orcas_before_sorting - n_orca_after_sorting} the orca objects disappeared after sorting!")
        labels_after_sorting = set(labels_after_sorting)
        labels_before_sorting = set(labels_before_sorting)
        difference = labels_before_sorting - labels_after_sorting
        print(f"These are the missing ones - {difference}")

    warnings = list(set(warnings))
    if len(warnings) != 0:
        print("----------------WARNING----------------")
        for warning in warnings:
            print(warning)

    # Check for common temperature in Orca4 objects
    common_temp = 0.0
    all_temp = {}
    if not no_opt and not no_freq:
        for idx, orca4 in enumerate(orca_objects):
            if len(orca4.extra_thermo) > 0:
                for _temp in orca4.extra_thermo:
                    all_temp[orca4.name_of_struct[0]] = _temp
            else:
                try:
                    all_temp[orca4.name_of_struct[0]] = orca4.geo_opt[0].thermochemistry.temp
                except AttributeError:
                    raise AttributeError(f"No thermochemistry object for  {orca4.name_of_struct[0]}")

    unique_temperatures = [value for key, value in all_temp.items()]
    unique_temperatures = (list(set(unique_temperatures)))
    has_common_temp = False

    if len(unique_temperatures) == 1:
        has_common_temp = True
        common_temp = unique_temperatures[0]

    # Check if all the orca objects has NMR data
    all_has_nmr = True
    for _item in orca_objects:
        if not _item.has_nmr:
            all_has_nmr = False
            break

    if write_excel:
        if has_common_temp:
            print("----------------Converting orca data to spreadsheet----------------")
            if custom_excel_filename is None:
                custom_excel_filename = excel_filename
                orca_to_excel(orca_objects, excel_filename=excel_filename, temperature=common_temp,
                              write_nmr=all_has_nmr)
            else:
                orca_to_excel(orca_objects, excel_filename=custom_excel_filename, temperature=common_temp,
                              write_nmr=all_has_nmr)
        else:
            if len(all_temp) == 0:
                print("No temperature is present. Only electronic energy will be written")
                _names = [orca4.name_of_struct[0] for orca4 in orca_objects]
                _energies = [orca4.extra_single_points[0].hamiltonian.elec_energy for orca4 in orca_objects]
                to_excel([_names, _energies], excel_filename=excel_filename)

            else:
                print("No common temperature amongst Orca4 objects! Please do something!")
                for key in all_temp:
                    print(f"{key} - {all_temp[key]}K")

    return orca_objects, custom_excel_filename
