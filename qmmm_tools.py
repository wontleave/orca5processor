import numpy as np
import pandas as pd
import networkx as nx
from ase import Atoms
from ase.io import read, write
from ase.geometry.analysis import Analysis
from orca5_utils import reorder_atoms, calc_rmsd_ase_multi_atoms, Qmmm
from reading_utilis import read_root_folders
from os import listdir, path
from networkx import from_numpy_matrix
from pathlib import Path
from itertools import combinations


def standardize_for_qmmm(root_folder, multilayers_from_atom_num=None,
                         qm_charge=0, qm_multiplicity=1,
                         total_charge=0, total_multiplicity=1,
                         qmmm_new_folder_name="qmmm"):
    """

    :param root_folder:
    :type root_folder:
    :param multilayers_from_atom_num: specific the number of atoms for each layers in a multiscale calculation
    :type multilayers_from_atom_num:
    :param qm_charge:
    :type qm_charge:
    :param qm_multiplicity:
    :type qm_multiplicity:
    :param total_charge:
    :type total_charge:
    :param total_multiplicity:
    :type total_multiplicity:
    :param qmmm_new_folder_name:
    :type qmmm_new_folder_name:
    :return:
    :rtype:
    """
    low_level_allowed_key = ("XTB", "XTB1", "HF-3c", "PBEH-3C", "R2SCAN-3C", "PM3", "AM1", "QM2")
    # Sanity Check

    atom_num_check = []
    if multilayers_from_atom_num is not None:
        # Check if the format is correct
        key_count = 0
        for key in multilayers_from_atom_num:
            if key in low_level_allowed_key or key == "QM":
                key_count += 1
                atom_num_check.append(multilayers_from_atom_num[key])

        assert 2 <= key_count <= 3, f"{key_count} of keys detected, it is an invalid QM/MM combination"
        atom_num_check = set(atom_num_check)
        assert len(atom_num_check) == key_count, f"You have duplicates number of atoms: {multilayers_from_atom_num}"

    req_folders = []
    read_root_folders(root_folder, req_folders)

    folder_to_xyz = {}
    ref_key = None

    for idx, folder_ in enumerate(req_folders):
        files = listdir(folder_)
        if idx == 0:
            ref_key = folder_
        for file in files:
            xyz_files = [path.join(folder_, file) for file in files if "xyz" in file]

        if len(xyz_files) > 0:
            folder_to_xyz[folder_] = {"xyz": xyz_files}

    # Check the number of xyz files detected
    for key in folder_to_xyz:
        print(f"Working on {key} ...", end="")
        n_xyz = len(folder_to_xyz[key]["xyz"])
        folder_to_xyz[key]["n_xyz"]= n_xyz

        if n_xyz > 1:
            # Compare the structures
            ase_atoms = [read(item) for item in folder_to_xyz[key]["xyz"]]
            rmsds = calc_rmsd_ase_multi_atoms(ase_atoms)
            assert np.all(rmsds<1.0e-6), f"RMSD of some structures are more than 1e-6, please specify which one to use"
            folder_to_xyz[key]["ref_struct"] = folder_to_xyz[key]["xyz"][0]

        ase_atoms = read(folder_to_xyz[key]["ref_struct"])
        elements = ase_atoms.get_chemical_symbols()
        coords = ase_atoms.get_positions()
        p_atoms_analysis = Analysis(ase_atoms)
        adj_matrix = p_atoms_analysis.adjacency_matrix[0].toarray()
        p_graph = from_numpy_matrix(adj_matrix)

        for idx, element in enumerate(elements):
            p_graph.nodes[idx]["element"] = element
            p_graph.nodes[idx]["positions"] = coords[idx]

        all_nodes = list(p_graph.nodes)
        sub_structures = []
        counter = 0
        while len(all_nodes) > 0:
            curr_node = all_nodes[0]
            sub_structures.append(list(nx.descendants(p_graph, curr_node)) + [curr_node])
            all_nodes = np.setdiff1d(all_nodes, sub_structures[counter])
            counter += 1

        qm_spec = ""

        # Create the matrix for tabulating two layers sum of atoms
        n_substructures = len(sub_structures)

        possible_combi = []
        combi_n_atoms = []
        for i in range(1, n_substructures):
            possible_combi += list(combinations(range(n_substructures), i))

        for combi in possible_combi:
            temp_sum = 0
            for combi_idx in combi:
                temp_sum += len(sub_structures[combi_idx])
            combi_n_atoms.append(temp_sum)

        sum_atoms_df = pd.DataFrame(combi_n_atoms, index=possible_combi, columns=["sum"])
        filter = sum_atoms_df["sum"] == multilayers_from_atom_num["QM"]
        qm_idx = sum_atoms_df.index[sum_atoms_df["sum"] == multilayers_from_atom_num["QM"]].tolist()

        assert len(qm_idx) == 1, f"ERROR {len(qm_idx)} candidates found ... cannot continue!"

        # Renumber the fragments such that the numbering is consecutive
        reordered_atoms_positions = []
        reordered_atoms_elements = ""
        # QM layer first
        last_qm_atom = 0
        for sub_struct_idx, sub_struct in enumerate(sub_structures):
            if sub_struct_idx in qm_idx[0]:
                for old_idx in sub_structures[sub_struct_idx]:
                    reordered_atoms_positions.append(coords[old_idx])
                    reordered_atoms_elements += elements[old_idx]
                    last_qm_atom += 1

        # The low level layer
        for sub_struct_idx, sub_struct in enumerate(sub_structures):
            if sub_struct_idx not in qm_idx[0]:
                for old_idx in sub_structures[sub_struct_idx]:
                    reordered_atoms_positions.append(coords[old_idx])
                    reordered_atoms_elements += elements[old_idx]

        qm_spec += f"0:{last_qm_atom} "

        reordered_atoms = Atoms(reordered_atoms_elements, positions=reordered_atoms_positions)
        current_basename = path.basename(key)
        qmmm_root_path = path.join(root_path, qmmm_new_folder_name)
        individial_qmmm_path = path.join(qmmm_root_path, current_basename)
        Path(individial_qmmm_path).mkdir(parents=True, exist_ok=True)
        qm_spec = "{" + qm_spec.strip() + "}"

        qmmm_obj = Qmmm(qm_spec=qm_spec, total_charge=total_charge, total_multiplicity=total_multiplicity)
        qmmm_obj.write_qmmm_inp(path.join(individial_qmmm_path, "QMMMpartial.inp"))
        write(path.join(individial_qmmm_path, "start.xyz"), reordered_atoms)
        print("done!")


if __name__ == "__main__":

    root_path = r"E:\TEST\DEBUG_ORCA5\MT-rxn"
    standardize_for_qmmm(root_path, multilayers_from_atom_num={"QM": 56, "XTB": 75})





