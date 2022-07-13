import pandas as pd
import numpy as np
import pyparsing as pp
from ase import Atoms

HA2J = 4.359744E-18
BOHRS2ANG = 0.529177
SPEEDOFLIGHT = 2.99792458E8
AMU2KG = 1.660538782E-27


def make_sym_mat(table_block):
    mat_size = int(table_block[1])
    # Orca prints blocks of 5 columns
    arr = np.array(table_block[2:], dtype=float)
    assert arr.size == mat_size ** 2
    block_size = 5 * mat_size
    cbs = [
        arr[i * block_size : (i + 1) * block_size].reshape(mat_size, -1)
        for i in range(arr.size // block_size + 1)
    ]
    return np.concatenate(cbs, axis=1)


def parse_hess_file(text):
    integer = pp.Word(pp.nums)
    float_ = pp.Word(pp.nums + ".-")
    plus = pp.Literal("+")
    minus = pp.Literal("-")
    E = pp.Literal("E")
    scientific = pp.Combine(float_ + E + pp.Or([plus, minus]) + integer)

    table_header_line = pp.Suppress(integer + pp.restOfLine)
    scientific_line = pp.Suppress(integer) + pp.OneOrMore(scientific)
    scientific_block = table_header_line + pp.OneOrMore(scientific_line)
    float_line = pp.Suppress(integer) + float_
    comment_line = pp.Literal("#") + pp.restOfLine
    mass_xyz_line = pp.Group(
        pp.Word(pp.alphas) + float_ + pp.Group(pp.OneOrMore(float_))
    )

    block_name = pp.Word(pp.alphas + "$_")
    block_length = integer

    block_int = block_name + block_length
    block_float = block_name + float_
    block_table = block_name + integer + pp.OneOrMore(scientific_block)
    block_table_two_int = (
        block_name + integer + pp.Suppress(integer) + pp.OneOrMore(scientific_block)
    )
    block_float_table = block_name + integer + pp.OneOrMore(float_line)
    block_atoms = block_name + integer + pp.OneOrMore(mass_xyz_line)

    act_atom = block_int.setResultsName("act_atom")
    act_coord = block_int.setResultsName("act_coord")
    act_energy = block_float.setResultsName("act_energy")
    hessian = block_table.setResultsName("hessian")
    vib_freqs = block_float_table.setResultsName("vib_freqs")
    normal_modes = block_table_two_int.setResultsName("normal_modes")
    atoms = block_atoms.setResultsName("atoms")

    parser = (
        block_name
        + act_atom
        + act_coord
        + act_energy
        + hessian
        + vib_freqs
        + normal_modes
        + pp.OneOrMore(comment_line)
        + atoms
    )
    parsed = parser.parseString(text)
    return parsed


def moi_tensor(massvec, expmassvec, xyz):
    # Center of Mass
    com = np.sum(expmassvec.reshape(-1, 3)
                 * xyz.reshape(-1, 3), axis=0) / np.sum(massvec)

    # xyz shifted to COM
    xyz_com = xyz.reshape(-1, 3) - com

    # Compute elements need to calculate MOI tensor
    mass_xyz_com_sq_sum = np.sum(
        expmassvec.reshape(-1, 3) * xyz_com ** 2, axis=0)

    mass_xy = np.sum(massvec * xyz_com[:, 0] * xyz_com[:, 1], axis=0)
    mass_yz = np.sum(massvec * xyz_com[:, 1] * xyz_com[:, 2], axis=0)
    mass_xz = np.sum(massvec * xyz_com[:, 0] * xyz_com[:, 2], axis=0)

    # MOI tensor
    moi = np.array([[mass_xyz_com_sq_sum[1] + mass_xyz_com_sq_sum[2], -1 * mass_xy, -1 * mass_xz],
                    [-1 * mass_xy, mass_xyz_com_sq_sum[0] +
                        mass_xyz_com_sq_sum[2], -1 * mass_yz],
                    [-1 * mass_xz, -1 * mass_yz, mass_xyz_com_sq_sum[0] + mass_xyz_com_sq_sum[1]]])

    # MOI eigenvectors and eigenvalues
    moi_eigval, moi_eigvec = np.linalg.eig(moi)

    return xyz_com, moi_eigvec

def trans_rot_vec(massvec, xyz_com, moi_eigvec):

    # Mass-weighted translational vectors
    zero_vec = np.zeros([len(massvec)])
    sqrtmassvec = np.sqrt(massvec)
    expsqrtmassvec = np.repeat(sqrtmassvec, 3)

    d1 = np.transpose(np.stack((sqrtmassvec, zero_vec, zero_vec))).reshape(-1)
    d2 = np.transpose(np.stack((zero_vec, sqrtmassvec, zero_vec))).reshape(-1)
    d3 = np.transpose(np.stack((zero_vec, zero_vec, sqrtmassvec))).reshape(-1)

    # Mass-weighted rotational vectors
    big_p = np.matmul(xyz_com, moi_eigvec)

    d4 = (np.repeat(big_p[:, 1], 3).reshape(-1)
          * np.tile(moi_eigvec[:, 2], len(massvec)).reshape(-1)
          - np.repeat(big_p[:, 2], 3).reshape(-1)
          * np.tile(moi_eigvec[:, 1], len(massvec)).reshape(-1)) * expsqrtmassvec

    d5 = (np.repeat(big_p[:, 2], 3).reshape(-1)
          * np.tile(moi_eigvec[:, 0], len(massvec)).reshape(-1)
          - np.repeat(big_p[:, 0], 3).reshape(-1)
          * np.tile(moi_eigvec[:, 2], len(massvec)).reshape(-1)) * expsqrtmassvec

    d6 = (np.repeat(big_p[:, 0], 3).reshape(-1)
          * np.tile(moi_eigvec[:, 1], len(massvec)).reshape(-1)
          - np.repeat(big_p[:, 1], 3).reshape(-1)
          * np.tile(moi_eigvec[:, 0], len(massvec)).reshape(-1)) * expsqrtmassvec

    d1_norm = d1 / np.linalg.norm(d1)
    d2_norm = d2 / np.linalg.norm(d2)
    d3_norm = d3 / np.linalg.norm(d3)
    d4_norm = d4 / np.linalg.norm(d4)
    d5_norm = d5 / np.linalg.norm(d5)
    d6_norm = d6 / np.linalg.norm(d6)

    dx_norms = np.stack((d1_norm,
                         d2_norm,
                         d3_norm,
                         d4_norm,
                         d5_norm,
                         d6_norm))

    return dx_norms


if __name__ == "__main__":
    root_ = r"E:\TEST\Orca5Processor_tests\hessian_md\methylBr_F\run.AnHess.r2scan-3c.hess"

    with open(root_) as f:
        hessian_txt = f.read()
    parsed = parse_hess_file(hessian_txt)
    cart_hessian = make_sym_mat(parsed["hessian"])
    atoms, massvec, coords3d = zip(*parsed["atoms"][2:])
    coords3d = np.array([c.asList() for c in coords3d], dtype=float)
    mol = Atoms(atoms, positions=coords3d)
    massvec = np.array(massvec, dtype=float) * AMU2KG
    expmassvec = np.repeat(massvec, 3)
    sqrtinvmassvec = np.divide(1.0, np.sqrt(expmassvec))
    hessian_mwc = np.einsum('i,ij,j->ij', sqrtinvmassvec,
                            cart_hessian, sqrtinvmassvec)
    hessian_eigval, hessian_eigvec = np.linalg.eig(hessian_mwc)

    xyz_com, moi_eigvec = moi_tensor(massvec, expmassvec, coords3d)
    dx_norms = trans_rot_vec(massvec, xyz_com, moi_eigvec)

    P = np.identity(3 * len(massvec))
    for dx_norm in dx_norms:
        P -= np.outer(dx_norm, dx_norm)

    # Projecting the T and R modes out of the hessian
    mwhess_proj = np.dot(P.T, hessian_mwc).dot(P)

    hessian_eigval, hessian_eigvec = np.linalg.eigh(mwhess_proj)

    hessian_eigval_abs = np.abs(hessian_eigval)
    pre_vib_freq_cm_1 = np.sqrt(
        hessian_eigval_abs * HA2J * 10e19) / (SPEEDOFLIGHT * 2 * np.pi * BOHRS2ANG * 100)
    print()
    # for folder in req_folders:
