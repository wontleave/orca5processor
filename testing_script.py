import pandas as pd
import numpy as np
import pyparsing as pp
from ase import Atoms, units
from ase.units import Bohr, Rydberg, kJ, kB, fs, Hartree, mol, kcal, second
from scipy.stats import rv_discrete

CM_2_AU = 4.5564e-6
ANGS_2_AU = 1.8897259886
AMU_2_AU = 1822.88985136
k_B = 1.38064852e-23
PLANCKS_CONS = 6.62607015e-34
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


class Boltzmann_gen(rv_discrete):
    "Boltzmann distribution"
    def _pmf(self, k, nu, temperature):
         return ((np.exp(-(k * PLANCKS_CONS * nu)/(k_B * temperature))) *
                (1 - np.exp(-(PLANCKS_CONS * nu)/(k_B * temperature))))


def reactive_normal_mode_sampling(xyz, force_constants_J_m_2,
                                  proj_vib_freq_cm_1, proj_hessian_eigvec,
                                  temperature,
                                  kick=1):
    """Normal Mode Sampling for Transition States. Takes in xyz(1,N,3), force_constants(3N-6) in J/m^2,
    projected vibrational frequencies(3N-6) in cm^-1,mass-weighted projected hessian eigenvectors(3N-6,3N)
    ,temperature in K, and scaling factor of initial velocity of the lowest imaginary mode.
    Returns displaces xyz and a pair of velocities(forward and backwards)"""

    # Determine the highest level occupany of each mode
    occ_vib_modes = []
    boltzmann = Boltzmann_gen(a=0, b=1000000, name="boltzmann")
    for i, nu in enumerate(proj_vib_freq_cm_1):
        if nu > 50:
            occ_vib_modes.append(boltzmann.rvs(nu * SPEEDOFLIGHT * 100,
                                               temperature))
        elif i == 0:
            occ_vib_modes.append(boltzmann.rvs(-1 * nu * SPEEDOFLIGHT * 100,
                                               temperature))
        else:
            occ_vib_modes.append(-1)

    # Determine maximum displacement (amplitude) of each mode

    amplitudes = []
    freqs = []
    for i, occ in enumerate(occ_vib_modes):
        if occ >= 0:
            energy = proj_vib_freq_cm_1[i] * SPEEDOFLIGHT * 100 * PLANCKS_CONS  # cm-1 to Joules
            amplitudes.append(np.sqrt((0.5 * (occ + 1) * energy) / force_constants_J_m_2[i]) * 1e9)  # Angstom
        else:
            amplitudes.append(0)

    # Determine the actual displacements and velocities
    displacements = []
    velocities = []
    random_0_1 = [np.random.normal(0, 1) for i in range(len(amplitudes))]
    for i, amplitude in enumerate(amplitudes):

        if force_constants_J_m_2[i] > 0:

            displacements.append(amplitude
                                 * np.cos(2 * np.pi * random_0_1[i])
                                 * proj_hessian_eigvec[i])

            velocities.append(-1 * proj_vib_freq_cm_1[i] * SPEEDOFLIGHT * 100 * 2 * np.pi
                              * amplitude
                              * np.sin(2 * np.pi * random_0_1[i])
                              * proj_hessian_eigvec[i] / Bohr ** 2)

        elif i == 0:

            displacements.append(0)
            velocities.append(0)

            # Extra kick for lowest imagninary mode(s)
            velocities.append(-1 * proj_vib_freq_cm_1[i] * SPEEDOFLIGHT * 100 * 2 * np.pi
                              * amplitude
                              * np.sin(2 * np.pi * random_0_1[i])
                              * proj_hessian_eigvec[i] / Bohr ** 2)


        else:

            displacements.append(0)
            velocities.append(0)

    tot_disp = np.sum(np.array(displacements), axis=0)
    # In angstroms
    disp_xyz = xyz + tot_disp.reshape(1, -1, 3)
    # In angstroms per second
    tot_vel_plus = np.sum(np.array(velocities), axis=0).reshape(1, -1, 3)
    tot_vel_minus = -1 * tot_vel_plus

    return disp_xyz, tot_vel_plus, tot_vel_minus


def vib_analysis(massvec_, cart_hessian, xyz_coords):

    massvec = np.array(massvec_, dtype=float) * units._amu
    expmassvec = np.repeat(massvec, 3)
    sqrtinvmassvec = np.divide(1.0, np.sqrt(expmassvec))
    hessian_mwc = np.einsum('i,ij,j->ij', sqrtinvmassvec,
                            cart_hessian, sqrtinvmassvec)

    xyz_com, moi_eigvec = moi_tensor(massvec, expmassvec, xyz_coords)
    dx_norms = trans_rot_vec(massvec, xyz_com, moi_eigvec)

    P = np.identity(3 * len(massvec))
    for dx_norm in dx_norms:
        P -= np.outer(dx_norm, dx_norm)

    # Projecting the T and R modes out of the hessian
    mwhess_proj = np.dot(P.T, hessian_mwc).dot(P)

    hessian_eigval, hessian_eigvec = np.linalg.eig(mwhess_proj)
    neg_ele = []

    for i, eigval in enumerate(hessian_eigval):
        if eigval < 0:
            neg_ele.append(i)

    hessian_eigval_abs = np.abs(hessian_eigval)
    pre_vib_freq_cm_1 = np.sqrt(
        hessian_eigval_abs * HA2J * 10e19) / (SPEEDOFLIGHT * 2 * np.pi * BOHRS2ANG * 100)

    vib_freq_cm_1 = pre_vib_freq_cm_1.copy()

    for i in neg_ele:
        vib_freq_cm_1[i] = -1 * pre_vib_freq_cm_1[i]

    trans_rot_elms = []
    for i, freq in enumerate(vib_freq_cm_1):
        # Modes that are less than 1.0cm-1 is not a normal mode
        if np.abs(freq) < 1.0:
            trans_rot_elms.append(i)

    force_constants_J_m_2 = np.delete(
        hessian_eigval * HA2J * 1e20 / (BOHRS2ANG ** 2) * units._amu, trans_rot_elms)

    proj_vib_freq_cm_1 = np.delete(vib_freq_cm_1, trans_rot_elms)
    proj_hessian_eigvec = np.delete(hessian_eigvec.T, trans_rot_elms, 0)

    print("PYTHON ---")
    print(proj_vib_freq_cm_1)

    print("ORCA ---")
    print(parsed["vib_freqs"])
    # for folder in req_folders:
    print(force_constants_J_m_2.astype(float))

    return force_constants_J_m_2, proj_vib_freq_cm_1, proj_hessian_eigvec


if __name__ == "__main__":
    root_ = r"E:\TEST\Orca5Processor_tests\hessian_md\methylBr_F\run.AnHess.r2scan-3c.hess"

    with open(root_) as f:
        hessian_txt = f.read()
    parsed = parse_hess_file(hessian_txt)
    cartesian_hessian = make_sym_mat(parsed["hessian"])
    atoms, mass, coords3d = zip(*parsed["atoms"][2:])
    coords3d = np.array([c.asList() for c in coords3d], dtype=float)
    
    mol = Atoms(atoms, positions=coords3d)
    fc_J_m_2, proj_freq_cm_1, proj_eigvec = vib_analysis(mass, cartesian_hessian, coords3d)

    fc_J_m_2 = np.real(fc_J_m_2)
    proj_freq_cm_1 = np.real(proj_freq_cm_1)
    proj_eigvec = np.real(proj_eigvec)

    displaced_xyz, total_velocities_plus, total_velocities_minus = reactive_normal_mode_sampling(coords3d, fc_J_m_2,
                                                                                                 proj_freq_cm_1,
                                                                                                 proj_eigvec,
                                                                                                 temperature=298.15)
    print(displaced_xyz)
    print(total_velocities_plus)
    print(total_velocities_minus)