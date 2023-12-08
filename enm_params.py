import numpy as np


def get_ca_coord(prot, i):
    return prot.atom_coord[prot.is_ca & (prot.residue_idx == i)][0]


def get_ca_dist(prot, i, j):
    return np.linalg.norm(get_ca_vec(prot, i, j))


def get_ca_vec(prot, i, j, norm=False):
    rij = get_ca_coord(prot, i) - get_ca_coord(prot, j)
    if norm:
        return rij / np.linalg.norm(rij)
    return rij


def get_force_vec(rpairs, i, j, norm=False):
    vec = np.mean([x[1] * x[0] for x in rpairs[(i, j)]], axis=0)
    if norm:
        return vec / np.linalg.norm(vec)
    return vec


def get_force(rpairs, i, j, norm=False):
    return np.linalg.norm(np.mean([x[1] * x[0] for x in rpairs[(i, j)]], axis=0))



def get_enm_params(prot, rpairs):
    idx_key = {j:i for i, j in enumerate(np.unique(prot.residue_idx))}
    N = len(idx_key)
    force_mat = np.zeros((N, N), float)
    for i, j in rpairs.keys():
        rij = get_ca_vec(prot, i, j, True)
        force_vec = get_force_vec(rpairs, i, j, True)
        force_mag = get_force(rpairs, i, j, True)

        i0, j0 = idx_key[i], idx_key[j]
        k = np.dot(rij, force_vec) * force_mag
        force_mat[i0,j0] = k
        force_mat[j0,i0] = k
    return force_mat



