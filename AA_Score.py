import argparse
from collections import defaultdict
import cProfile
import os
import sys
import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from interaction_components.plinteraction import get_interactions
from utils.hbonds import calc_hbond_strength
from utils.hydrophobic import calc_hydrophobic
from utils.vdw import calc_vdw


residue_names = [
    "HIS",
    "ASP",
    "ARG",
    "PHE",
    "ALA",
    "CYS",
    "GLY",
    "GLN",
    "GLU",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "SER",
    "TYR",
    "THR",
    "ILE",
    "TRP",
    "PRO",
    "VAL"]


def energy_hbond(dist):
    return -(1 / (1 + np.power(dist / 2.6, 6))) / 0.58


### Obtained from WolframAlpha
def force_hbond(dist):
    return 0.0335 * dist**5 / (0.00324 * dist**6 + 1)**2
#   return -(1 / (1 + np.power(dist / 2.6, 6))) / 0.58


def calc_hbonds_descriptor(protein, interactions):
    # Indices (i,j) should be stored in interactions (PLInteraction) object
    idx = interactions.hbond_idx
    if not len(idx):
        return np.nan

    # Get the interatomic distances (dist_mat) from interactions object
    dist = interactions.dist_mat[idx[:,0], idx[:,1]]
    return hbond_energy(dist)


def calc_hydrophybic_descriptor(protein, interactions):
    # Indices (i,j) should be stored in interactions (PLInteraction) object
    idx = interactions.hphob_idx
    if not len(idx):
        return np.nan
    # Atomic radii (np.ndarray) should be stored in protein object
    atomic_radii = protein.atomic_radii[idx]

    # Sum the radii (r[i] + r[j])
    radii_sum = atomic_radii.sum(axis=1)

    # Get the interatomic distances (dist_mat) from interactions object
    dist = interactions.dist_mat[idx[:,0], idx[:,1]]

    hphob_energy = -0.66666 * (radii_sum + 2.0 - dist)
    hphob_energy = np.clip(hphob_energy, -1, 0)
    return hphob_energy


def energy_vdw(dist, radii):
    return np.power((radii / dist), 8) - 2 * np.power((radii / dist), 4)
    

def force_vdw(dist, radii):
    radii4 = radii ** 4
    return 8 * radii4 * (dist**5 - radii4) / dist**9
    

def calc_vdw_descriptor(protein, interactions):
    # Indices (i,j) should be stored in interactions (PLInteraction) object
    idx = interactions.vdw_idx

    # Get atomic radii sum
    atomic_radii = protein.atom_radii[idx]
    radii_sum = atomic_radii.sum(axis=1)

    # Get the interatomic distances (dist_mat) from interactions object
    dist = interactions.dist_mat[idx[:,0], idx[:,1]]

    return energy_vdw(dist, radii_sum)


def energy_ele(dist, charge_product):
    return - charge_product / dist
#   return - charge_product / dist * np.exp(-dist)


def force_ele(dist, charge_product):
    return charge_product / dist**2


def calc_ele_descriptor(protein, interactions):
    # Indices (i,j) should be stored in interactions (PLInteraction) object
    idx = interactions.vdw_idx

    # Partial charges (np.ndarray) should be stored in protein object
    partial_charge = protein.atom_charge[idx]

    # Multiply the charges (q[i] * q[j])
    charge_product = np.product(partial_charge, axis=1)

    # Get the interatomic distances (dist_mat) from interactions object
    dist = interactions.dist_mat[idx[:,0], idx[:,1]]

    ele_energy = charge_product / dist
    return ele_energy


def calc_score(mol_prot):
    tm = time.time()
    result = get_interactions(mol_prot)
    print((time.time() - tm))
    interactions = result.interactions
    print((time.time() - tm))

    hbond_energy = calc_hbonds_descriptor(result.prot, interactions)
    print((time.time() - tm))
#   hphob_energy = calc_hydrophybic_descriptor(result.prot, interactions)
#   print((time.time() - tm))
    vdw_energy = calc_vdw_descriptor(result.prot, interactions)
    print((time.time() - tm))
    ele_energy = calc_ele_descriptor(result.prot, interactions)
    print((time.time() - tm))

#   metal_ligand = calc_metal_descriptor(interactions)
#   print((time.time() - tm))
#   tpp_energy, ppp_energy = calc_pistacking_descriptor(interactions)
#   print((time.time() - tm))
#   ppc_energy, pic_dict = calc_pication_descriptor(interactions)
#   print((time.time() - tm))



def run_test(output_file=None):
    #start_time = time.time()

    protein_file = "data/2reg/2reg_H.pdb"
    mol_prot = Chem.MolFromPDBFile(protein_file, removeHs=False)
    _ = calc_score(mol_prot)



def get_residue_interactions(protein, inter):
    residue_pairs = defaultdict(list)
    idx_list = [inter.hbond_idx, inter.vdw_idx, inter.saltbridge_idx]
    bond_type = ['hbond', 'vdw', 'saltbridge']
    for idx, bt in zip(idx_list, bond_type):
        for i, j in idx:
            i, j = min(i,j), max(i,j)
            dist = inter.dist_mat[i,j]
            atoms = protein.atom_name[[i,j]]
            vec = np.diff(protein.atom_coord[[i,j]], axis=0)[0] / dist
            if bt == 'hbond':
                force = force_hbond(dist)
            elif bt == 'vdw':
                radii_sum = protein.atom_radii[i] + protein.atom_radii[j]
                force = force_vdw(dist, radii_sum)
            elif bt == 'saltbridge':
                charge_product = protein.atom_charge[i] * protein.atom_charge[j]
                force = force_ele(dist, charge_product)
            i0, j0 = protein.residue_idx[[i,j]]
            residue_pairs[(i0,j0)].append((vec, force, bt, '_'.join(atoms)))

    return residue_pairs



if __name__ == "__main__":
#   run_test(sys.argv[1])
#   cProfile.run("run_test()")
    run_test()


