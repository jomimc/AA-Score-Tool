from interaction_components.plinteraction import get_interactions
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from utils.hbonds import calc_hbond_strength
from utils.hydrophobic import calc_hydrophobic
from utils.vdw import calc_vdw
import os
import sys
import argparse


# import time

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


def is_sidechain(atom):
    res = atom.GetPDBResidueInfo()
    atom_name = res.GetName().strip(" ")
    if atom_name in ("C", "CA", "N", "O", "H"):
        return False
    else:
        return True


def create_dict():
    interaction_dict = {}
    for name in residue_names:
        interaction_dict.update({name + "_side": 0})
        interaction_dict.update({name + "_main": 0})
    return interaction_dict


def calc_hbonds_descriptor(protein, interactions):
    # Indices (i,j) should be stored in interactions (PLInteraction) object
    idx = interactions.hbond_idx
    if not len(idx):
        return np.nan
    # Atomic radii (np.ndarray) should be stored in protein object
    atomic_radii = protein.atomic_radii[idx]
    # Sum the radii (r[i] + r[j])
    #radii_sum = atomic_radii.sum(axis=1)
    # Get the interatomic distances (dist_mat) from interactions object
    dist = interactions.dist_mat[idx[:,0], idx[:,1]]

    hb_energy = -(1 / (1 + np.power(dist / 2.6, 6))) / 0.58
    return hb_energy


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


def calc_vdw_descriptor(protein, interactions):
    # Indices (i,j) should be stored in interactions (PLInteraction) object
    idx = interactions.vdw_idx
    # Atomic radii (np.ndarray) should be stored in protein object
    atomic_radii = protein.atomic_radii[idx]
    # Sum the radii (r[i] + r[j])
    radii_sum = atomic_radii.sum(axis=1)
    # Get the interatomic distances (dist_mat) from interactions object
    dist = interactions.dist_mat[idx[:,0], idx[:,1]]

    vdw_energy = np.sum(np.power((radii_sum / dist), 8) - 2 * np.power((radii_sum / dist), 4))
    return vdw_energy


def calc_ele_descriptor(protein, interactions):
    # Indices (i,j) should be stored in interactions (PLInteraction) object
    idx = interactions.vdw_idx
    # Partial charges (np.ndarray) should be stored in protein object
    partial_charge = protein.partial_charges[idx]
    # Multiply the charges (q[i] * q[j])
    charge_product = np.product(partial_charge, axis=1)
    # Get the interatomic distances (dist_mat) from interactions object
    dist = interactions.dist_mat[idx[:,0], idx[:,1]]

    ele_energy = charge_product / dist
    return ele_energy


def calc_metal_complexes(metal):
    dist = metal.distance
    if dist < 2.0:
        return -1.0
    elif 2.0 <= dist < 3.0:
        return -3.0 + dist
    else:
        return 0.0


def calc_metal_descriptor(interactions):
    ml_energy = 0
    for ml in interactions.metal_complexes:
        if ml.target.location != "ligand":
            continue
        energy = calc_metal_complexes(ml)
        ml_energy += energy
    return ml_energy


def calc_pistacking_descriptor(interactions):
    T_pistacking_energy, P_pistacking_energy = 0, 0
    for pis in interactions.pistacking:
        if pis.type == "T":
            T_pistacking_energy += -1
        else:
            P_pistacking_energy += -1
    return T_pistacking_energy, P_pistacking_energy


def calc_pication_laro(interactions):
    pic_dict = create_dict()
    for pic in interactions.pication_laro:
        restype = pic.restype
        sidechain = is_sidechain(pic.charge.atoms[0])
        energy = -1
        if restype[:2] == "HI" and restype not in residue_names:
            restype = "HIS"
        if restype == "ACE":
            continue
        if sidechain:
            key = restype + "_side"
        else:
            key = restype + "_main"
        pic_dict[key] += energy
    return pic_dict


def calc_pication_descriptor(interactions):
    paro_pication_energy, laro_pication_energy = 0, 0
    for pic in interactions.pication_paro:
        paro_pication_energy += -1
    pic_dict = calc_pication_laro(interactions)
    return paro_pication_energy, pic_dict


### THIS IS THE MAIN FUNCTION TO RUN
def calc_score(mol_prot):
    result = get_interactions(mol_prot)
    interactions = result.interactions

    hbond_energy = calc_hbonds_descriptor(result.prot, interactions)
    hphob_energy = calc_hydrophybic_descriptor(result.prot, interactions)
    vdw_energy = calc_vdw_descriptor(result.prot, interactions)
    ele_energy = calc_ele_descriptor(result.prot, interactions)

    metal_ligand = calc_metal_descriptor(interactions)
    tpp_energy, ppp_energy = calc_pistacking_descriptor(interactions)
    ppc_energy, pic_dict = calc_pication_descriptor(interactions)



def run_test(protein_file, output_file=None):
    #start_time = time.time()


    mol_prot = Chem.MolFromPDBFile(protein_file, removeHs=False)
    _ = calc_score(mol_prot)


    # end_time = time.time()
    # time_spend = end_time - start_time
    # print(f"time spend: {time_spend} seconds")
    # if output_file:
    #     with open(output_file, "a") as f:
    #         f.write(name + "\t" + str(score) + "\n")
    # else:
    #     return name, score


if __name__ == "__main__":
    run_test(sys.argv[1])


