#!/usr/bin/env python
# coding: utf-8
try:
    from openbabel import pybel
except:
    import pybel

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import Descriptors

from collections import namedtuple
from operator import itemgetter
from interaction_components.utils import *

from interaction_components.utils import centroid, tilde_expansion, tmpfile, classify_by_name, get_atom_coords
from interaction_components.utils import cluster_doubles, is_lig, normalize_vector, vector, ring_is_planar
from interaction_components.utils import extract_pdbid, read_pdb, create_folder_if_not_exists, canonicalize
from interaction_components.utils import read, nucleotide_linkage, sort_members_by_importance, is_acceptor, is_donor
from interaction_components.utils import whichchain, whichatomname, whichrestype, whichresnumber, euclidean3d, int32_to_negative
from interaction_components.detection import halogen, pication, water_bridges, metal_complexation
from interaction_components.detection import filter_contacts, pistacking, hbonds, saltbridge
from interaction_components import config
from scipy.spatial.distance import  cdist
from utils.vdw import calc_vdw

import time

def get_features(mol):
    donors, acceptors, hydrophobics = [], [], []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        atomics = [a.GetAtomicNum() for a in atom.GetNeighbors()]
        if symbol in ["O", "N", "S"]:
            if atomics.count(1) >= 1:
                donors.append(atom.GetIdx())
            elif symbol in ["O", "S"] and atom.GetExplicitValence() <= 2:
                acceptors.append(atom.GetIdx())
            elif symbol == "N" and atomics.count(1) == 0 and atom.GetExplicitValence() <= 3:
                acceptors.append(atom.GetIdx())
        elif (atom.GetAtomicNum() == 6 and set(atomics).issubset({1, 6})):
            hydrophobics.append(atom.GetIdx())

    data = namedtuple("features", "donors acceptors hydrophobics")
    donors = list(set(donors))
    acceptors = list(set(acceptors))
    hydrophobics = list(set(hydrophobics))
    return data(donors=donors, acceptors=acceptors, hydrophobics=hydrophobics)


class Mol:
    def __init__(self, mol):
        self.mol = mol
        self.mol_conf = mol.GetConformers()[0]
        self.rings = None
        self.hydroph_atoms = None
        self.charged = None
        self.hbond_don_atom_pairs = None
        self.hbond_acc_atoms = None

    def find_hba(self):
        raise Exception("have to find hbond acceptors!")

    def find_hbd(self):
        raise Exception("have to find hbond donors!")

    def find_hal(self):
        """Look for halogen bond acceptors (Y-{O|P|N|S}, with Y=C,P,S)"""
        data = namedtuple(
            'hal_acceptor',
            'o o_orig_idx y y_orig_idx o_coords y_coords')
        a_set = []
        # All oxygens, nitrogen, sulfurs with neighboring carbon, phosphor,
        # nitrogen or sulfur
        for a in [
            at for at in self.mol.GetAtoms() if at.GetAtomicNum() in [
                8,
                7,
                16]]:
            n_atoms = [na for na in a.GetNeighbors() if na.GetAtomicNum() in [
                6, 7, 15, 16]]
            if len(n_atoms) == 1:  # Proximal atom
                o_orig_idx = a.GetIdx()
                y_orig_idx = n_atoms[0].GetIdx()
                o_coords = get_atom_coords(a)
                y_coords = get_atom_coords(n_atoms[0])
                a_set.append(data(o=a, o_orig_idx=o_orig_idx, y=n_atoms[0],
                                  y_orig_idx=y_orig_idx, o_coords=o_coords,
                                  y_coords=y_coords))
        return a_set

    def get_hydrophobic_atoms(self):
        return self.hydroph_atoms

    def get_hba(self):
        return self.hbond_acc_atoms

    def get_hbd(self):
        return [
            don_pair for don_pair in self.hbond_don_atom_pairs if don_pair.type == 'regular']

    def get_weak_hbd(self):
        return [
            don_pair for don_pair in self.hbond_don_atom_pairs if don_pair.type == 'weak']



### Need to get:
###     atomic indices
###     atom names
###     residue indices
###     residue names
###     atom coords
###     atomic radii
###     neighbors (for hbond?)
###     partial charges
###     
class Protein:
    def __init__(self, mol):

        self.mol = mol
        self.all_atoms = mol.GetAtoms()

        # Load attributes into numpy arrays
        tm = time.time()
        self.natom = len(self.all_atoms)
        self.vectorize_features()
        print(time.time() - tm)

        self.rings = self.find_rings()
        print(time.time() - tm)
#       self.hydrophobic_idx = self.hydrophobic_atoms()
#       print(time.time() - tm)
        self.hbond_acc_idx = self.find_hba()
        print(time.time() - tm)
        self.hbond_don_idx = self.find_hbd()
        print(time.time() - tm)
#       self.residues = residue_order(mol)
#       print(time.time() - tm)

        self.charged = self.find_charged()
        print(time.time() - tm)
#       self.halogenbond_acc = self.find_hal()
#       print(time.time() - tm)
#       self.metal_binding = self.find_metal_binding()
#       print(time.time() - tm)

#       self.atom_prop_dict = config.atom_prop_dict
#       print(time.time() - tm)

        # ADDED dummy variables so that PLInteraction will work with two Protein objects
#       self.halogenbond_don = []
#       self.water = []


#       self.metals = []
#       data = namedtuple('metal', 'm orig_m m_orig_idx m_coords')
#       atomic_symbols = np.array([atom.GetSymbol().upper() for atom in self.all_atoms])
#       metal_atoms_mask = np.isin(atomic_symbols, config.METAL_IONS)

#       for a in np.array(self.all_atoms)[metal_atoms_mask]:
#           m_orig_idx = a.GetIdx()
#           orig_m = m_orig_idx
#           self.metals.append(
#               data(
#                   m=a,
#                   m_orig_idx=m_orig_idx,
#                   orig_m=orig_m,
#                   m_coords=self.atoms_coords[m_orig_idx]))


    def get_atomic_radius(self, atom):
        radius = {"N": 1.8, "O": 1.7, "S": 2.0, "P": 2.1, "F": 1.5, "Cl": 1.8,
          "Br": 2.0, "I": 2.2, "C": 1.9, "H": 0.0, "Zn": 0.5, "B": 1.8,
          "Si": 1.8, "As": 1.8, "Se": 1.8}
        atomic_symbol = atom.GetSymbol()
        return radius.get(atomic_symbol, np.nan) 



    def vectorize_features(self):
        # Atom vectors
        self.atom_idx = np.arange(self.natom)
        self.atom_name = np.zeros(self.natom, str)
        self.atom_radii = np.zeros(self.natom, float)
        self.atom_charge = np.zeros(self.natom, float)
        self.atom_coord = np.zeros((self.natom, 3), float)

        # Residue vectors (for each atom)
        self.residue_idx = np.zeros(self.natom, int)
        self.residue_name = np.zeros(self.natom, object)
        self.is_ca = np.zeros(self.natom, bool)

        # Load alternative format (for partial charges and easy coord access)
        mol = pybel.readstring("pdb", Chem.MolToPDBBlock(self.mol))

        for i, (atom, alt_atom) in enumerate(zip(self.all_atoms, mol.atoms)):
            self.atom_name[i] = atom.GetSymbol()
            self.atom_radii[i] = self.get_atomic_radius(atom)
            self.atom_charge[i] = alt_atom.partialcharge
            self.atom_coord[i] = alt_atom.coords

            res_info = atom.GetPDBResidueInfo()
            self.residue_idx[i] = res_info.GetResidueNumber()
            self.residue_name[i] = res_info.GetResidueName()
            self.is_ca[i] = 'CA' in res_info.GetName()

    
    def calculate_atomic_radii(self):
        atomic_radii = np.zeros(len(self.all_atoms))
        for i, atom in enumerate(self.all_atoms):
            atomic_radii[i] = self.get_atomic_radius(atom)
        return atomic_radii
    
    def calculate_partial_charges(self, mol_prot):
        tm = time.time()
        print("Inside partial charges")
        mol = pybel.readstring("pdb", Chem.MolToPDBBlock(mol_prot))
        print(time.time() - tm)
        charges = [atom.partialcharge for atom in mol.atoms]
        print(time.time() - tm)
        return np.array(charges)
                                    
    def getAll_atoms_coord(self):
        atom_coords = np.zeros((len(self.all_atoms), 3))
        for atom in self.all_atoms:
            mol = atom.GetOwningMol()
            conf = mol.GetConformers()[0]
            pos = conf.GetAtomPosition(atom.GetIdx())
            atom_coords[atom.GetIdx()] = (pos.x, pos.y, pos.z)
        return atom_coords

    def hydrophobic_atoms(self):
        """Select all carbon atoms which have only carbons and/or hydrogens as direct neighbors."""
        carbon_idx = np.where(self.atom_name == 6)[0]
        hphob_idx = np.array([i for i in carbon_idx if {natom.GetAtomicNum() for natom in self.all_atoms[i].GetNeighbors()} <= {1, 6}])

        # Just return the indices
        return hphob_idx
#       filter_mask = is_atomic_num_6 & neighbors_1_or_6
#       atm = np.array(self.all_atoms)[filter_mask]

#       for atom in atm:
#           orig_idx = atom.GetIdx()
#           orig_atom = orig_idx
#           atom_set.append(
#               data(
#                   atom=atom,
#                   orig_atom=orig_atom,
#                   orig_idx=orig_idx,
#                   coords=self.atoms_coords[orig_idx]))
#       return atom_set

    def find_hba(self):
        # Find indices for atoms that are not H or C
        potential_acceptor_idx = np.where(~np.in1d(self.atom_name, ['H', 'C']))[0]
        hba_idx = np.array([i for i in potential_acceptor_idx if is_acceptor(self.atom_name[i], self.residue_name[i])], int)
        return hba_idx


    def find_hbd(self):
        donor_pairs = []
        # Find non-carbon donor pairs
        potential_donor_idx = np.where(~np.in1d(self.atom_name, ['H', 'C']))[0]
        for i in potential_donor_idx:
            i = int(i)
            if is_donor(self.atom_name[i], self.residue_name[i]):
                neighbor_H = [a for a in self.all_atoms[i].GetNeighbors() if a.GetAtomicNum() == 1]
                for atom in neighbor_H:
                    donor_pairs.append([i, atom.GetIdx()])

        # Find carbon donor pairs
        for i in np.where(self.atom_name == 'C')[0]:
            i = int(i)
            neighbor_H = [a for a in self.all_atoms[i].GetNeighbors() if a.GetAtomicNum() == 1]
            for atom in neighbor_H:
                donor_pairs.append([i, atom.GetIdx()])
        return np.array(donor_pairs, int)


    def find_rings(self):
        """Find rings and return only aromatic.
        Rings have to be sufficiently planar OR be detected by OpenBabel as aromatic."""
        data = namedtuple(
            'aromatic_ring',
            'atoms orig_atoms atoms_orig_idx normal obj center type')
        rings = []
        aromatic_amino = ['TYR', 'TRP', 'HIS', 'PHE']
        ring_info = self.mol.GetRingInfo()
        rings_atom_idx = ring_info.AtomRings()
        for ring in rings_atom_idx:
            if 4 < len(ring) <= 6:
                r_atoms = [self.mol.GetAtomWithIdx(idx) for idx in ring]
                r_atoms = sorted(r_atoms, key=lambda x: x.GetIdx())
                atom_pos = self.atom_coord[list(ring)]

                res = list(set([whichrestype(a) for a in r_atoms]))
                if res[0] == "UNL":
                    ligand_orig_idx = ring
                    sort_order = np.argsort(np.array(ligand_orig_idx))
                    r_atoms = [r_atoms[i] for i in sort_order]
                if is_aromatic(r_atoms) or res[0] in aromatic_amino or ring_is_planar(
                        atom_pos, ring, r_atoms):
                    ring_type = '%s-membered' % len(ring)
                    ring_atms = atom_pos[[0,2,4]]
                    ringv1 = vector(ring_atms[0], ring_atms[1])
                    ringv2 = vector(ring_atms[2], ring_atms[0])

                    atoms_orig_idx = [r_atom.GetIdx() for r_atom in r_atoms]
                    orig_atoms = r_atoms
                    rings.append(
                        data(
                            atoms=r_atoms,
                            orig_atoms=orig_atoms,
                            atoms_orig_idx=atoms_orig_idx,
                            normal=normalize_vector(
                                np.cross(
                                    ringv1,
                                    ringv2)),
                            obj=ring,
                            center=np.mean(atom_pos, axis=0),
                            type=ring_type))

        return rings


#   def find_hal(self):
#       """Look for halogen bond acceptors (Y-{O|P|N|S}, with Y=C,P,S)"""
#       data = namedtuple(
#           'hal_acceptor',
#           'o o_orig_idx y y_orig_idx o_coords y_coords')
#       hal_pair = []
#       # All oxygens, nitrogen, sulfurs with neighboring carbon, phosphor,
#       # nitrogen or sulfur
#       is_N_O_S = np.in1d(self.atom_name, ['N', 'O', 'S'])
#       for i in np.where(is_N_O_S)[0]:
#           i = int(i)
#           neighbors = [a for a in self.all_atoms[i].GetNeighbors() if a.GetAtomicNum() in [6, 7, 15, 16]]
#           if len(neighbors) == 1:  # Proximal atom
#               hal_pair.append([i, neighbors[0].GetIdx()])
#       return np.array(hal_pair)


    ### For some reason they originally excluded backbone atoms...
    ### Can't think of any negative consequence of keeping them in,
    ### other than they might rarely be involved in signficant
    ### interactions...?
    def find_charged(self):
        """Looks for positive charges in arginine, histidine or lysine, for negative in aspartic and glutamic acid."""
        is_charged = np.in1d(self.residue_name, ['ARG', 'HIS', 'LYS', 'GLU', 'ASP'])
        is_N_O = np.in1d(self.atom_name, ['N', 'O'])
        charged_idx = np.where(is_charged & is_N_O)[0]
        return charged_idx


    ### Ask a chemist or other person about this!
    ### It could be important later...
#   def find_metal_binding(self):
#       """Looks for atoms that could possibly be involved in chelating a metal ion.
#       This can be any main chain oxygen atom or oxygen, nitrogen and sulfur from specific amino acids"""
#       data = namedtuple(
#           'metal_binding',
#           'atom atom_orig_idx type restype resnr reschain location coords')
#       a_set = []
#       for res in self.residues:
#           restype, resnr = res.residue_name, res.residue_number
#           reschain = 'P'
#           if restype in ("ASP", "GLU", "SER", "THR", "TYR"):
#               for a in res.residue_atoms:
#                   if a.GetSymbol() == "O" and a.GetPDBResidueInfo().GetName().strip(" ") != "O":
#                       atom_orig_idx = a.GetIdx()
#                       a_set.append(
#                           data(
#                               atom=a,
#                               atom_orig_idx=atom_orig_idx,
#                               type='O',
#                               restype=restype,
#                               resnr=resnr,
#                               reschain=reschain,
#                               coords=self.atoms_coords[atom_orig_idx],
#                               location='protein.sidechain'))
#           if restype == 'HIS':  # Look for nitrogen here
#               for a in res.residue_atoms:
#                   if a.GetSymbol() == "N" and a.GetPDBResidueInfo().GetName().strip(" ") != "N":
#                       atom_orig_idx = a.GetIdx()
#                       a_set.append(
#                           data(
#                               atom=a,
#                               atom_orig_idx=atom_orig_idx,
#                               type='N',
#                               restype=restype,
#                               resnr=resnr,
#                               reschain=reschain,
#                               coords=self.atoms_coords[atom_orig_idx],
#                               location='protein.sidechain'))
#           if restype == 'CYS':  # Look for sulfur here
#               for a in res.residue_atoms:
#                   if a.GetSymbol() == "S":
#                       atom_orig_idx = a.GetIdx()
#                       a_set.append(
#                           data(
#                               atom=a,
#                               atom_orig_idx=atom_orig_idx,
#                               type='S',
#                               restype=restype,
#                               resnr=resnr,
#                               reschain=reschain,
#                               coords=self.atoms_coords[atom_orig_idx],
#                               location='protein.sidechain'))

#           for a in res.residue_atoms:  # All main chain oxygens
#               if a.GetSymbol() == "O" and a.GetPDBResidueInfo().GetName().strip(" ") == "O":
#                   atom_orig_idx = a.GetIdx()
#                   a_set.append(
#                       data(
#                           atom=a,
#                           atom_orig_idx=atom_orig_idx,
#                           type='O',
#                           restype=res.residue_name,
#                           resnr=res.residue_number,
#                           reschain=reschain,
#                           coords=self.atoms_coords[atom_orig_idx],
#                           location='protein.mainchain'))
#       return a_set

class PLInteraction:
    """Class to store a protein and its self interactions."""

    def __init__(self, prot):
        """Detect all self interactions when initializing"""
        self.protein = prot

        self.dist_mat = self.calc_dist_mat()
#       self.vectors = self.calc_vectors()
#       self.atomic_radii_sum = self.calc_atomic_radii_sum()
        self.is_intra_residue = self.calc_is_intra_residue()

        # #@todo Refactor code to combine different directionality

        self.saltbridge_idx = self.get_salt_bridges()
        self.hbond_idx = self.get_hbonds()
        self.vdw_idx = self.get_vdw()

        # Ignore 'hydrophobic interactions' for now,
        # since they will be duplicated too many times 
        # in other interactions
#       self.hphob_idx = self.hydrophobic_interactions()

        ### Probably need something like this to reduce the contribution
        ### of hydrophobic contacts
#       self.hydrophobic_contacts = self.refine_hydrophobic(
#           self.all_hydrophobic_contacts, self.pistacking)

        prot_rings = self.protein.rings
        self.pistacking = pistacking(prot_rings, prot_rings)

#       self.pi_cation_laro = pication(prot_rings, prot_pchar)
#       self.all_pi_cation_laro = pication(prot_rings, prot_pchar, True)
#       self.pication_paro = pication(prot_rings, prot_pchar, False)

#       self.pication_laro = self.refine_pi_cation_laro(
#           self.all_pi_cation_laro, self.pistacking)

#       self.halogen_bonds = halogen(
#           self.protein.halogenbond_acc,
#           self.ligand.halogenbond_don)
        
#       prot_hba, prot_hbd = self.protein.get_hba(), self.protein.get_hbd()
#       self.all_water_bridges = water_bridges(prot_hba, prot_hba, prot_hbd, prot_hbd, self.ligand.water)

#       self.water_bridges = self.refine_water_bridges(
#           self.all_water_bridges, self.hbonds_ldon, self.hbonds_pdon)

#       self.metal_complexes = metal_complexation(
#           self.protein.metals,
#           self.ligand.metal_binding,
#           self.protein.metal_binding)

#       self.all_itypes = self.saltbridge_lneg + \
#           self.saltbridge_pneg + self.hbonds_pdon
#       self.all_itypes = self.all_itypes + self.hbonds_ldon + \
#           self.pistacking + self.pication_laro + self.pication_paro
#       self.all_itypes = self.all_itypes + self.hydrophobic_contacts + \
#           self.halogen_bonds + self.water_bridges
#       self.all_itypes = self.all_itypes + self.metal_complexes

#       self.no_interactions = all(len(i) == 0 for i in self.all_itypes)
#       self.unpaired_hba, self.unpaired_hbd, self.unpaired_hal = self.find_unpaired_ligand()
#       self.unpaired_hba_orig_idx = [atom.GetIdx()
#                                     for atom in self.unpaired_hba]
#       self.unpaired_hbd_orig_idx = [atom.GetIdx()
#                                     for atom in self.unpaired_hbd]
#       self.unpaired_hal_orig_idx = [atom.GetIdx()
#                                     for atom in self.unpaired_hal]
#       self.num_unpaired_hba, self.num_unpaired_hbd = len(
#           self.unpaired_hba), len(self.unpaired_hbd)
#       self.num_unpaired_hal = len(self.unpaired_hal)

#       self.hphob_idx = self.hphob_calcInd(lig_obj)
#       self.hbond_idx = self.hbond_calcInd()
#       self.vdw_idx = self.vdw_calcInd(lig_obj, bs_obj)

        # Exclude empty chains (coming from ligand as a target, from metal
        # complexes)
#       self.interacting_chains = sorted(list(set([i.reschain for i in self.all_itypes
#                                                  if i.reschain not in [' ', None]])))

        # Get all interacting residues, excluding ligand and water molecules
#       self.interacting_res = list(set([''.join([str(i.resnr), str(
#           i.reschain)]) for i in self.all_itypes if i.restype not in ['LIG', 'HOH']]))
#       if len(self.interacting_res) != 0:
#           interactions_list = []
#           num_saltbridges = len(self.saltbridge_lneg + self.saltbridge_pneg)
#           num_hbonds = len(self.hbonds_ldon + self.hbonds_pdon)
#           num_pication = len(self.pication_laro + self.pication_paro)
#           num_pistack = len(self.pistacking)
#           num_halogen = len(self.halogen_bonds)
#           num_waterbridges = len(self.water_bridges)
#           if num_saltbridges != 0:
#               interactions_list.append('%i salt bridge(s)' % num_saltbridges)
#           if num_hbonds != 0:
#               interactions_list.append('%i hydrogen bond(s)' % num_hbonds)
#           if num_pication != 0:
#               interactions_list.append(
#                   '%i pi-cation interaction(s)' %
#                   num_pication)
#           if num_pistack != 0:
#               interactions_list.append('%i pi-stacking(s)' % num_pistack)
#           if num_halogen != 0:
#               interactions_list.append('%i halogen bond(s)' % num_halogen)
#           if num_waterbridges != 0:
#               interactions_list.append(
#                   '%i water bridge(s)' %
#                   num_waterbridges)
#           if not len(interactions_list) == 0:
#               # raise RuntimeWarning(f'complex uses {interactions_list}')
#               #print(f'complex uses {interactions_list}')
#               pass
#       else:
#           # raise RuntimeWarning('no interactions for this ligand')
#           print('no interactions for this ligand')


    def calc_dist_mat(self):
        # Calculate the pairwise distances between binding site and ligand atoms
        dist_matrix = cdist(*[self.protein.atom_coord]*2, 'euclidean')
        return dist_matrix


    def calc_vectors(self):
        # Calculate normal vectors between all pairs of atoms
        vectors = self.protein.atom_coord.reshape(-1, 1, 3) - self.protein.atom_coord
        vectors = vectors / self.dist_mat.reshape(self.dist_mat.shape + (1,))
        dist_mat = self.dist_mat.copy()
        np.fill_diagonal(dist_mat, 1)
        return vectors / dist_mat.reshape(self.dist_mat.shape + (1,))

    
#   def calc_atomic_radii_sum(self):
#      # Calculate the sum of atomic radii between binding site and ligand atoms
#       return np.sum(np.meshgrid(*[self.protein.atom_radii]*2), axis=0)

    
    def calc_is_intra_residue(self):
       # Calculate the sum of atomic radii between binding site and ligand atoms
        return ~np.equal(*np.meshgrid(*[self.protein.residue_idx]*2))

    
    def get_salt_bridges(self):
        # Get indices of positively / negatively charged atoms
        charge = self.protein.atom_charge[self.protein.charged]
        pos_idx = self.protein.charged[charge > 0]
        neg_idx = self.protein.charged[charge < 0]

        # Find oppositely-charged pairs within cutoff distance,
        # that are not part of the same residue
        dist_okay = self.dist_mat[pos_idx][:,neg_idx] < config.SALTBRIDGE_DIST_MAX
        intra_res = self.is_intra_residue[pos_idx][:,neg_idx]
        i, j = np.where((dist_okay) & (intra_res))
        return np.array([pos_idx[i], neg_idx[j]]).T
    

    def get_hbonds(self):
        acc_idx = self.protein.hbond_acc_idx
        don_idx = self.protein.hbond_don_idx

        # Find atom pairs within a certain distance,
        # that are not in the same residue
        dist_ad = self.dist_mat[acc_idx][:,don_idx[:,0]]
        dist_okay = (config.MIN_DIST < dist_ad) & (dist_ad < config.HBOND_DIST_MAX)
        intra_res = self.is_intra_residue[acc_idx][:,don_idx[:,0]]

        i, j = np.where(dist_okay & intra_res)

        # Calculate H-bond angles, and only include if > 120 degrees
        dond_coord = self.protein.atom_coord[don_idx[j,0]]#.reshape(-1,1,3)
        donh_coord = self.protein.atom_coord[don_idx[j,1]]#.reshape(1,-1,3)
        acc_coord = self.protein.atom_coord[acc_idx[i]]#.reshape(1,-1,3)

        vec_hd = (donh_coord - dond_coord) / self.dist_mat[don_idx[j,1], don_idx[j,0]].reshape(-1, 1)
        vec_ha = (donh_coord - acc_coord) / self.dist_mat[don_idx[j,1], acc_idx[i]].reshape(-1, 1)


        angle = np.degrees(np.arccos(np.sum(vec_hd * vec_ha, axis=1)))
        angle_okay = angle > config.HBOND_DON_ANGLE_MIN

        idx = np.where(angle_okay)[0]
        acc_idx = acc_idx[i][idx]
        don_idx = don_idx[j][idx]
        return np.array([acc_idx, don_idx[:,0]]).T
            

    def get_vdw(self):
        not_hyd = np.where(self.protein.atom_name != 'H')[0]
        dist_okay = self.dist_mat[not_hyd][:,not_hyd] < 6.0
        is_intra = self.is_intra_residue[not_hyd][:,not_hyd]
        return not_hyd[np.array([[i, j] for i, j in zip(*np.where(dist_okay & is_intra)) if i < j])]

    
    def hydrophobic_interactions(self):
        """Detection of hydrophobic pliprofiler between atom_set_a (binding site) and atom_set_b (ligand).
        Definition: All pairs of qualified carbon atoms within a distance of HYDROPH_DIST_MAX
        """
        hp_atom_idx = self.protein.hydrophobic_idx.astype(int)
        distances = self.dist_mat[hp_atom_idx][:,hp_atom_idx]

        dist_okay = (config.MIN_DIST < distances) & (distances < config.HYDROPH_DIST_MAX)
        intra_res = self.is_intra_residue[hp_atom_idx][:,hp_atom_idx]

        i, j = np.where(dist_okay & intra_res)
        atom_idx = np.where(hp_atom_idx)[0]
        atom_pairs = np.array([atom_idx[i], atom_idx[j]]).T
        return atom_pairs


    def refine_hydrophobic(self, all_h, pistacks):
        """Apply several rules to reduce the number of hydrophobic interactions."""
        sel = {}
        # 1. Rings interacting via stacking can't have additional hydrophobic
        # contacts between each other.
        for pistack, h in itertools.product(pistacks, all_h):
            h1, h2 = h.bsatom.GetIdx(), h.ligatom.GetIdx()
            brs, lrs = [p1.GetIdx() for p1 in pistack.proteinring.atoms], [
                p2.GetIdx() for p2 in pistack.ligandring.atoms]
            if h1 in brs and h2 in lrs:
                sel[(h1, h2)] = "EXCLUDE"
        hydroph = [
            h for h in all_h if not (
                h.bsatom.GetIdx(),
                h.ligatom.GetIdx()) in sel]
        sel2 = {}
        #  2. If a ligand atom interacts with several binding site atoms in the same residue,
        #  keep only the one with the closest distance
        for h in hydroph:
            if not (h.ligatom.GetIdx(), h.resnr) in sel2:
                sel2[(h.ligatom.GetIdx(), h.resnr)] = h
            else:
                if sel2[(h.ligatom.GetIdx(), h.resnr)].distance > h.distance:
                    sel2[(h.ligatom.GetIdx(), h.resnr)] = h
        hydroph = [h for h in sel2.values()]
        hydroph_final = []
        bsclust = {}
        # 3. If a protein atom interacts with several neighboring ligand atoms,
        # just keep the one with the closest dist
        for h in hydroph:
            if h.bsatom.GetIdx() not in bsclust:
                bsclust[h.bsatom.GetIdx()] = [h, ]
            else:
                bsclust[h.bsatom.GetIdx()].append(h)

        idx_to_h = {}
        for bs in [a for a in bsclust if len(bsclust[a]) == 1]:
            hydroph_final.append(bsclust[bs][0])

        # A list of tuples with the idx of an atom and one of its neighbours is
        # created
        for bs in [a for a in bsclust if not len(bsclust[a]) == 1]:
            tuples = []
            all_idx = [i.ligatom.GetIdx() for i in bsclust[bs]]
            for b in bsclust[bs]:
                idx = b.ligatom.GetIdx()
                neigh = [na for na in b.ligatom.GetNeighbors()]
                for n in neigh:
                    n_idx = n.GetIdx()
                    if n_idx in all_idx:
                        if n_idx < idx:
                            tuples.append((n_idx, idx))
                        else:
                            tuples.append((idx, n_idx))
                        idx_to_h[idx] = b

            tuples = list(set(tuples))
            tuples = sorted(tuples, key=itemgetter(1))
            # Cluster connected atoms (i.e. find hydrophobic patches)
            clusters = cluster_doubles(tuples)

            for cluster in clusters:
                min_dist = float('inf')
                min_h = None
                for atm_idx in cluster:
                    h = idx_to_h[atm_idx]
                    if h.distance < min_dist:
                        min_dist = h.distance
                        min_h = h
                hydroph_final.append(min_h)
        before, reduced = len(all_h), len(hydroph_final)
        if not before == 0 and not before == reduced:
            # raise RuntimeWarning(f'reduced number of hydrophobic contacts from {before} to {reduced}')
            #print(f'reduced number of hydrophobic contacts from {before} to {reduced}')
            pass
        return hydroph_final

    def refine_hbonds_ldon(self, all_hbonds, salt_lneg, salt_pneg):
        """Refine selection of hydrogen bonds. Do not allow groups which already form salt bridges to form H-Bonds."""
        i_set = {}
        for hbond in all_hbonds:
            i_set[hbond] = False
            for salt in salt_pneg:
                protidx, ligidx = [
                    at.GetIdx() for at in salt.negative.atoms], [
                    at.GetIdx() for at in salt.positive.atoms]
                if hbond.d.GetIdx() in ligidx and hbond.a.GetIdx() in protidx:
                    i_set[hbond] = True
            for salt in salt_lneg:
                protidx, ligidx = [
                    at.GetIdx() for at in salt.positive.atoms], [
                    at.GetIdx() for at in salt.negative.atoms]
                if hbond.d.GetIdx() in ligidx and hbond.a.GetIdx() in protidx:
                    i_set[hbond] = True

        # Allow only one hydrogen bond per donor, select interaction with
        # larger donor angle
        second_set = {}
        hbls = [k for k in i_set.keys() if not i_set[k]]
        for hbl in hbls:
            if hbl.d.GetIdx() not in second_set:
                second_set[hbl.d.GetIdx()] = (hbl.angle, hbl)
            else:
                if second_set[hbl.d.GetIdx()][0] < hbl.angle:
                    second_set[hbl.d.GetIdx()] = (hbl.angle, hbl)
        return [hb[1] for hb in second_set.values()]

    def refine_hbonds_pdon(self, all_hbonds, salt_lneg, salt_pneg):
        """Refine selection of hydrogen bonds. Do not allow groups which already form salt bridges to form H-Bonds with
        atoms of the same group.
        """
        i_set = {}
        for hbond in all_hbonds:
            i_set[hbond] = False
            for salt in salt_lneg:
                protidx, ligidx = [
                    at.GetIdx() for at in salt.positive.atoms], [
                    at.GetIdx() for at in salt.negative.atoms]
                if hbond.a.GetIdx() in ligidx and hbond.d.GetIdx() in protidx:
                    i_set[hbond] = True
            for salt in salt_pneg:
                protidx, ligidx = [
                    at.GetIdx() for at in salt.negative.atoms], [
                    at.GetIdx() for at in salt.positive.atoms]
                if hbond.a.GetIdx() in ligidx and hbond.d.GetIdx() in protidx:
                    i_set[hbond] = True

        # Allow only one hydrogen bond per donor, select interaction with
        # larger donor angle
        second_set = {}
        hbps = [k for k in i_set.keys() if not i_set[k]]
        for hbp in hbps:
            if hbp.d.GetIdx() not in second_set:
                second_set[hbp.d.GetIdx()] = (hbp.angle, hbp)
            else:
                if second_set[hbp.d.GetIdx()][0] < hbp.angle:
                    second_set[hbp.d.GetIdx()] = (hbp.angle, hbp)
        return [hb[1] for hb in second_set.values()]

    def refine_pi_cation_laro(self, all_picat, stacks):
        """Just important for constellations with histidine involved. If the histidine ring is positioned in stacking
        position to an aromatic ring in the ligand, there is in most cases stacking and pi-cation interaction reported
        as histidine also carries a positive charge in the ring. For such cases, only report stacking.
        """
        i_set = []
        for picat in all_picat:
            exclude = False
            for stack in stacks:
                if whichrestype(
                        stack.proteinring.atoms[0]) == 'HIS' and picat.ring.obj == stack.ligandring.obj:
                    exclude = True
            if not exclude:
                i_set.append(picat)
        return i_set

    def refine_water_bridges(self, wbridges, hbonds_ldon, hbonds_pdon):
        """A donor atom already forming a hydrogen bond is not allowed to form a water bridge. Each water molecule
        can only be donor for two water bridges, selecting the constellation with the omega angle closest to 110 deg."""
        donor_atoms_hbonds = [hb.d.GetIdx()
                              for hb in hbonds_ldon + hbonds_pdon]
        wb_dict = {}
        wb_dict2 = {}
        omega = 110.0

        # Just one hydrogen bond per donor atom
        for wbridge in [
                wb for wb in wbridges if wb.d.GetIdx() not in donor_atoms_hbonds]:
            if (wbridge.water.GetIdx(), wbridge.a.GetIdx()) not in wb_dict:
                wb_dict[(wbridge.water.GetIdx(), wbridge.a.GetIdx())] = wbridge
            else:
                if abs(omega - wb_dict[(wbridge.water.GetIdx(),
                                        wbridge.a.GetIdx())].w_angle) < abs(omega - wbridge.w_angle):
                    wb_dict[(wbridge.water.GetIdx(),
                             wbridge.a.GetIdx())] = wbridge
        for wb_tuple in wb_dict:
            water, acceptor = wb_tuple
            if water not in wb_dict2:
                wb_dict2[water] = [
                    (abs(
                        omega -
                        wb_dict[wb_tuple].w_angle),
                        wb_dict[wb_tuple]),
                ]
            elif len(wb_dict2[water]) == 1:
                wb_dict2[water].append(
                    (abs(omega - wb_dict[wb_tuple].w_angle), wb_dict[wb_tuple]))
                wb_dict2[water] = sorted(wb_dict2[water], key=lambda x: x[0])
            else:
                if wb_dict2[water][1][0] < abs(
                        omega - wb_dict[wb_tuple].w_angle):
                    wb_dict2[water] = [ wb_dict2[water][0],
                        (wb_dict[wb_tuple].w_angle, wb_dict[wb_tuple])]

        filtered_wb = []
        for fwbridges in wb_dict2.values():
            [filtered_wb.append(fwb[1]) for fwb in fwbridges]
        return filtered_wb


### MODIFY THIS SO THAT IT RUNS WITH TWO PROTEIN OBJECTS
def get_interactions(mol_protein, pdbid=None):
    data = namedtuple("interaction", "lig prot interactions")
    prot = Protein(mol_protein)

    interactions = PLInteraction(prot)
    return data(lig=prot, prot=prot, interactions=interactions)
