import re
from ase.data import atomic_numbers, covalent_radii, atomic_masses
from mendeleev import element
import numpy as np
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Informatics.lacRACAssemble import get_descriptor_vector
from molSimplify.Informatics.rac155_geo import rac155_list

# Global normalization constants
MAX_Z = float(max(atomic_numbers.values()))
MAX_RC = float(np.nanmax(covalent_radii))
MAX_WEIGHT = float(np.nanmax(atomic_masses))
# On Pauling scale, electronegativity max ~4.0 (F)
MAX_CHI = 4.0

def get_node_features(elements):
    """
    Given a list of atomic symbols, returns normalized feature arrays:
      Zn, Rcn, wn, chin
    """
    Z = np.array([atomic_numbers[s] for s in elements], dtype=float)
    Rc = np.array([covalent_radii[atomic_numbers[s]] for s in elements], dtype=float)
    weight = np.array([atomic_masses[atomic_numbers[s]] for s in elements], dtype=float)
    chi = np.array([
        element(s).en_pauling if element(s).en_pauling is not None else 0.0
        for s in elements
    ], dtype=float)

    # Normalize by global max
    Zn = Z / MAX_Z
    Rcn = Rc / MAX_RC
    wn = weight / MAX_WEIGHT
    chin = chi / MAX_CHI

    return Zn, Rcn, wn, chin

def compute_rac(xyz_path):
    mol = mol3D()
    mol.readfromxyz(xyz_path)

    # 1) compute all 184 diagnostics
    names, values = get_descriptor_vector(mol)

    # Remove the two metadata entries
    features_153 = [name for name in rac155_list if name not in ('ox', 'alpha')]

    # Extract the corresponding values by matching against all_names
    values_153 = [values[ names.index(name) ] for name in features_153]
    return values_153

def normalize_molecule_name(name):
    """
    Strip oxidation state tags (e.g. _II, _III) and spin flags (_HS, _LS)
    from an XYZ basename to match CSV 'molecule' entries.
    """
    name = re.sub(r'_(II|III)', '', name)
    name = re.sub(r'_(HS|LS)$', '', name)
    return name