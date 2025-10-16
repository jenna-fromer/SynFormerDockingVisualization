""" Utility functions to generate docked pose interactions as 3D poses and 2D interaction maps using ProLif """

import io
from contextlib import redirect_stdout, redirect_stderr

from rdkit import Chem
from prolif import Molecule
import prolif as plf
import MDAnalysis as mda
import py3Dmol

u = mda.Universe(
    '../data/pli_pdbs/oe_protonated_MOR_endorphin_usedforDOCK.pdb')
protein_mol = plf.Molecule.from_mda(u)

HBHD_parameters = {
    'HBDonor': {
        'distance': 4, 'DHA_angle':(0, 180), # very flexible definition
        'acceptor': '[#8&!$(*~N~[OD1]),#7&H0;!$([D4]);!$([D3]-*=,:[$([#7,#8,#15,#16])])]', 
        'donor': '[$([#7,#8,#15,#16])]-[H]'
    }, 
    'HBAcceptor': {
        'distance': 4, 'DHA_angle':(0, 180), # very flexible definition
        'acceptor': '[#8&!$(*~N~[OD1]),#7&H0;!$([D4]);!$([D3]-*=,:[$([#7,#8,#15,#16])])]', 
        'donor': '[$([#7,#8,#15,#16])]-[H]'
    }, 
}

def make_prolif_viewer(sdf_path: str = 'data/redocked/call_78271.sdf'):
    """
    Create 3D and 2D ProLIF visualizations of ligand–protein interactions.

    Parameters
    ----------
    sdf_path : str, optional
        Path to the ligand SDF file.

    Returns
    -------
    view_3d : py3Dmol.view or None
        3D interaction viewer.
    map_2d_html : str or None
        2D interaction map in HTML format.
    """
    f = io.StringIO()
    with redirect_stdout(f), redirect_stderr(f):  # suppress all stdout and stderr
        mol_supplier = Chem.SDMolSupplier(sdf_path)
        mol = [m for m in mol_supplier][0]
        plf_molec = Molecule.from_rdkit(mol)

        try: 
            fp_count = plf.Fingerprint(
                interactions=[
                    'Hydrophobic', 'HBDonor', 'HBAcceptor', 
                    'PiStacking', 'Anionic', 'Cationic', 
                    'CationPi', 'PiCation', 'VdWContact'
                ], 
                parameters=HBHD_parameters, count=True,
            )
            fp_count.run_from_iterable([plf_molec], protein_mol)
            view_3d = fp_count.plot_3d(
                plf_molec, protein_mol, frame=0, display_all=False
            )
            view_3d.addSurface(py3Dmol.SES, {'opacity':0.70, 'colorscheme':'rwb'})
        except:
            view_3d = None

        try: 
            fp_count_2d = plf.Fingerprint(
                interactions=[
                    'HBDonor', 'HBAcceptor', 'PiStacking', 
                    'Anionic', 'Cationic', 'CationPi', 'PiCation'
                ], parameters=HBHD_parameters, count=True)
            fp_count_2d.run_from_iterable([plf_molec], protein_mol)
            map_2d_html = fp_count_2d.plot_lignetwork(
                plf_molec, kind="frame", frame=0, display_all=True, width="500px", height="300px"
            )
        except:
            map_2d_html = None

    return view_3d, map_2d_html

if __name__=='__main__': 
    make_prolif_viewer()