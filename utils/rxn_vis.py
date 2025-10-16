""" Utility functions to visualize reactions and synthesis paths proposed by SynFormer """

import itertools
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Union
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw, rdChemReactions
from rdkit.Chem.Draw import rdMolDraw2D

# Suppress all RDKit logging messages
RDLogger.DisableLog('rdApp.*') 

with open('data/rxn_templates/comprehensive.txt', 'r') as f:
    lines = f.readlines()
    RXN_LIST = [line.strip() for line in lines]

    
def get_products(smirks: str, reactants: Union[list, str], keep_main_product: bool = True) -> str:
    """
    Run an RDKit reaction SMIRKS on given reactants and return the product SMILES.
    
    Inputs:
        smirks (str): Reaction SMARTS/SMIRKS pattern.
        reactants (list[str] | str): Reactant SMILES strings.
        keep_main_product (bool): If True, return only the primary product.

    Outputs:
        str: Product SMILES string(s) or "None" if reaction fails.
    """

    rxn = AllChem.ReactionFromSmarts(smirks)
    rdChemReactions.ChemicalReaction.Initialize(rxn)
    if type(reactants) == str:
        reactants = [reactants]
    assert rxn.GetNumReactantTemplates() == len(reactants)
    r = tuple(Chem.MolFromSmiles(smiles) for smiles in reactants if smiles is not None)
    if len(reactants) == 1:
        if rxn.IsMoleculeReactant(r[0]):
            pass
        else:
            return "None"
    elif len(reactants) == 2:
        if Chem.MolFromSmiles(reactants[0]).HasSubstructMatch(rxn.GetReactants()[0]) and Chem.MolFromSmiles(reactants[0]).HasSubstructMatch(rxn.GetReactants()[0]):
            pass
        elif Chem.MolFromSmiles(reactants[0]).HasSubstructMatch(rxn.GetReactants()[1]) and Chem.MolFromSmiles(reactants[1]).HasSubstructMatch(rxn.GetReactants()[0]):
            r = tuple(reversed(r))
        else:
            return "None"
    elif len(reactants) == 3:
        if Chem.MolFromSmiles(reactants[0]).HasSubstructMatch(rxn.GetReactants()[0]) \
            and Chem.MolFromSmiles(reactants[1]).HasSubstructMatch(rxn.GetReactants()[1]) \
            and Chem.MolFromSmiles(reactants[2]).HasSubstructMatch(rxn.GetReactants()[2]):
            pass
        elif Chem.MolFromSmiles(reactants[0]).HasSubstructMatch(rxn.GetReactants()[0]) \
            and Chem.MolFromSmiles(reactants[2]).HasSubstructMatch(rxn.GetReactants()[1]) \
            and Chem.MolFromSmiles(reactants[1]).HasSubstructMatch(rxn.GetReactants()[2]):
            r = (r[0], r[2], r[1])
        elif Chem.MolFromSmiles(reactants[1]).HasSubstructMatch(rxn.GetReactants()[0]) \
            and Chem.MolFromSmiles(reactants[0]).HasSubstructMatch(rxn.GetReactants()[1]) \
            and Chem.MolFromSmiles(reactants[2]).HasSubstructMatch(rxn.GetReactants()[2]):
            r = (r[1], r[0], r[2])
        elif Chem.MolFromSmiles(reactants[1]).HasSubstructMatch(rxn.GetReactants()[0]) \
            and Chem.MolFromSmiles(reactants[2]).HasSubstructMatch(rxn.GetReactants()[1]) \
            and Chem.MolFromSmiles(reactants[0]).HasSubstructMatch(rxn.GetReactants()[2]):
            r = (r[1], r[2], r[0])
        elif Chem.MolFromSmiles(reactants[2]).HasSubstructMatch(rxn.GetReactants()[0]) \
            and Chem.MolFromSmiles(reactants[0]).HasSubstructMatch(rxn.GetReactants()[1]) \
            and Chem.MolFromSmiles(reactants[1]).HasSubstructMatch(rxn.GetReactants()[2]):
            r = (r[2], r[0], r[1])
        elif Chem.MolFromSmiles(reactants[2]).HasSubstructMatch(rxn.GetReactants()[0]) \
            and Chem.MolFromSmiles(reactants[1]).HasSubstructMatch(rxn.GetReactants()[1]) \
            and Chem.MolFromSmiles(reactants[0]).HasSubstructMatch(rxn.GetReactants()[2]):
            r = (r[2], r[1], r[0])
        else:
            return "None"
    else:
        raise ValueError("This reaction is neither uni- nor bi-molecular.")

    # Run reaction with rdkit magic
    ps = rxn.RunReactants(r)
    uniqps = list({Chem.MolToSmiles(p) for p in itertools.chain(*ps)})

    if keep_main_product:
        uniqps = uniqps[:1]

    if len(uniqps) == 0:
        return "None"
    uniqps = uniqps[0]

    return uniqps

def routes_from_str(synthesis: str) -> tuple[str, str]:
    """Parse a synthesis string into full reaction routes and final product sequence.

    Inputs:
        synthesis (str): Encoded synthesis route string (e.g., 'A;B;R1;C;R2').

    Outputs:
        tuple[str, str]: (Full route with reactions, product-only sequence).
    """
    full_string = ''
    molecule_only = ''
    action_sequnce = synthesis.split(";")
    stack = []
    for action in action_sequnce:
        if action[0] == 'R':
            rxn_smirks = RXN_LIST[int(action[1:])]
            products = get_products(rxn_smirks, stack[-AllChem.ReactionFromSmarts(rxn_smirks).GetNumReactantTemplates():])
            stack = stack[:-AllChem.ReactionFromSmarts(rxn_smirks).GetNumReactantTemplates()] + [products]
            full_string += action + ':' + products + ';'
            molecule_only += products + '.'
        else:
            stack.append(action)
            full_string += action + ';'
            molecule_only += action + '.'
    return full_string, molecule_only[:-1]

def reactions_from_syn(synthesis: str) -> Union[list[str], None]:
    """Convert a synthesis string into a list of reaction SMILES strings.

    Inputs:
        synthesis (str): Encoded synthesis route string.

    Outputs:
        list[str] | None: List of reaction SMILES or None if parsing fails.
    """
    try:
        reactions = []
        reacs = routes_from_str(synthesis)[0].split(':')
        for i, reac in enumerate(reacs[:-1]):
            reacsmi = '.'.join([
                r for r in reac.split(';')
                if not r.startswith('R')
            ])
            psmi = reacs[i+1].split(';')[0]
            reactions.append(f'{reacsmi}>>{psmi}')
        return reactions
    except:
        return None

def get_visible(arr: np.ndarray) -> np.ndarray:
    """ Return a boolean mask of visible (non-white) pixels in an image array """
    if arr.shape[2] == 4:
        visible = np.any((arr[:, :, :3] > 0) & (arr[:, :, :3] < 250), axis=2) & (arr[:, :, 3] > 0)
    else:
        visible = np.any((arr > 0) & (arr < 250), axis=2)
    return visible     

def get_leftmost_nonwhite_pixel(img: Image.Image) -> int:
    """ Return the x-coordinate of the leftmost non-white (or non-transparent) pixel """
    arr = np.array(img)
    visible = get_visible(arr)
    cols = np.where(np.any(visible, axis=0))[0]
    if len(cols) == 0:
        return 0
    return int(cols[0])

def crop_horizontal_whitespace(img: Image.Image, left_padding: int = 0, right_padding: int = 0) -> Image.Image:
    """ Crop horizontal whitespace from an image with optional padding """
    arr = np.array(img)
    visible = get_visible(arr)
    cols = np.where(np.any(visible, axis=0))[0]
    if len(cols) == 0:
        return img
    left = max(cols[0] - left_padding, 0)
    right = min(cols[-1] + 1 + right_padding, img.width)
    return img.crop((left, 0, right, img.height))


def crop_vertical_whitespace(img: Image.Image, top_padding: int = 0, bottom_padding: int = 0) -> Image.Image:
    """ Crop vertical whitespace from an image with optional padding. """
    arr = np.array(img)
    visible = get_visible(arr)
    rows = np.where(np.any(visible, axis=1))[0]
    if len(rows) == 0:
        return img
    top, bottom = rows[0], rows[-1]+1
    return img.crop((0, max(0,top-top_padding), img.width, bottom+bottom_padding))

def visualize_reactions(reactions: list[str], headers: list[str] = None) -> Union[Image.Image, None]:
    """
    Render and stack 2D reaction images from SMILES with optional headers.

    Inputs:
        reactions (list[str]): Reaction SMILES strings to visualize.
        margin (int): Margin size around each image.
        border_width (int): Width of border lines.
        headers (list[str] | None): Optional list of titles for each reaction.

    Outputs:
        PIL.Image | None: Combined reaction image or None if no input reactions.
    """
    if not reactions:
        return None 
    
    images = []

    for idx, rxnsmi in enumerate(reactions):
        drawer = rdMolDraw2D.MolDraw2DCairo(100,100)
        opts = drawer.drawOptions()
        opts.fixedBondLength = 15
        opts.baseFontSize = 1.1
        opts.bondLineWidth = 1.25
        opts.additionalAtomLabelPadding = 0.1
        opts.centreMoleculesBeforeDrawing = True
        opts.prepareMolsBeforeDrawing = True

        rxn = rdChemReactions.ReactionFromSmarts(rxnsmi, useSmiles=True)
        img = Draw.ReactionToImage(rxn, drawOptions=opts)
        img = crop_vertical_whitespace(img, top_padding=30, bottom_padding=10)
        x_start = get_leftmost_nonwhite_pixel(img)

        # Draw reaction label
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("LiberationSans-Regular.ttf", 12)
         
        if headers: 
            draw.text((x_start, 5), headers[idx], fill=(0, 0, 0), font=font)
        else: 
            draw.text((x_start, 5), f'Reaction {idx+1}', fill=(0, 0, 0), font=font)
        img = crop_horizontal_whitespace(img, left_padding=5, right_padding=5)
        images.append(img)

    # Pad images to the same width
    max_width = max(im.width for im in images)
    padded_images = []
    for im in images:
        if im.width < max_width:
            new_im = Image.new("RGBA", (max_width, im.height), "white")
            new_im.paste(im, (0, 0))
            padded_images.append(new_im)
        else:
            padded_images.append(im)

    padded_images = [im.convert("RGBA") for im in padded_images]

    # Stack reactions within one projection
    images_np = [np.array(im) for im in padded_images]
    merged_np = np.vstack(images_np)
    merged_img = Image.fromarray(merged_np)

    return merged_img