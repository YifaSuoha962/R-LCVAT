import pandas as pd

import torch
import torch.nn.utils.rnn as rnn_utils
from rdkit import Chem, RDLogger
from rdkit.Chem import MolStandardize, AllChem


def canonicalize(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
    except:
        print('no mol', flush=True)
        return smi
    if mol is None:
        return smi
    mol = Chem.RemoveHs(mol)
    [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]
    return Chem.MolToSmiles(mol)


rxn_df = pd.read_csv('raw_test.csv')
rm_rxn_df = rxn_df.copy(deep=True)

for i, row in rm_rxn_df.iterrows():
    rxn = row['reactants>reagents>production'].split('>>')
    react = canonicalize(rxn[0])
    prod = canonicalize(rxn[1])
    rm_rxn_df.at[i, 'reactants>reagents>production'] = react + ">>" + prod

targ_csv = 'rm_am_test.csv'
rm_rxn_df.to_csv(targ_csv, index=False)
