import json
from tqdm import tqdm
from rdkit import Chem


def canonicalize_smiles(smi: str, map_clear=True, cano_with_heavyatom=True) -> str:
    cano_smi = ''
    mol = Chem.MolFromSmiles(smi)

    if mol is None:
        cano_smi = ''
    else:
        if mol.GetNumHeavyAtoms() < 2 and cano_with_heavyatom:
            cano_smi = 'CC'
        elif map_clear:
            for a in mol.GetAtoms():
                a.ClearProp('molAtomMapNumber')
            cano_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        else:
            cano_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
    return cano_smi


def main():
    query = 'O = [N+] ( [O] ) c 1 c c c ( F ) c c 1 C Br'.replace(' ', '')
    query = canonicalize_smiles(query)
    print(f"query: {query}")
    for file in ['./src-test.txt', './src-val.txt', './src-train.txt']:
        with open(file, 'r') as rf:
            smis = rf.readlines()
            for i, smi in tqdm(enumerate(smis)):
                smi = smi.replace('\n', '').replace(' ', '')
                cano_smi = canonicalize_smiles(smi)
                if i == 1:
                    print(f"1st prod: {cano_smi}")
                if query == cano_smi:
                    print(f" In {file}, line: {i}")
                    return

if __name__ == "__main__":
    main()