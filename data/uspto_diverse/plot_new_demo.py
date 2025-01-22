from rdkit import Chem
from rdkit.Chem import  Draw
from rdkit.Chem.Draw import IPythonConsole #Needed to show molecules
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions #Only needed if modifying defaults

opts =  DrawingOptions()
opts.includeAtomNumbers=True
opts.includeAtomNumbers=True
opts.bondLineWidth=2.8


prod_smi = 'N c 1 c c c ( - c 2 c c c c c 2 ) c n 1'.replace(' ', '')
react_smi_1 = 'Cl c 1 c c c ( - c 2 c c c c c 2 ) c n 1 . O = [N+] ( [O-] ) O'.replace(' ', '')
react_smi_2 = 'N c 1 c c c ( Br ) c n 1 . O B ( O ) c 1 c c c c c 1'.replace(' ', '')

prod_mol = Chem.MolFromSmiles(prod_smi)
img_p = Draw.MolToImage(prod_mol,options=opts)
f_p = 'demo_prod.jpg'
img_p.save(f_p)

react_mol_1 = Chem.MolFromSmiles(react_smi_1)
img_react_1 = Draw.MolToImage(react_mol_1,options=opts)
f_r1 = 'demo_react_1.jpg'
img_react_1.save(f_r1)

react_mol_2 = Chem.MolFromSmiles(react_smi_2)
img_react_2 = Draw.MolToImage(react_mol_2,options=opts)
f_r2 = 'demo_react_2.jpg'
img_react_2.save(f_r2)
