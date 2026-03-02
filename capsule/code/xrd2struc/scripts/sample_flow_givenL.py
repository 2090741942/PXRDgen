from pathlib import Path
import argparse
import hydra
from hydra import initialize_config_dir
import warnings
import numpy as np
import torch
import os
import sys
sys.path.append('.')
from pxrdgen.model.flow_shift_align import CSPFlow
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader
from eval_utils import chemical_symbols, lattices_to_params_shape, get_crystals_list
import chemparse
import time
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.cif import CifWriter
from p_tqdm import p_map
warnings.filterwarnings('ignore')




class SampleDataset(Dataset):

    def __init__(self, formula, xrd_path, num_evals):
        super().__init__()
        self.formula = formula
        self.xrd = np.loadtxt(xrd_path)
        assert len(self.xrd) == 7500
        self.num_evals = num_evals
        self.get_structure()

    def get_structure(self):
        self.composition = chemparse.parse_formula(self.formula)
        chem_list = []
        for elem in self.composition:
            num_int = int(self.composition[elem])
            chem_list.extend([chemical_symbols.index(elem)] * num_int)
        self.chem_list = chem_list

    def __len__(self) -> int:
        return self.num_evals

    def __getitem__(self, index):
        return Data(
            atom_types=torch.LongTensor(self.chem_list),
            xrd_array=torch.Tensor(self.xrd).view(1, -1),
            num_atoms=len(self.chem_list),
            num_nodes=len(self.chem_list),
        )

def load_model(model_path, label=-1):
    with initialize_config_dir(str(model_path/'.hydra')):
        
        cfg = hydra.compose(config_name='config')
        ckpt = list(sorted(model_path.glob('*.ckpt')))
        print(str(ckpt[label]))
        model = CSPFlow.load_from_checkpoint(str(ckpt[label]), **cfg.model)
        model.eval()

        return model
        


def flow(loader, model, cell, step_lr = 5, infer_timesteps=200):

    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []
    for idx, batch in enumerate(loader):

        if torch.cuda.is_available():
            batch.cuda()
        cell = cell.to(model.device)
        outputs = model.sample_given_inital_cell(batch, cell, step_lr = step_lr, infer_timesteps=infer_timesteps)
        frac_coords.append(outputs['frac_coords'].detach().cpu())
        num_atoms.append(outputs['num_atoms'].detach().cpu())
        atom_types.append(outputs['atom_types'].detach().cpu())
        lattices.append(outputs['lattices'].detach().cpu())

    frac_coords = torch.cat(frac_coords, dim=0)
    num_atoms = torch.cat(num_atoms, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    lattices = torch.cat(lattices, dim=0)
    lengths, angles = lattices_to_params_shape(lattices)

    return (
        frac_coords, atom_types, lattices, lengths, angles, num_atoms
    )


def get_pymatgen(crystal_array):
    frac_coords = crystal_array['frac_coords']
    atom_types = crystal_array['atom_types']
    lengths = crystal_array['lengths']
    angles = crystal_array['angles']
    try:
        structure = Structure(
            lattice=Lattice.from_parameters(
                *(lengths.tolist() + angles.tolist())),
            species=atom_types, coords=frac_coords, coords_are_cartesian=False)
        return structure
    except:
        return None

def main(args):
    startime = time.time()
    model_path = Path(args.model_path)
    
    diff_model = load_model(model_path, label=args.label)
    tar_dir = os.path.join(args.save_path, args.formula)

    os.makedirs(tar_dir, exist_ok=True)
    test_set = SampleDataset(args.formula, args.xrd_path, args.num_evals)
    test_loader = DataLoader(test_set, batch_size = min(256, args.num_evals))
    cell = np.array(list(map(float, args.cell)))
    cell = np.tile(cell, (min(256, args.num_evals),1))
    cell = torch.tensor(cell, dtype=torch.float32)
    (frac_coords, atom_types, lattices, lengths, angles, num_atoms) = flow(test_loader, diff_model, cell)

    crystal_list = get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms)
    strcuture_list = p_map(get_pymatgen, crystal_list)

    for i,structure in enumerate(strcuture_list):
        tar_file = os.path.join(tar_dir, f"{args.formula}_{i+1}.cif")
        if structure is not None:
            writer = CifWriter(structure)
            writer.write_file(tar_file)
        else:
            print(f"{i+1} Error Structure.")

    if os.path.exists('/code/xrd2struc/samples/%s_target.cif'%args.formula):
        print('The target file exists, try to compare the predicted files with the target file:')
        s1 = Structure.from_file('/code/xrd2struc/samples/%s_target.cif'%args.formula)
        predict_files = '/results/%s'%args.formula
        s2_list = os.listdir(predict_files)
        for s in s2_list:
            try:
                s2 = Structure.from_file(os.path.join(predict_files, s))
                matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
                rms_dist = matcher.get_rms_dist(s1, s2)
                rms_dist = None if rms_dist is None else rms_dist[0]
                print(rms_dist)
            except:
                print(None)

    endtime = time.time()
    print('use time %s s.' %(endtime-startime))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--xrd_path', required=True)
    parser.add_argument('--cell', nargs='+', required=True)
    parser.add_argument('--formula', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--label', default=-1, type=int)
    args = parser.parse_args()
    main(args)
