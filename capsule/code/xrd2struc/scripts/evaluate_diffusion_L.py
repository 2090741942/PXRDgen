from pathlib import Path
import argparse
import hydra
from hydra import initialize_config_dir
from torch_geometric.data import Batch
from torch_geometric.data import DataLoader
import time
import warnings
import torch
import sys
sys.path.append('.')
from pxrdgen.model.diffusion import CSPDiffusion
warnings.filterwarnings('ignore')


def load_model(model_path, label=1):
    with initialize_config_dir(str(model_path/'.hydra')):
        cfg = hydra.compose(config_name='config')
        ckpt = list(sorted(model_path.glob('*.ckpt')))
        print(str(ckpt[label]))
        model = CSPDiffusion.load_from_checkpoint(str(ckpt[label]), **cfg.model)
        model.eval()
        # test_dataset = hydra.utils.instantiate(cfg.data.datamodule.datasets.test, _recursive_=False)
        # testloader = DataLoader(test_dataset, shuffle=False, batch_size=256, num_workers=4)
        datamodule = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)
        datamodule.setup()
        testloader = datamodule.test_dataloader()
        return model, testloader

def get_cell_from_pt(pt_file, refine=0):
    a = torch.load(pt_file)
    if refine==0:
        true_lattices = a['true_lattices']
        return true_lattices
    elif refine==1:
        generate_lattices = a['generate_lattices']
        return generate_lattices
    


def lattices_to_params_shape(lattices):

    lengths = torch.sqrt(torch.sum(lattices ** 2, dim=-1))
    angles = torch.zeros_like(lengths)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[...,i] = torch.clamp(torch.sum(lattices[...,j,:] * lattices[...,k,:], dim = -1) /
                            (lengths[...,j] * lengths[...,k]), -1., 1.)
    angles = torch.arccos(angles) * 180.0 / torch.pi

    return lengths, angles


def diffusion(loader, model, given_lattices, num_evals, step_lr = 1e-5):

    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []
    for idx, batch in enumerate(iter(loader)):
        batch.to(model.device)
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lattices = []
        for eval_idx in range(num_evals):

            print(f'batch {idx} / {len(loader)}, sample {eval_idx} / {num_evals}')
            initial_cell = given_lattices[eval_idx, idx*len(batch):(idx+1)*len(batch)]
            initial_cell = initial_cell.to(model.device)
            outputs = model.sample_given_inital_cell(batch, initial_cell, step_lr=step_lr)
            batch_frac_coords.append(outputs['frac_coords'].detach().cpu())
            batch_num_atoms.append(outputs['num_atoms'].detach().cpu())
            batch_atom_types.append(outputs['atom_types'].detach().cpu())
            batch_lattices.append(outputs['lattices'].detach().cpu())

        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lattices.append(torch.stack(batch_lattices, dim=0))

        input_data_list = input_data_list + batch.to_data_list()


    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lattices = torch.cat(lattices, dim=1)
    lengths, angles = lattices_to_params_shape(lattices)
    input_data_batch = Batch.from_data_list(input_data_list)


    return frac_coords, atom_types, lattices, lengths, angles, num_atoms, input_data_batch




def main(args):
    startime = time.time()
    num_eval = args.num_evals
    model_path = Path(args.model_path)
    pt_file = args.pt_path
    refine = args.refine

    ###### label=1 -- last_one.ckpt --ckpt0_?.pt
    diff_model, test_loader = load_model(model_path, label=args.label)
    given_cell = get_cell_from_pt(pt_file, refine=refine)
    (frac_coords, atom_types, lattices, lengths, angles, num_atoms, input_data_batch) = diffusion(test_loader, diff_model, given_cell, num_eval)
    if args.label == -1:
        save_name = 'last_sample%s_refine%s_%s.pt'%(str(num_eval),str(refine), str(args.order))               ######last_one.ckpt
    else:
        save_name = 'best_sample%s_refine%s_%s.pt'%(str(num_eval),str(refine), str(args.order))               ######best_one.ckpt
    
    torch.save({
            'input_data_batch': input_data_batch,
            'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lattices': lattices,
            'lengths': lengths,
            'angles': angles,
        }, model_path / save_name)
    endtime = time.time()
    print('use time %s s.' %(endtime-startime))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--pt_path', required=True)
    parser.add_argument('--refine', default=0, type=int)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--label', default=-1, type=int)
    parser.add_argument('--order', default=0, type=int)
    args = parser.parse_args()
    main(args)
