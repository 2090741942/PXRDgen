from pathlib import Path
import argparse
import hydra
from hydra import initialize_config_dir
import warnings
import torch
import numpy as np
import sys
sys.path.append('.')
from pxrdgen.model.flow_shift_align import LFlow
sys.path.insert(0, '/g2full/GSAS-II/GSASII')
import GSASIIlattice as G2lat
from fastdtw import fastdtw
from tqdm import tqdm
from torch_geometric.data import DataLoader
import time
warnings.filterwarnings('ignore')


def load_model(model_path, load_data = False, label=1):
    with initialize_config_dir(str(model_path/'.hydra')):
        
        cfg = hydra.compose(config_name='config')
        ckpt = list(sorted(model_path.glob('*.ckpt')))
        print(str(ckpt[label]))
        model = LFlow.load_from_checkpoint(str(ckpt[label]), **cfg.model)
        model.eval()

        if load_data:
            # test_dataset = hydra.utils.instantiate(cfg.data.datamodule.datasets.test, _recursive_=False)
            # testloader = DataLoader(test_dataset, shuffle=False, batch_size=64, num_workers=4)
            datamodule = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)
            datamodule.setup(stage='test')
            testloader = datamodule.test_dataloader()
            return model, testloader
        
        else:
            return model


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


def get_d_from_lattice(cell_predict):
    A = G2lat.cell2A(cell_predict)
    TTmax = 80 #2theta
    dmin = 1.54056 / (2.0 * np.sin(np.pi * TTmax / 360))
    HKL = G2lat.GenHBravais(dmin, 17, A)
    d_predict = []  
    for r in HKL:
        d_predict.append(r[3])
    return sorted(d_predict, reverse=True)


def error_func(cell_pre, d_true):
    d1 = get_d_from_lattice(cell_pre)
    distance, _ = fastdtw(np.array(d1), np.array(d_true))
    return distance


def flow(loader, model, num_evals, num_L_sample):

    generate_lattices = []
    true_lattices = []
    for idx, batch in enumerate(iter(loader)):
        batch.to(model.device)
        batch_lattices = []
        for eval_idx in range(num_evals):
            print(f'batch {idx} / {len(loader)}, sample {eval_idx} / {num_evals}')
            
            true_batch_lattice = torch.cat((batch.lengths, batch.angles), dim=-1)   #ture_lattice.shape(bs,6)
            top_batch_lattice = []
            num_L_lattice = []
            for num_L in range(num_L_sample):
                outputs = model.sample(batch, infer_timesteps=200)            #output.shape (bs,3,3)
                num_L_lattice.append(outputs)
            num_L_lattice_tensor0 = torch.stack(num_L_lattice)                #(num_L_sample, bs, 3, 3)
            num_L_lattice_tensor = get_lengths_angles(num_L_lattice_tensor0)  #(num_L_sample, bs, 6)
            num_L_lattice_tensor = num_L_lattice_tensor.transpose(0, 1)       #(bs, num_L_sample, 6)

            for j in tqdm(range(len(num_L_lattice_tensor))):
                error=1000000000
                induce=0
                d_true = get_d_from_lattice(true_batch_lattice[j].cpu().numpy())
                for k in range(len(num_L_lattice_tensor[j])):
                    test_error = error_func(num_L_lattice_tensor[j][k].cpu().numpy(), d_true)
                    if error >= test_error:
                        error = test_error
                        induce = k
                top_batch_lattice.append(num_L_lattice_tensor[j][induce].detach().cpu())

            ### torch.stack(top_batch_lattice, dim=0) shape (bs, 6)
            batch_lattices.append(torch.stack(top_batch_lattice, dim=0))
        
        # torch.stack(batch_lattices, dim=0) shape (num_evals, bs, 6)
        generate_lattices.append(torch.stack(batch_lattices, dim=0))   
        # true_batch_lattice.shape (bs, 6)
        true_lattices.append(true_batch_lattice.detach().cpu())
    
    generate_lattices = torch.cat(generate_lattices, dim=1) # (num_evals, all_num, 6)
    true_lattices = torch.cat(true_lattices, dim=0)         # (all_num, 6)
    
    return generate_lattices, true_lattices


def get_lengths_angles(lattices):
    lengths, angles = lattices_to_params_shape(lattices)
    out = torch.cat((lengths, angles), dim=-1)
    return out


def main(args):
    startime = time.time()
    num_evals = args.num_evals
    num_L_sample = args.num_L_sample
    model_path = Path(args.model_path)
    if args.label == -1:
        save_name = 'last_sample%s_L%s_fastdtw.pt'%(str(num_evals), str(num_L_sample))     ######last_one.ckpt
    else:
        save_name = 'best_sample%s_L%s_fastdtw.pt'%(str(num_evals), str(num_L_sample))     ######best_one.ckpt
    
    diff_model, test_loader = load_model(model_path, load_data=True, label=args.label)
    generate_lattices, true_lattices = flow(test_loader, diff_model, num_evals, num_L_sample)   # generate_lattices (num_evals, all_num, 6);  true_lattices (all_num, 6)
    torch.save({
            'generate_lattices': generate_lattices,
            'true_lattices': true_lattices,
        }, model_path / save_name)
    print(generate_lattices.shape)
    print(true_lattices.shape)
    endtime = time.time()
    print('use time %s s.' %(endtime-startime))

###########

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--num_L_sample', default=1, type=int)
    parser.add_argument('--label', default=-1, type=int)
    args = parser.parse_args()
    main(args)
