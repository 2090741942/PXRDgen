import argparse
import torch
from tqdm import tqdm
import sys
sys.path.insert(0, '/g2full/GSAS-II/GSASII')
import GSASIIlattice as G2lat


def main(args):
    
    print(args.pt_path)
    loadfile = torch.load(args.pt_path)
    true_lattices = loadfile['true_lattices']         #(all_num, 6)
    generate_lattices = loadfile['generate_lattices'] #(num_eval, all_num, 6)
    
    def calc_volume(cell):
        A = G2lat.cell2A(cell)
        volume = G2lat.calc_V(A)
        return volume

    for j in tqdm(range(len(generate_lattices))):
        print('num_eval %s:' %j)
        true_list = []
        diff1_list = []
        for k in range(len(true_lattices)):
            cell0=true_lattices[k]
            cell1=generate_lattices[j][k]
            volume1 = abs(calc_volume(cell1) - calc_volume(cell0))
            true_cell = torch.tensor(list(cell0)+[calc_volume(cell0)], dtype=torch.float32)
            diff1 = torch.tensor([abs(cell1[p]-cell0[p]) for p in range(len(cell0))] + [volume1], dtype=torch.float32)
            true_list.append(true_cell)
            diff1_list.append(diff1)

        true_tensor = torch.stack(true_list)
        diff1_tensor = torch.stack(diff1_list)
        
        nan_induces = torch.isnan(diff1_tensor)
        valid_induces = ~nan_induces
        true_tensor = torch.masked_select(true_tensor, valid_induces).view(-1,7)
        
        diff1_tensor = torch.masked_select(diff1_tensor, valid_induces).view(-1,7)
        print('number of Nan:')
        print(len(true_lattices)-len(true_tensor))
        print('the MAPE of generated cell')
        print(torch.mean(diff1_tensor/true_tensor, dim=0)*100)
        print('the RMSE of generated cell')
        print(torch.mean(diff1_tensor**2, dim=0)**0.5)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_path', required=True)
    args = parser.parse_args()
    main(args)