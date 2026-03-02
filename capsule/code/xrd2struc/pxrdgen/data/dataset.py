import torch
from omegaconf import ValueNode
from torch.utils.data import Dataset
from torch_geometric.data import Data

class XRDDataset(Dataset):
    def __init__(self, save_path: ValueNode):
        self.cached_data = torch.load(save_path)
        
    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]
        (frac_coords, atom_types, lengths, angles, num_atoms) = data_dict['graph_arrays']
        xrd_array = data_dict['xrd_array']
        data = Data(
            id = index,
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            num_atoms=num_atoms,
            num_nodes=num_atoms,
            xrd_array=(torch.Tensor(xrd_array)[:-1]).view(1, -1),
        )
        return data

class XRDDataset_int(Dataset):
    def __init__(self, save_path: ValueNode):
        self.cached_data = torch.load(save_path)
        
    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]
        (frac_coords, atom_types, lengths, angles, num_atoms) = data_dict['graph_arrays']
        xrd_array = data_dict['xrd_array']
        data = Data(
            id = index,
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            num_atoms=num_atoms,
            num_nodes=num_atoms,
            xrd_array=torch.Tensor(xrd_array[:-1]*100).int().view(1, -1),
        )
        return data

