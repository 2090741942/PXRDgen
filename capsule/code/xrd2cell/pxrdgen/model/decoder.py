import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, latent_dim=256, time_dim=256, hidden_dim1=512, hidden_dim2=256, fc_num_layers=5):
        super().__init__()
        mods = [nn.Linear(latent_dim+time_dim+9, hidden_dim1), nn.ReLU()]
        mods += [nn.Linear(hidden_dim1, hidden_dim2), nn.ReLU()]
        for i in range(fc_num_layers):
            mods += [nn.Linear(hidden_dim2, hidden_dim2), nn.ReLU()]
        mods += [nn.Linear(hidden_dim2, 9)]
        self.mlp = nn.Sequential(*mods)

    def forward(self, t, lattices, encoded_xrd):
        # t (bs, time_dim), lattices (bs, 3, 3) encoded_xrd (bs, latent_szie)
        lattice_ips = lattices @ lattices.transpose(-1,-2)
        lattice_ips_flatten = lattice_ips.view(-1, 9)
        input = torch.cat((t, lattice_ips_flatten, encoded_xrd), dim=1)   
        lattice_out = self.mlp(input)
        lattice_out = lattice_out.view(-1, 3, 3)
        lattice_out = torch.einsum('bij,bjk->bik', lattice_out, lattices)
        
        return lattice_out



class MLPUnet(nn.Module):
    def __init__(self, latent_dim=256, time_dim=256, hidden_dim1=64, hidden_dim2=128, hidden_dim3=256, hidden_dim4=512):
        super().__init__()
       
        self.encoder1 = nn.Sequential(nn.Linear(latent_dim+time_dim+9, hidden_dim1), nn.ReLU())
        self.encoder2 = nn.Sequential(nn.Linear(hidden_dim1, hidden_dim2), nn.ReLU())
        self.encoder3 = nn.Sequential(nn.Linear(hidden_dim2, hidden_dim3), nn.ReLU())
        
        self.bottleneck = nn.Sequential(nn.Linear(hidden_dim3, hidden_dim4), nn.ReLU())
        
        self.decoder3 = nn.Sequential(nn.Linear(hidden_dim4+hidden_dim3, hidden_dim3), nn.ReLU())
        self.decoder2 = nn.Sequential(nn.Linear(hidden_dim3+hidden_dim2, hidden_dim2), nn.ReLU())
        self.decoder1 = nn.Sequential(nn.Linear(hidden_dim2+hidden_dim1, hidden_dim1), nn.ReLU())
        
        self.final = nn.Sequential(nn.Linear(hidden_dim1, 9))

    def forward(self, t, lattices, encoded_xrd):
        # t (bs, time_dim), lattices (bs, 3, 3) encoded_xrd (bs, latent_szie)
        lattice_ips = lattices @ lattices.transpose(-1,-2)
        lattice_ips_flatten = lattice_ips.view(-1, 9)
        input = torch.cat((t, lattice_ips_flatten, encoded_xrd), dim=1)

        enc1 = self.encoder1(input)                   #(bs, hidden_dim1)
        enc2 = self.encoder2(enc1)                    #(bs, hidden_dim2)
        enc3 = self.encoder3(enc2)                    #(bs, hidden_dim3)
        bottleneck = self.bottleneck(enc3)            #(bs, hidden_dim4)
        dec3 = torch.cat((bottleneck, enc3), dim=-1)
        dec3 = self.decoder3(dec3)                    #(bs, hidden_dim3)
        dec2 = torch.cat((dec3, enc2), dim=-1)
        dec2 = self.decoder2(dec2)                    #(bs, hidden_dim2)
        dec1 = torch.cat((dec2, enc1), dim=1)
        dec1 = self.decoder1(dec1)                    #(bs, hidden_dim1)
        
        out = self.final(dec1)
        lattice_out = out.view(-1, 3, 3)
        lattice_out = torch.einsum('bij,bjk->bik', lattice_out, lattices)
        
        return lattice_out