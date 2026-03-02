import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
import lightning as L
from tqdm import tqdm
from pxrdgen.model.diff_utils import lattice_params_to_matrix_torch



class BaseModule(L.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}


### Model definition

class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings



class LFlow(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        
        self.decoder = hydra.utils.instantiate(self.hparams.decoder, latent_dim = self.hparams.time_dim, _recursive_=False)
        self.timesteps = self.hparams.timesteps
        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.xrd_encoder = hydra.utils.instantiate(self.hparams.encoder_xrd)
        ckpt_fix = self.hparams.encoder_xrd_fix
        encoder_xrd_ckpt = self.hparams.ckpt_path + self.hparams.encoder_xrd_ckpt
        self.xrd_encoder.load_state_dict(torch.load(encoder_xrd_ckpt))
        if ckpt_fix:
            self.xrd_encoder.eval()
            for para in self.xrd_encoder.parameters():
                para.requires_grad = False


    def uniform_sample_t(self, batch_size, device):
        ts = np.random.choice(np.arange(1, self.timesteps+1), batch_size)
        return torch.from_numpy(ts).to(device)


    def forward(self, batch):

        encoded_xrd = self.xrd_encoder(batch.xrd_array)
        encoded_xrd = encoded_xrd/encoded_xrd.norm(dim=-1, keepdim=True)

        batch_size = batch.num_graphs
        times = self.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)
        c1 = times / self.timesteps
        c0 = 1 - c1

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        rand_l = torch.randn_like(lattices) # x0
        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
        pred_l = self.decoder(time_emb, input_lattice, encoded_xrd)

        loss = F.mse_loss(pred_l, rand_l)
        
        return loss


    @torch.no_grad()
    def sample(self, batch, infer_timesteps=200):
        
        encoded_xrd = self.xrd_encoder(batch.xrd_array)
        encoded_xrd = encoded_xrd/encoded_xrd.norm(dim=-1, keepdim=True)
        batch_size = batch.num_graphs

        l_T = torch.randn([batch_size, 3, 3]).to(self.device)
        
        assert self.timesteps % infer_timesteps == 0

        mult = self.timesteps // infer_timesteps

        time_start = self.timesteps - mult

        traj = {time_start : l_T}


        for t in tqdm(range(time_start, 0, -mult)):

            times = torch.full((batch_size, ), t, device = self.device)
            time_emb = self.time_embedding(times)

            l_t = traj[t]
            pred_l = self.decoder(time_emb, l_t, encoded_xrd)
            step_size = 1. / infer_timesteps
            l_t_minus_1 = l_t - step_size * (pred_l - l_t) / (1. - t / self.timesteps)

            traj[t - mult] = l_t_minus_1 

        tar = traj[0]

        return tar
        

    def training_step(self, batch, batch_idx):
            loss = self(batch)
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss
        
    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
        
    def test_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('test_loss', loss)
        return loss

    