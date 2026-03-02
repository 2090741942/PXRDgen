import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
import lightning as L
from tqdm import tqdm
from pxrdgen.model.diff_utils import lattice_params_to_matrix_torch
import hydra

MAX_ATOMIC_NUM=100

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


class LDiffusion(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.decoder = hydra.utils.instantiate(self.hparams.decoder, _recursive_=False)
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
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
    
    def forward(self, batch):

        encoded_xrd = self.xrd_encoder(batch.xrd_array)
        encoded_xrd = encoded_xrd/encoded_xrd.norm(dim=-1, keepdim=True)
        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)
        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        rand_l = torch.randn_like(lattices)      #(bs, 3, 3)
        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
        pred_l = self.decoder(time_emb, input_lattice, encoded_xrd)

        loss = F.mse_loss(pred_l, rand_l)
        
        return loss

    @torch.no_grad()
    def sample(self, batch):

        encoded_xrd = self.xrd_encoder(batch.xrd_array)
        encoded_xrd = encoded_xrd/encoded_xrd.norm(dim=-1, keepdim=True)
        batch_size = batch.num_graphs

        l_T = torch.randn([batch_size, 3, 3]).to(self.device)
        time_start = self.beta_scheduler.timesteps
        traj = {time_start : l_T}


        for t in tqdm(range(time_start, 0, -1)):

            times = torch.full((batch_size, ), t, device = self.device)
            time_emb = self.time_embedding(times)
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            l_t = traj[t]
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            pred_l = self.decoder(time_emb, l_t, encoded_xrd)
            l_t_minus_1 = c0 * (l_t - c1 * pred_l) + sigmas * rand_l
            traj[t - 1] = l_t_minus_1           

        # traj_stack = { torch.stack([traj[i] for i in range(time_start, -1, -1)])}
        return traj[0]
        # return traj[0], traj_stack

    
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
    

    