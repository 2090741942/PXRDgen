import lightning as L
from omegaconf import ValueNode
import hydra
import torch
import torch.nn.functional as F

class BaseModule(L.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
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

class LightMain(BaseModule):
    def __init__(self, encoder_struc: ValueNode, encoder_xrd: ValueNode, temperature: ValueNode, *args, **kwargs):
        super().__init__()
        self.encoder_xrd = hydra.utils.instantiate(encoder_xrd)
        self.encoder_struc = hydra.utils.instantiate(encoder_struc)
        self.temperature = temperature
        
    def forward(self, batch):
        struc = self.encoder_struc(batch)                                           #(batch_size, latent_size)
        xrd = self.encoder_xrd(batch.xrd_array)                                     #(batch_size, latent_size)
        
        struc_n = struc.norm(dim=-1, keepdim=True)
        xrd_n = xrd.norm(dim=-1, keepdim=True)
        sim_matirx = torch.einsum('ik,jk->ij', xrd/xrd_n, struc/struc_n)
        sim_matirx = sim_matirx / self.temperature
        label = torch.arange(sim_matirx.shape[0]).to(sim_matirx.device).long()
        loss_cl = ( F.cross_entropy(sim_matirx, label) + F.cross_entropy(sim_matirx.T, label) ) / 2
        return loss_cl

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
    
    def on_test_epoch_end(self):
        encoder_state_dict1 = self.encoder_xrd.state_dict()
        # encoder_state_dict2 = self.encoder_struc.state_dict()
        torch.save(encoder_state_dict1, 'encoder_xrd.ckpt')
        # torch.save(encoder_state_dict2, 'encoder_struc.ckpt')

    @torch.no_grad()
    def get_similarity(self, batch):
        struc = self.encoder_struc(batch)                                          #(batch_size, latent_size)
        xrd = self.encoder_xrd(batch.xrd_array)                                    #(batch_size, latent_size)
        struc_n = struc.norm(dim=-1, keepdim=True)
        xrd_n = xrd.norm(dim=-1, keepdim=True)
        sim_matirx = torch.einsum('ik,jk->ij', xrd/xrd_n, struc/struc_n)
        sim_matirx = sim_matirx / self.temperature
        return sim_matirx
    
    @torch.no_grad()
    def dotopk(self, batch, k):
        struc = self.encoder_struc(batch)                                          #(batch_size, latent_size)
        xrd = self.encoder_xrd(batch.xrd_array)                                    #(batch_size, latent_size)
        struc_n = struc.norm(dim=-1, keepdim=True)
        xrd_n = xrd.norm(dim=-1, keepdim=True)
        sim_matirx = torch.einsum('ik,jk->ij', xrd/xrd_n, struc/struc_n)
        sim_matirx = sim_matirx / self.temperature
        _, induces_k = sim_matirx.topk(k, dim=1, largest=True, sorted=True)
        label = torch.arange(sim_matirx.shape[0]).to(sim_matirx.device).long()
        correct_induces = induces_k.eq(label.view(-1, 1))
        correct_predictions = correct_induces.sum().item()
        return correct_predictions
    
    @torch.no_grad()
    def test_topk(self, batch, struc, step, k):  
        xrd = self.encoder_xrd(batch.xrd_array)                                    #(batch_size, latent_size)
        struc_n = struc.norm(dim=-1, keepdim=True)                                 #(len(test_dataset), latent_size)
        xrd_n = xrd.norm(dim=-1, keepdim=True)
        sim_matirx = torch.einsum('ik,jk->ij', xrd/xrd_n, struc/struc_n)           #(batch_size, len(test_dataset))
        sim_matirx = sim_matirx / self.temperature
        _, induces_k = sim_matirx.topk(k, dim=1, largest=True, sorted=True)
        label = torch.arange(step, (sim_matirx.shape[0]+step)).to(sim_matirx.device).long()
        correct_induces = induces_k.eq(label.view(-1, 1))
        correct_predictions = correct_induces.sum().item()
        return correct_predictions, label, induces_k
