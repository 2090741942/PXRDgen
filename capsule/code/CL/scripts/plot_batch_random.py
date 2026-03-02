import hydra
from hydra import initialize_config_dir
import warnings
from torch_geometric.data import DataLoader
import sys
sys.path.append('.')
from app.model.LightMain import LightMain
import torch
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')



def load_model(model_path, bs, label=-1):
    with initialize_config_dir(str(model_path/'.hydra')):
        
        cfg = hydra.compose(config_name='config')
        ckpt = list(sorted(model_path.glob('*.ckpt')))
        print(str(ckpt[label]))
        print('batch size is %s.' %str(bs))
        model = LightMain.load_from_checkpoint(str(ckpt[label]), **cfg.model)
        model.eval()

        test_dataset = hydra.utils.instantiate(cfg.data.datamodule.datasets.test, _recursive_=False)
        testloader = DataLoader(test_dataset, shuffle=True, batch_size=bs, num_workers=4)
        return model, testloader



@torch.no_grad()
def get_plot(testloader, model, model_path, label):
    ckpt = list(sorted(model_path.glob('*.ckpt')))
    for batch_demo in iter(testloader):
        batch_demo = batch_demo.to(model.device)
        similarity = model.get_similarity(batch_demo)
        similarity = similarity.cpu().numpy()
        plt.imshow(similarity, cmap='hot_r', interpolation='nearest')
        plt.colorbar()  # 添加颜色条
        plt.title(str(ckpt[label]).split('/')[-1])  # 设置标题
        plt.xlabel('CXRD')                           # 设置X轴标签
        plt.ylabel('Crystal')                       # 设置Y轴标签
        name = str(model_path).split('/')[-1]
        plt.savefig('/results/'+str(name)+'.png')
        break


def main(args):
    
    model_path = Path(args.model_path)
    label = args.label

    diff_model, test_loader = load_model(model_path, 100, label=label)
    get_plot(test_loader, diff_model, model_path, label)
    
    print('----')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--label', default=-1, type=int)
    args = parser.parse_args()
    main(args)

    
    

