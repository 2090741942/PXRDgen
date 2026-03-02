import lightning as L
import argparse
from pathlib import Path
import hydra
from hydra import initialize_config_dir
import warnings
import torch
from torch_geometric.data import DataLoader
import sys
sys.path.append('.')
from app.model.LightMain import LightMain
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
        testloader = DataLoader(test_dataset, shuffle=False, batch_size=bs, num_workers=4)
        return model, testloader
        

@torch.no_grad()
def test_batch_accuracy(testloader, model):
    
    correct_top1, correct_top3, correct_top5, correct_top10 = 0, 0, 0, 0
    num_all = 0
    
    for batch_demo in iter(testloader):
        
        batch_demo = batch_demo.to(model.device)
        num_all += len(batch_demo)

        top1 = model.dotopk(batch_demo, 1)
        correct_top1 += top1
        
        top3 = model.dotopk(batch_demo, 3)
        correct_top3 += top3
        
        top5 = model.dotopk(batch_demo, 5)
        correct_top5 += top5

        top10 = model.dotopk(batch_demo, 10)
        correct_top10 += top10

    print('num_all: ' + str(num_all))
    print('top1 is : ' + str(round(100*correct_top1/num_all, 2)) + '%.')
    print('top3 is : ' + str(round(100*correct_top3/num_all, 2)) + '%.')
    print('top5 is : ' + str(round(100*correct_top5/num_all, 2)) + '%.')
    print('top10 is : ' + str(round(100*correct_top10/num_all, 2)) + '%.')



def main(args):
    
    model_path = Path(args.model_path)
    label = args.label

    diff_model1, test_loader1 = load_model(model_path, 256, label=label)
    test_batch_accuracy(test_loader1, diff_model1)
    diff_model2, test_loader2 = load_model(model_path, 64, label=label)
    test_batch_accuracy(test_loader2, diff_model2)

    print('----')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--label', default=-1, type=int)
    args = parser.parse_args()
    main(args)


