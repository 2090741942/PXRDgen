from pathlib import Path
import argparse
import hydra
from hydra import initialize_config_dir
import warnings
import torch
from torch_geometric.data import Batch
from copy import deepcopy
import numpy as np
import sys
sys.path.append('.')
from pxrdgen.model.diffusion import LDiffusion

# sys.path.insert(0, '/workspace/g2full/GSAS-II/GSASII')
# import GSASIIlattice as G2lat

sys.path.insert(0, '/workspace/g2full/GSAS-II')
from GSASII import GSASIIlattice as G2lat

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
        model = LDiffusion.load_from_checkpoint(str(ckpt[label]), **cfg.model)
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


# def error_func(cell_pre, d_true):
#     d1 = get_d_from_lattice(cell_pre)
#     distance, _ = fastdtw(np.array(d1), np.array(d_true))
#     return distance

def error_func(cell_pre, d_true):
    try:
        d1 = get_d_from_lattice(cell_pre)
        if len(d1) == 0:
            print("empty d1, cell_pre =", cell_pre)
            return float("inf")
        if len(d_true) == 0:
            print("empty d_true")
            return float("inf")
        distance, _ = fastdtw(np.array(d1), np.array(d_true))
        return distance
    except Exception as e:
        print("fastdtw error:", e)
        print("cell_pre:", cell_pre)
        print("len(d_true):", len(d_true))
        return float("inf")

# def diffusion(loader, model, num_evals, num_L_sample):

#     generate_lattices = []
#     true_lattices = []
#     for idx, batch in enumerate(iter(loader)):
#         batch.to(model.device)
#         batch_lattices = []
#         for eval_idx in range(num_evals):
#             print(f'batch {idx} / {len(loader)}, sample {eval_idx} / {num_evals}')
            
#             true_batch_lattice = torch.cat((batch.lengths, batch.angles), dim=-1)   #ture_lattice.shape(bs,6)
#             top_batch_lattice = []
#             num_L_lattice = []
#             for num_L in range(num_L_sample):
#                 outputs = model.sample(batch)                                 #output.shape (bs,3,3)
#                 num_L_lattice.append(outputs)
#             num_L_lattice_tensor0 = torch.stack(num_L_lattice)                #(num_L_sample, bs, 3, 3)
#             num_L_lattice_tensor = get_lengths_angles(num_L_lattice_tensor0)  #(num_L_sample, bs, 6)
#             num_L_lattice_tensor = num_L_lattice_tensor.transpose(0, 1)       #(bs, num_L_sample, 6)

#             for j in tqdm(range(len(num_L_lattice_tensor))):
#                 error=1000000000
#                 induce=0
#                 d_true = get_d_from_lattice(true_batch_lattice[j].cpu().numpy())
#                 for k in range(len(num_L_lattice_tensor[j])):
#                     test_error = error_func(num_L_lattice_tensor[j][k].cpu().numpy(), d_true)
#                     if error >= test_error:
#                         error = test_error
#                         induce = k
#                 top_batch_lattice.append(num_L_lattice_tensor[j][induce].detach().cpu())

#             ### torch.stack(top_batch_lattice, dim=0) shape (bs, 6)
#             batch_lattices.append(torch.stack(top_batch_lattice, dim=0))
        
#         # torch.stack(batch_lattices, dim=0) shape (num_evals, bs, 6)
#         generate_lattices.append(torch.stack(batch_lattices, dim=0))   
#         # true_batch_lattice.shape (bs, 6)
#         true_lattices.append(true_batch_lattice.detach().cpu())
    
#     generate_lattices = torch.cat(generate_lattices, dim=1) # (num_evals, all_num, 6)
#     true_lattices = torch.cat(true_lattices, dim=0)         # (all_num, 6)
    
#     return generate_lattices, true_lattices

# from copy import deepcopy
# import torch
# from torch_geometric.data import Batch
# from tqdm import tqdm


# def _repeat_pyg_batch(batch, repeat_times):
#     """
#     把一个 PyG Batch 中的所有图重复 repeat_times 次，
#     返回一个新的、更大的 Batch。
#     """
#     data_list = batch.to_data_list()
#     big_list = []
#     for _ in range(repeat_times):
#         big_list.extend([deepcopy(data) for data in data_list])
#     return Batch.from_data_list(big_list)

def _repeat_data_list(data_list_cpu, repeat_times):
    big_list = []
    for _ in range(repeat_times):
        big_list.extend([deepcopy(data) for data in data_list_cpu])
    return Batch.from_data_list(big_list)


def diffusion(loader, model, num_evals, num_L_sample, sample_chunk_size=16):
    """
    优化版 diffusion：
    1. 显式把 batch 放到 model.device
    2. 对 num_L_sample 采用分块并行采样，而不是串行一次一次 sample
    3. error_func / get_d_from_lattice 保留在 CPU，但整块搬运，避免碎拷贝

    参数：
    - loader: dataloader
    - model: 生成模型
    - num_evals: 外层评估次数
    - num_L_sample: 每个样本要生成多少个 lattice 候选
    - sample_chunk_size: 每次并行生成多少个候选，按显存调，常见可试 8 / 16 / 32
    """

    device = model.device
    model = model.to(device)
    model.eval()

    generate_lattices = []
    true_lattices = []

    # for idx, batch in enumerate(loader):
    #     # ---------- 优化1：显式放到 GPU ----------
    #     batch = batch.to(device)

    #     # 当前 batch 的真实 lattice: (bs, 6)
    #     true_batch_lattice = torch.cat((batch.lengths, batch.angles), dim=-1)
    #     bs = true_batch_lattice.shape[0]

    #     # ---------- 优化3：真实值一次性搬到 CPU，并预先算好 d_true ----------
    #     true_batch_lattice_cpu = true_batch_lattice.detach().cpu().numpy()
    #     d_true_list = [get_d_from_lattice(true_batch_lattice_cpu[j]) for j in range(bs)]

    #     batch_lattices = []

    #     for eval_idx in range(num_evals):
    #         print(f'batch {idx} / {len(loader)}, sample {eval_idx} / {num_evals}')

    #         candidate_chunks = []
    #         remaining = num_L_sample

    #         # ---------- 优化2：分块并行生成多个候选 ----------
    #         with torch.inference_mode():
    #             while remaining > 0:
    #                 cur_chunk = min(sample_chunk_size, remaining)

    #                 # 把当前 batch 重复 cur_chunk 份，形成一个大 batch
    #                 big_batch = _repeat_pyg_batch(batch, cur_chunk).to(device)

    #                 # 一次 sample，对应生成 cur_chunk 组候选
    #                 # 输出形状: (cur_chunk * bs, 3, 3)
    #                 outputs = model.sample(big_batch)

    #                 # reshape -> (cur_chunk, bs, 3, 3)
    #                 outputs = outputs.view(cur_chunk, bs, 3, 3)

    #                 candidate_chunks.append(outputs)
    #                 remaining -= cur_chunk

    #         # 拼起来 -> (num_L_sample, bs, 3, 3)
    #         num_L_lattice_tensor0 = torch.cat(candidate_chunks, dim=0)
    for idx, batch in enumerate(loader):
        # batch 此时默认还在 CPU
        data_list_cpu = batch.to_data_list()

        # 当前 batch 的 GPU 版，只用于真实 lattice 和需要 GPU 的地方
        batch_gpu = Batch.from_data_list(data_list_cpu).to(device)

        true_batch_lattice = torch.cat((batch_gpu.lengths, batch_gpu.angles), dim=-1)
        bs = true_batch_lattice.shape[0]

        true_batch_lattice_cpu = true_batch_lattice.detach().cpu().numpy()
        d_true_list = [get_d_from_lattice(true_batch_lattice_cpu[j]) for j in range(bs)]

        batch_lattices = []

        for eval_idx in range(num_evals):
            candidate_chunks = []
            remaining = num_L_sample

            with torch.inference_mode():
                while remaining > 0:
                    cur_chunk = min(sample_chunk_size, remaining)

                    big_batch = _repeat_data_list(data_list_cpu, cur_chunk).to(device)
                    outputs = model.sample(big_batch)
                    outputs = outputs.view(cur_chunk, bs, 3, 3)

                    candidate_chunks.append(outputs.detach())

                    del big_batch, outputs
                    remaining -= cur_chunk

            num_L_lattice_tensor0 = torch.cat(candidate_chunks, dim=0)

            # 转成 (num_L_sample, bs, 6)
            num_L_lattice_tensor = get_lengths_angles(num_L_lattice_tensor0)

            # 转成 (bs, num_L_sample, 6)
            num_L_lattice_tensor = num_L_lattice_tensor.transpose(0, 1)

            # ---------- 优化3：整块搬到 CPU ----------
            num_L_lattice_cpu = num_L_lattice_tensor.detach().cpu().numpy()

            top_batch_lattice = []

            for j in tqdm(range(bs)):
                d_true = d_true_list[j]
                best_error = float("inf")
                best_idx = 0

                for k in range(num_L_sample):
                    test_error = error_func(num_L_lattice_cpu[j, k], d_true)
                    if test_error <= best_error:
                        best_error = test_error
                        best_idx = k

                top_batch_lattice.append(
                    torch.from_numpy(num_L_lattice_cpu[j, best_idx]).float()
                )

            # (bs, 6)
            batch_lattices.append(torch.stack(top_batch_lattice, dim=0))

        # (num_evals, bs, 6)
        generate_lattices.append(torch.stack(batch_lattices, dim=0))

        # (bs, 6)
        true_lattices.append(true_batch_lattice.detach().cpu())

    # (num_evals, all_num, 6)
    generate_lattices = torch.cat(generate_lattices, dim=1)

    # (all_num, 6)
    true_lattices = torch.cat(true_lattices, dim=0)

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
    generate_lattices, true_lattices = diffusion(test_loader, diff_model, num_evals, num_L_sample)   # generate_lattices (num_evals, all_num, 6);  true_lattices (all_num, 6)
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
