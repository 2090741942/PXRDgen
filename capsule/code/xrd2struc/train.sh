################################
# '''
# Remember to change the new path in this template before training:
# 1. main.py -- PROJECT_ROOT
# 2. conf/default.yaml -- workpath
# 3. conf/data/default.yaml -- file_path

# Copy the saved xrd_encoder in CL module to the pre_ckpt file. 
# '''
################################


work_path='/workspace/PXRDgen/capsule/code/xrd2struc'
cd $work_path


###############
### Section 1 -- xrd_encoder + generative_model
###############
# 1.1 pretrained xrd_encoder

# name='flow_CNN_0.05'
# daytime='2026-3-16'
# python main.py expname=$name model=flow model.encoder_xrd_fix=false model.encoder_xrd._target_=pxrdgen.model.encoder_xrd.xrd_encoder_CNN model.encoder_xrd_ckpt=CL_CNN_0_05_train.ckpt


# 1.2 xrd_encoder without pretraining

# name='flow_CNN'
# daytime='2026-3-16'
# python main.py expname=$name model=flow model.encoder_xrd_fix=None model.encoder_xrd._target_=pxrdgen.model.encoder_xrd.xrd_encoder_CNN




###############
### Section 2 -- xrd_encoder + generative_model + L
###############
# 2.1 Ltruth

# name='flow_CNN_L'
# daytime='2026-3-16'
# python main.py expname=$name model=flow model.encoder_xrd_fix=false model.encoder_xrd_ckpt=CL_CNN_0_05_train.ckpt model.encoder_xrd._target_=pxrdgen.model.encoder_xrd.xrd_encoder_CNN model.cost_lattice=0


# 2.2 Lpredict
# '''
# Before using L_predict, go to the xrd2cell file to train the CellNet first.
# '''
# python scripts/evaluate_diffusion_L.py --model_path ${work_path}/outputs/${daytime}/${name} --pt_path /data/outs/xrd2cell/cell_diffusion_CNN/last_sample1_L1000_upper.pt --refine 1 --num_evals 1 --label -1 --order 1
# python scripts/compute.py --pt_path  ${work_path}/outputs/${daytime}/${name}/last_sample1_refine1_1.pt

# python /workspace/PXRDgen/capsule/code/xrd2struc/scripts/evaluate_flow.py --model_path /workspace/PXRDgen/capsule/code/xrd2struc/outputs/2026-03-17/flow_CNN --num_evals 1 --label -1 &
# python /workspace/PXRDgen/capsule/code/xrd2struc/scripts/evaluate_flow.py --model_path /workspace/PXRDgen/capsule/code/xrd2struc/outputs/2026-03-17/flow_CNN --num_evals 20 --label -1 &

# wait

# python /workspace/PXRDgen/capsule/code/xrd2struc/scripts/evaluate_flow.py --model_path /workspace/PXRDgen/capsule/code/xrd2struc/outputs/2026-03-17/flow_CNN_0.05 --num_evals 1 --label -1 &
# python /workspace/PXRDgen/capsule/code/xrd2struc/scripts/evaluate_flow.py --model_path /workspace/PXRDgen/capsule/code/xrd2struc/outputs/2026-03-17/flow_CNN_0.05 --num_evals 20 --label -1 &

# wait

# python /workspace/PXRDgen/capsule/code/xrd2struc/scripts/evaluate_flow_L.py --model_path /workspace/PXRDgen/capsule/code/xrd2struc/outputs/2026-03-17/flow_CNN_L --pt_path /workspace/PXRDgen/capsule/code/xrd2cell/outputs/2026-03-16/cell_diffusion_CNN_0_05/last_sample1_L1000_upper.pt  --num_evals 1 --label -1 --refine 0 
# python /workspace/PXRDgen/capsule/code/xrd2struc/scripts/evaluate_flow_L.py --model_path /workspace/PXRDgen/capsule/code/xrd2struc/outputs/2026-03-17/flow_CNN_L --pt_path /workspace/PXRDgen/capsule/code/xrd2cell/outputs/2026-03-16/cell_diffusion_CNN_0_05/last_sample1_L1000_upper.pt  --num_evals 1 --label -1 --refine 1 &
# python /workspace/PXRDgen/capsule/code/xrd2struc/scripts/evaluate_flow_L.py --model_path /workspace/PXRDgen/capsule/code/xrd2struc/outputs/2026-03-17/flow_CNN_L --num_evals 20 --label -1 --refine 0

# python /workspace/PXRDgen/capsule/code/xrd2struc/scripts/evaluate_flow_L.py --model_path /workspace/PXRDgen/capsule/code/xrd2struc/outputs/2026-03-17/flow_CNN_L --pt_path /workspace/PXRDgen/capsule/code/xrd2cell/outputs/2026-03-16/cell_diffusion_CNN_0_05/last_sample20_L1000_upper.pt  --num_evals 20 --label -1 --refine 0 
python /workspace/PXRDgen/capsule/code/xrd2struc/scripts/evaluate_flow_L.py --model_path /workspace/PXRDgen/capsule/code/xrd2struc/outputs/2026-03-17/flow_CNN_L --pt_path /workspace/PXRDgen/capsule/code/xrd2cell/outputs/2026-03-16/cell_diffusion_CNN_0_05/last_sample20_L1000_upper.pt  --num_evals 20 --label -1 --refine 1 

# 计算不同encoder，不同L下得到的生成结果
# python scripts/compute.py --pt_path /workspace/PXRDgen/capsule/code/xrd2struc/outputs/2026-03-17/flow_CNN/last_sample1_0.pt >> zout_train.txt
# python scripts/compute.py --pt_path /workspace/PXRDgen/capsule/code/xrd2struc/outputs/2026-03-17/flow_CNN/last_sample20_0.pt --multi_eval >> zout_train.txt
# python scripts/compute.py --pt_path /workspace/PXRDgen/capsule/code/xrd2struc/outputs/2026-03-17/flow_CNN_0.05/last_sample1_0.pt >> zout_train.txt
# python scripts/compute.py --pt_path /workspace/PXRDgen/capsule/code/xrd2struc/outputs/2026-03-17/flow_CNN_0.05/last_sample20_0.pt --multi_eval >> zout_train.txt
# python scripts/compute.py --pt_path /workspace/PXRDgen/capsule/code/xrd2struc/outputs/2026-03-17/flow_CNN_L/last_sample1_refine0_0_upper.pt >> zout_train.txt
# python scripts/compute.py --pt_path /workspace/PXRDgen/capsule/code/xrd2struc/outputs/2026-03-17/flow_CNN_L/last_sample1_refine1_0_upper.pt >> zout_train.txt

# python scripts/compute.py --pt_path /workspace/PXRDgen/capsule/code/xrd2struc/outputs/2026-03-17/flow_CNN_L/last_sample1_refine0_0_upper.pt --multi_eval >> zout_train.txt
# python scripts/compute.py --pt_path /workspace/PXRDgen/capsule/code/xrd2struc/outputs/2026-03-17/flow_CNN_L/last_sample1_refine1_0_upper.pt --multi_eval >> zout_train.txt





