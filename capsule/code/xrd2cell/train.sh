################################
# '''
# Remember to change the new path in this template before training:
# 1. main.py -- PROJECT_ROOT
# 2. conf/default.yaml -- workpath
# 3. conf/data/default.yaml -- file_path
# '''
################################


work_path='/workspace/PXRDgen/capsule/code/xrd2cell'
cd $work_path

# name='cell_diffusion_CNN_1_00'
# daytime='2026-3-16'

# python main.py expname=${name} model=diffusion model.encoder_xrd._target_=pxrdgen.model.encoder_xrd.xrd_encoder_CNN model.encoder_xrd_fix=false model.encoder_xrd_ckpt=CL_CNN_1_00_train.ckpt

# name='cell_diffusion_CNN_0_05'
# python main.py expname=${name} model=diffusion model.encoder_xrd._target_=pxrdgen.model.encoder_xrd.xrd_encoder_CNN model.encoder_xrd_fix=false model.encoder_xrd_ckpt=CL_CNN_0_05_train.ckpt


################
# '''
# Before evaluating L, remeber to change the path of GSASII in python files.
# sys.path.insert(0, '/g2full/GSAS-II/GSASII') --> sys.path.insert(0, Your GSASII path)
# '''
################


# ## evaluate L by comparing with Ltruth directly
model_path='/workspace/PXRDgen/capsule/code/xrd2cell/outputs/2026-03-16/cell_diffusion_CNN_0_05'

# python /workspace/PXRDgen/capsule/code/xrd2cell/scripts/evaluate_diffusion_upper.py --model_path ${model_path} --num_evals 1 --num_L_sample 1000 --label -1
# python scripts/compute.py --pt_path {pt_path}
# python /workspace/PXRDgen/capsule/code/xrd2cell/scripts/evaluate_diffusion_upper.py --model_path ${model_path} --num_evals 20 --num_L_sample 1000 --label -1
# python scripts/compute.py --pt_path {pt_path}


# ## evaluate L by comparing with fastftd

python /workspace/PXRDgen/capsule/code/xrd2cell/scripts/evaluate_diffusion_fastdtw.py --model_path ${model_path} --num_evals 1 --num_L_sample 1000 --label -1 
# python scripts/compute.py --pt_path {pt_path}
# python /workspace/PXRDgen/capsule/code/xrd2cell/scripts/evaluate_diffusion_fastdtw.py --model_path ${model_path} --num_evals 20 --num_L_sample 1000 --label -1
# python scripts/compute.py --pt_path {pt_path}

