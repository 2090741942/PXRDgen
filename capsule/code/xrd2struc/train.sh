################################
'''
Remember to change the new path in this template before training:
1. main.py -- PROJECT_ROOT
2. conf/default.yaml -- workpath
3. conf/data/default.yaml -- file_path

Copy the saved xrd_encoder in CL module to the pre_ckpt file. 
'''
################################


work_path='/code/xrd2struc'
cd $work_path


###############
### Section 1 -- xrd_encoder + generative_model
###############
# 1.1 pretrained xrd_encoder

# name='flow_CNN_0.05'
# daytime='2024-06-17'
# python main.py expname=$name model=flow model.encoder_xrd_fix=false model.encoder_xrd._target_=pxrdgen.model.encoder_xrd.xrd_encoder_CNN model.encoder_xrd_ckpt=CL_CNN_0_05.ckpt


# 1.2 xrd_encoder without pretraining

# name='flow_CNN'
# daytime='2024-06-17'
# python main.py expname=$name model=flow model.encoder_xrd_fix=None model.encoder_xrd._target_=pxrdgen.model.encoder_xrd.xrd_encoder_CNN




###############
### Section 2 -- xrd_encoder + generative_model + L
###############
# 2.1 Ltruth

# name='flow_CNN_L'
# daytime='2024-07-10'
# python main.py expname=$name model=flow model.encoder_xrd_fix=false model.encoder_xrd_ckpt=CL_CNN_0_05.ckpt model.encoder_xrd._target_=pxrdgen.model.encoder_xrd.xrd_encoder_CNN model.cost_lattice=0


# 2.2 Lpredict
'''
Before using L_predict, go to the xrd2cell file to train the CellNet first.
'''
# python scripts/evaluate_diffusion_L.py --model_path ${work_path}/outputs/${daytime}/${name} --pt_path /data/outs/xrd2cell/cell_diffusion_CNN/last_sample1_L1000_upper.pt --refine 1 --num_evals 1 --label -1 --order 1
# python scripts/compute.py --pt_path  ${work_path}/outputs/${daytime}/${name}/last_sample1_refine1_1.pt


