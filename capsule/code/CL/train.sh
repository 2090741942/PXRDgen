################################
'''
Remember to change the new path in this template before training:
1. main.py -- PROJECT_ROOT
2. conf/default.yaml -- workpath
3. conf/data/default.yaml -- file_path
'''
################################

work_path='/workspace/PXRDgen/capsule/code/CL'
cd $work_path

daytime='2026-03-16'
# name='CL_CNN_t_0_05'
# python main.py expname=$name model.temperature=0.05 model.encoder_xrd._target_=app.model.encoder_xrd.xrd_encoder_CNN optim.optimizer.weight_decay=0 optim.optimizer.lr=1e-3 optim.lr_scheduler.eta_min=1e-7
# name='CL_CNN_t_1_00'
# python main.py expname=$name model.temperature=1.0 model.encoder_xrd._target_=app.model.encoder_xrd.xrd_encoder_CNN optim.optimizer.weight_decay=1e-4 optim.optimizer.lr=1e-3


# daytime='2024-06-26'
name='CL_T_t_1_00'
python main.py expname=$name model.temperature=1.0 model.encoder_xrd._target_=app.model.encoder_xrd.xrd_encoder_T optim.optimizer.weight_decay=0 optim.lr_scheduler.eta_min=1e-8 logging.pl_trainer.precision=bf16-mixed
name='CL_T_t_0_05'
python main.py expname=$name model.temperature=0.05 model.encoder_xrd._target_=app.model.encoder_xrd.xrd_encoder_T optim.optimizer.weight_decay=0 optim.lr_scheduler.eta_min=1e-8 logging.pl_trainer.precision=bf16-mixed
