################################
'''
Remember to change the new path in this template before training:
1. main.py -- PROJECT_ROOT
2. conf/default.yaml -- workpath
3. conf/data/default.yaml -- file_path
'''
################################

work_path='/code/CL'
cd $work_path

# daytime='2024-06-26'
# name='CL_CNN'
# python main.py expname=$name model.temperature=0.05 model.encoder_xrd._target_=app.model.encoder_xrd.xrd_encoder_CNN optim.optimizer.weight_decay=0 optim.optimizer.lr=1e-3 optim.lr_scheduler.eta_min=1e-7
# python main.py expname=$name model.temperature=1.0 model.encoder_xrd._target_=app.model.encoder_xrd.xrd_encoder_CNN optim.optimizer.weight_decay=1e-4 optim.optimizer.lr=1e-3


# daytime='2024-06-26'
# name='CL_T'
# python main.py expname=$name model.temperature=1.0 model.encoder_xrd._target_=app.model.encoder_xrd.xrd_T optim.optimizer.weight_decay=0 optim.lr_scheduler.eta_min=1e-8 logging.pl_trainer.precision=bf16-mixed
# python main.py expname=$name model.temperature=0.05 model.encoder_xrd._target_=app.model.encoder_xrd.xrd_T optim.optimizer.weight_decay=0 optim.lr_scheduler.eta_min=1e-8 logging.pl_trainer.precision=bf16-mixed
