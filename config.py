 
__all__ = ['r3d_18', 'se_r3d_18','da_18','da_se_18']


NET_NAME = 'r3d_18'
VERSION = 'v1.0'
DEVICE = '0'
# Must be True when pre-training and inference
PRE_TRAINED = True 
# 1,2,3,4,5
CURRENT_FOLD = 1
GPU_NUM = len(DEVICE.split(','))


WEIGHT_PATH = {
  'r3d_18':'../ckpt/{}/'.format(VERSION),
  'se_r3d_18':'../ckpt/{}/'.format(VERSION),
  'da_18':'../ckpt/{}/'.format(VERSION),
  'da_se_18':'../ckpt/{}/'.format(VERSION),
}

# Arguments when trainer initial
INIT_TRAINER = {
  'net_name':NET_NAME,
  'lr':1e-3, 
  'n_epoch':30,
  'channels':1,
  'num_classes':3,
  'input_shape':(64,256,256),
  'crop':48,
  'batch_size':4,
  'num_workers':2,
  'device':DEVICE,
  'pre_trained':PRE_TRAINED,
  'weight_path':WEIGHT_PATH[NET_NAME],
  'weight_decay': 0.,
  'momentum': 0.9,
  'gamma': 0.1,
  'milestones': [40,80],
  'T_max':5,
 }

# Arguments when perform the trainer 
SETUP_TRAINER = {
  'output_dir':'./ckpt/{}'.format(VERSION),
  'log_dir':'./log/{}'.format(VERSION),
  'optimizer':'Adam',
  'loss_fun':'Cross_Entropy',
  'class_weight':None,
  'lr_scheduler':'CosineAnnealingLR'
  }

