import os, sys
sys.path.append('../')

from config.config_utils import check_shape
from core.utils.check_model_name import check as model_name
from core.utils.check_loss_name import check as loss_name

note = 'all_c1_05-05-10_trilinear_trilinear_Spacing-None'
config = {
    "TARGET_NAME"   : "abd_vein",
    "VERSION"       : 1,
    "FOLD"          : 1,
    "FOLDS"         : 5, 
    "SPACING"       : [None, None, None],
    "CHANNEL_IN"    : 1,
    "CHANNEL_OUT"   : 1,
    "Interpolation" : "trilinear_trilinear",
    "JSON"          : "binary",
    "CLASS_NAMES"   : {1: "Whole_Vein"},
    "LOAD_MODEL"    : '/raid/users/mi_pje_0/train_results/abd_vein/nnunet/dice+dist+navi/all_c1_10-10-10_area_3[128, 128, 128]_drop[0.1]_window[-200, 250]_fold1/model_best.pth',
    #### wandb
    "LOGGING"       : True,
    "ENTITY"        : "jeune",
    "PROJ_NAME"     : "abd_vein",
    "VISUAL_AXIS"   : 3, # 1 or 2 or 3
    #### system
    "GPUS"          : "2,3",
    "VALID_GPU"     : True,
    "WORKERS"       : 20,
    "AMP"           : False,
    #### training
    "DATA_SPLIT"    : 1.0,
    "MODEL_NAME"    : model_name("nnunet"), #"unetr_pretrained",# "caddunet",
    "BATCH_SIZE"    : 1,
    "SAMPLES"       : 4,
    "ACTIVATION"    : 'sigmoid',
    "LOSS_NAME"     : loss_name("dice+dist+navi"),
    "LOSS_WEGITHS"  : [0.5,0.5,1.0],
    "TRANSFORM"     : "ArteryPre",
    "CROP_RATIO"    : [1,3],
    "INPUT_SHAPE"   : [128,128,128], 
    "DROPOUT"       : 0.1,
    "CONTRAST"      : [-150,300], 
    "MAX_ITERATIONS": 100000,
    "SAVE_FREQUENT" : 1000,
    "EVAL_NUM"      : 250,
    "SEEDS"         : 12321,
    "OPTIM_NAME"    : "AdamW",
    "LR_INIT"       : 0.001,
    "LR_DECAY"      : 1e-05,
    "MOMENTUM"      : 0.9,
    "DEEP_SUPERVISION": [True,[1,1,1,1,1]],
    "SCHEDULER"     : "plateau", # expoential, plateau, cosine, none
    "CHECK_MODEL"   : True,
    "THRESHOLD"     : 0.5,
    ## SDF
    "SDF"           : [False, 0.5],
    ## unet
    "CHANNEL_LIST"  : (32, 64, 128, 256, 512), 
    # "STRIDES"       : [2,2,2,2], # [2,2,2],
    # "NUM_RES_UNITS" : 2,
    # ## cadd unet
    # "PADDING"       : [1,2,4],
    # "DILATION"      : [1,2,4],
    # "KERNEL_SIZE"   : (3, 3, 3),
}
config["JSON"] = f'/raid/dataset/hutom/medical_image/vein/{config["Interpolation"]}/ALL_c1_cc3d6/{config["SPACING"]}/{config["JSON"]}.json'
config["NUM_GPUS"] = len(config["GPUS"].split(','))

experiment_name = f'{config["INPUT_SHAPE"]}_drop{[config["DROPOUT"]]}_window{config["CONTRAST"]}'
experiment_name += f'_fold{config["FOLD"]}'
if 'multi' in config["JSON"].lower(): experiment_name = 'MultiClass_'+experiment_name
if config["AMP"] : experiment_name = 'Mixed_'+experiment_name
if config["MODEL_NAME"] in ["caddunet"]:
    experiment_name += f'_param{config["PADDING_DILATION"]}'
if config["MODEL_NAME"] in ["swin_unetr", "unetr"]:
    experiment_name += f'_feature{config["FEATURE_SIZE"]}'
if config["MODEL_NAME"] in ["unest"]:
    experiment_name += f'_feature{config["FEATURE_SIZE"]}_patch{config["PATCH_SIZE"]}'


if len(note)>1:
    experiment_name = note + experiment_name
# config["LOGDIR"] = f'/nas3/jepark/train_results/{config["TARGET_NAME"]}/{config["MODEL_NAME"]}/{config["LOSS_NAME"]}/{experiment_name}'
config["LOGDIR"] = f'/raid/users/mi_pje_0/train_results/{config["TARGET_NAME"]}/{config["MODEL_NAME"]}/{config["LOSS_NAME"]}/{experiment_name}'
config["LOGDIR"] = config["LOGDIR"]
if os.path.isdir(config["LOGDIR"])==False:
    os.makedirs(config["LOGDIR"])
