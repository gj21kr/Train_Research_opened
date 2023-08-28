def check(name):
    name = name.lower()
    if name in ['navi', 'navi airway', 'navi_airway']:
        return 'navi'
    elif 'trans:' in name:
        t = name.split(':')[-1].split('-')
        en = t[0]; de = t[1]
        if en not in ['beit', 'swint', 'swint3d', 'dino']:
            print('Wrong TransSeg model name!')
            raise 
        if de not in ['upernet', 'upernet_swint', 'setrpup', 'convtrans', 'unetr']:
            print('Wrong TransSeg model name!')
            raise
        return name
    if name in ['ernet', 'er_net', 'er net']:
        return 'er_net'
    elif name in ['swin_unet', 'swinunetr_unet', 'swin unetr_unet', 'ctn_swin_unet']:
        return 'swin_unet'
    elif name in ['unest']:
        return 'unest'
    elif name in ['unetr']:
        return 'unetr'
    elif name in ['unetr_pretrained', 'unetr_pre', 'pre_unetr']:
        return 'unetr_pretrained'
    elif name in ['vnet']:
        return 'vnet'
    elif name in ['unet']:
        return 'unet'
    elif name in ['dynunet']:
        return 'dynunet'
    elif name in ['swin_unetr', 'sunetr', 'swinunetr']:
        return 'swin_unetr'
    elif name in ['denseunet', 'dense_unet', 'dunet', 'd_unet']:
        return 'dense_unet'
    elif name in ['cadd_unet', 'caddunet']:
        return 'cadd_unet'
    elif name in ['dd_unet', 'ddunet']:
        return 'dd_unet'
    elif name in ["voxresnet", "voxel_resnet"]:
        return 'voxel_resnet'
    elif name in ["r2att_unet", "r2attunet"]:
        return 'r2att_unet'
    elif name in ['deeplabv3']:
        return 'deeplabv3'
    elif name in ['seunet', 'se_unet', 'se unet']:
        return 'se_unet'
    elif name in ['cas_net', 'casnet', 'cas net']:
        return 'cas_net'
    elif name in ['ys_unet', 'unet_ys', 'ys unet', 'unet ys']:
        return 'ys_unet'
    elif name in ['nnunet', 'nn_unet', 'ys_nnunet', 'ys_nn_unet']:
        return 'nnunet'
    elif name in ['universal_unet', 'uni_unet']:
        return 'universal_unet'
    elif name in ['universal_swinunetr', 'universal_swin', 'uni_swinunetr']:
        return 'universal_swinunetr'
    elif name in ['universal_dint', 'uni_dint']:
        return 'universal_dint'
    else:
        assert "Wrong Model Name!"
