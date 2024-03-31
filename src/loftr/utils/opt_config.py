from yacs.config import CfgNode as CN


def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


_CN = CN()
_CN.BACKBONE_TYPE = 'RepVGG'
_CN.ALIGN_CORNER = False
_CN.RESOLUTION = (8, 1)
_CN.FINE_WINDOW_SIZE = 8  # window_size in fine_level, must be even
_CN.MP = False
_CN.REPLACE_NAN = True
_CN.HALF = False

# 1. LoFTR-backbone (local feature CNN) config
_CN.BACKBONE = CN()
_CN.BACKBONE.BLOCK_DIMS = [64, 128, 256]  # s1, s2, s3

# 2. LoFTR-coarse module config
_CN.COARSE = CN()
_CN.COARSE.D_MODEL = 256
_CN.COARSE.D_FFN = 256
_CN.COARSE.NHEAD = 8
_CN.COARSE.LAYER_NAMES = ['self', 'cross'] * 4
_CN.COARSE.AGG_SIZE0 = 4
_CN.COARSE.AGG_SIZE1 = 4
_CN.COARSE.NO_FLASH = False
_CN.COARSE.ROPE = True
_CN.COARSE.NPE = [832, 832, 832, 832] # [832, 832, long_side, long_side] Suggest setting based on the long side of the input image, especially when the long_side > 832

# 3. Coarse-Matching config
_CN.MATCH_COARSE = CN()
_CN.MATCH_COARSE.THR = 25 # recommend 0.2 for full model and 25 for optimized model
_CN.MATCH_COARSE.BORDER_RM = 2
_CN.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.MATCH_COARSE.SKIP_SOFTMAX = True # False for full model and True for optimized model
_CN.MATCH_COARSE.FP16MATMUL = True # False for full model and True for optimized model
_CN.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.2  # training tricks: save GPU memory
_CN.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200  # training tricks: avoid DDP deadlock

# 4. Fine-Matching config
_CN.MATCH_FINE = CN()
_CN.MATCH_FINE.LOCAL_REGRESS_TEMPERATURE = 10.0 # use 10.0 as fine local regress temperature, not 1.0
_CN.MATCH_FINE.LOCAL_REGRESS_SLICEDIM = 8

opt_default_cfg = lower_config(_CN)
