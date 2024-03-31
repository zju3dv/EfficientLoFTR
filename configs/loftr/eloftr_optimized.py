from src.config.default import _CN as cfg

# training config
cfg.TRAINER.CANONICAL_LR = 8e-3
cfg.TRAINER.WARMUP_STEP = 1875  # 3 epochs
cfg.TRAINER.WARMUP_RATIO = 0.1
cfg.TRAINER.MSLR_MILESTONES = [8, 12, 16, 20, 24]
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5
cfg.TRAINER.OPTIMIZER = "adamw"
cfg.TRAINER.ADAMW_DECAY = 0.1
cfg.TRAINER.EPI_ERR_THR = 5e-4 # recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth (from SuperGlue)
cfg.TRAINER.GRADIENT_CLIPPING = 0.0
cfg.LOFTR.LOSS.FINE_TYPE = 'l2'  # ['l2_with_std', 'l2']
cfg.LOFTR.LOSS.COARSE_OVERLAP_WEIGHT = True
cfg.LOFTR.LOSS.FINE_OVERLAP_WEIGHT = True
cfg.LOFTR.LOSS.LOCAL_WEIGHT = 0.25
cfg.LOFTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3
cfg.LOFTR.MATCH_COARSE.SPARSE_SPVS = True

# model config
cfg.LOFTR.RESOLUTION = (8, 1)
cfg.LOFTR.FINE_WINDOW_SIZE = 8  # window_size in fine_level, must be even
cfg.LOFTR.ALIGN_CORNER = False
cfg.LOFTR.MP = True # just for reproducing paper, FP16 is much faster on modern GPUs
cfg.LOFTR.REPLACE_NAN = True
cfg.LOFTR.EVAL_TIMES = 5
cfg.LOFTR.COARSE.NO_FLASH = True # Not use Flash-Attention just for reproducing paper timing
cfg.LOFTR.MATCH_FINE.LOCAL_REGRESS_TEMPERATURE = 10.0
cfg.LOFTR.MATCH_FINE.LOCAL_REGRESS_SLICEDIM = 8

# dataset config
cfg.DATASET.FP16 = False

# optimized model config
cfg.LOFTR.MATCH_COARSE.FP16MATMUL = True
cfg.LOFTR.MATCH_COARSE.SKIP_SOFTMAX = True
cfg.LOFTR.MATCH_COARSE.THR = 25.0 # recommend 0.2 for full model and 25 for optimized model