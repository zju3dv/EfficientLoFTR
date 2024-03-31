import pytorch_lightning as pl
import argparse
import pprint
from loguru import logger as loguru_logger

from src.config.default import get_cfg_defaults
from src.utils.profiler import build_profiler

from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_loftr import PL_LoFTR

import torch

def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'data_cfg_path', type=str, help='data config path')
    parser.add_argument(
        'main_cfg_path', type=str, help='main config path')
    parser.add_argument(
        '--ckpt_path', type=str, default="weights/indoor_ds.ckpt", help='path to the checkpoint')
    parser.add_argument(
        '--dump_dir', type=str, default=None, help="if set, the matching results will be dump to dump_dir")
    parser.add_argument(
        '--profiler_name', type=str, default=None, help='options: [inference, pytorch], or leave it unset')
    parser.add_argument(
        '--batch_size', type=int, default=1, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=2)
    parser.add_argument(
        '--thr', type=float, default=None, help='modify the coarse-level matching threshold.')
    parser.add_argument(
        '--pixel_thr', type=float, default=None, help='modify the RANSAC threshold.')
    parser.add_argument(
        '--ransac', type=str, default=None, help='modify the RANSAC method')
    parser.add_argument(
        '--scannetX', type=int, default=None, help='ScanNet resize X')
    parser.add_argument(
        '--scannetY', type=int, default=None, help='ScanNet resize Y')
    parser.add_argument(
        '--megasize', type=int, default=None, help='MegaDepth resize')
    parser.add_argument(
        '--npe', action='store_true', default=False, help='')
    parser.add_argument(
        '--fp32', action='store_true', default=False, help='')
    parser.add_argument(
        '--ransac_times', type=int, default=None, help='repeat ransac multiple times for more robust evaluation')
    parser.add_argument(
        '--rmbd', type=int, default=None, help='remove border matches')
    parser.add_argument(
        '--deter', action='store_true', default=False, help='use deterministic mode for testing')
    parser.add_argument(
        '--half', action='store_true', default=False, help='pure16')
    parser.add_argument(
        '--flash', action='store_true', default=False, help='flash')

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    pprint.pprint(vars(args))

    # init default-cfg and merge it with the main- and data-cfg        
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    if args.deter:
        torch.backends.cudnn.deterministic = True
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility

    # tune when testing
    if args.thr is not None:
        config.LOFTR.MATCH_COARSE.THR = args.thr

    if args.scannetX is not None and args.scannetY is not None:
        config.DATASET.SCAN_IMG_RESIZEX = args.scannetX
        config.DATASET.SCAN_IMG_RESIZEY = args.scannetY
    if args.megasize is not None:
        config.DATASET.MGDPT_IMG_RESIZE = args.megasize

    if args.npe:
        if config.LOFTR.COARSE.ROPE:
            assert config.DATASET.NPE_NAME is not None
        if config.DATASET.NPE_NAME is not None:
            if config.DATASET.NPE_NAME == 'megadepth':
                config.LOFTR.COARSE.NPE = [832, 832, config.DATASET.MGDPT_IMG_RESIZE, config.DATASET.MGDPT_IMG_RESIZE] # [832, 832, 1152, 1152]
            elif config.DATASET.NPE_NAME == 'scannet':
                config.LOFTR.COARSE.NPE = [832, 832, config.DATASET.SCAN_IMG_RESIZEX, config.DATASET.SCAN_IMG_RESIZEX] # [832, 832, 640, 640]
    else:
        config.LOFTR.COARSE.NPE = [832, 832, 832, 832]

    if args.ransac_times is not None:
        config.LOFTR.EVAL_TIMES = args.ransac_times

    if args.rmbd is not None:
        config.LOFTR.MATCH_COARSE.BORDER_RM = args.rmbd

    if args.pixel_thr is not None:
        config.TRAINER.RANSAC_PIXEL_THR = args.pixel_thr

    if args.ransac is not None:
        config.TRAINER.POSE_ESTIMATION_METHOD = args.ransac
        if args.ransac == 'LO-RANSAC' and config.TRAINER.RANSAC_PIXEL_THR == 0.5:
            config.TRAINER.RANSAC_PIXEL_THR = 2.0

    if args.fp32:
        config.LOFTR.MP = False

    if args.half:
        config.LOFTR.HALF = True
        config.DATASET.FP16 = True
    else:
        config.LOFTR.HALF = False
        config.DATASET.FP16 = False

    if args.flash:
        config.LOFTR.COARSE.NO_FLASH = False

    loguru_logger.info(f"Args and config initialized!")

    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = PL_LoFTR(config, pretrained_ckpt=args.ckpt_path, profiler=profiler, dump_dir=args.dump_dir)
    loguru_logger.info(f"LoFTR-lightning initialized!")

    # lightning data
    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f"DataModule initialized!")

    # lightning trainer
    trainer = pl.Trainer.from_argparse_args(args, replace_sampler_ddp=False, logger=False)

    loguru_logger.info(f"Start testing!")
    trainer.test(model, datamodule=data_module, verbose=False)