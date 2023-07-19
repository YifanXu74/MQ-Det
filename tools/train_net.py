# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import numpy as np
import torch
from maskrcnn_benchmark.config import cfg, try_to_find
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, is_main_process, all_gather
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.metric_logger import (MetricLogger, TensorboardLogger)
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
import random
from maskrcnn_benchmark.utils.amp import autocast, GradScaler

from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


def tuning_highlevel_override(cfg,):
    if cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE == "vision_query":
        cfg.MODEL.BACKBONE.FREEZE = True
        cfg.MODEL.FPN.FREEZE = True
        cfg.MODEL.RPN.FREEZE = True if not cfg.VISION_QUERY.QUERY_FUSION else False
        cfg.MODEL.LINEAR_PROB = False
        cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = False
        cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = False
        cfg.MODEL.DYHEAD.USE_CHECKPOINT = False # Disable checkpoint
        cfg.VISION_QUERY.ENABLED = True
    if cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE == "vs_with_txt_enc":
        cfg.MODEL.BACKBONE.FREEZE = True
        cfg.MODEL.FPN.FREEZE = True
        cfg.MODEL.RPN.FREEZE = True if not cfg.VISION_QUERY.QUERY_FUSION else False
        cfg.MODEL.LINEAR_PROB = False
        cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = False
        cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = False
        cfg.MODEL.DYHEAD.USE_CHECKPOINT = False # Disable checkpoint
        cfg.VISION_QUERY.ENABLED = True

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def extract_query(cfg):
    if cfg.DATASETS.FEW_SHOT:
        assert cfg.DATASETS.FEW_SHOT == cfg.VISION_QUERY.MAX_QUERY_NUMBER, 'To extract the right query instances, set VISION_QUERY.MAX_QUERY_NUMBER = DATASETS.FEW_SHOT.'
    # if cfg.num_gpus > 1:
    #     max_query_number = cfg.VISION_QUERY.MAX_QUERY_NUMBER
    #     cfg.defrost()
    #     cfg.VISION_QUERY.MAX_QUERY_NUMBER = int(cfg.VISION_QUERY.MAX_QUERY_NUMBER/cfg.num_gpus)
    #     cfg.freeze()

    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)


    checkpointer = DetectronCheckpointer(
        cfg, model
    )
    checkpointer.load(try_to_find(cfg.MODEL.WEIGHT))

    data_loader = make_data_loader(
        cfg,
        is_train=False,
        is_cache=True,
        is_distributed= cfg.num_gpus > 1,
    )
    assert isinstance(data_loader, list) and len(data_loader)==1
    data_loader=data_loader[0]

    # if cfg.VISION_QUERY.CUSTOM_DATA_IDS is not None:
    #     data_loader.dataset.ids = cfg.VISION_QUERY.CUSTOM_DATA_IDS

    if cfg.num_gpus > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.local_rank], output_device=cfg.local_rank,
            broadcast_buffers=cfg.MODEL.BACKBONE.USE_BN,
            find_unused_parameters=cfg.SOLVER.FIND_UNUSED_PARAMETERS
        )

    query_images=defaultdict(list)
    _iterator = tqdm(data_loader)
    # _iterator = data_loader # for debug
    model.eval()
    for i, batch in enumerate(_iterator):
        images, targets, *_ = batch
        if cfg.num_gpus > 1:
            query_images = model.module.extract_query(images.to(device), targets, query_images)
        else:
            query_images = model.extract_query(images.to(device), targets, query_images)
    
    if cfg.num_gpus > 1:
        ## not stable when using all_gather, easy to OOM.
        # all_query_images = all_gather(query_images)
        # if is_main_process():
        #     accumulated_query_images = defaultdict(list)
        #     for r, query_images_dict in enumerate(all_query_images):
        #         print('accumulating results: {}/{}'.format(r, len(all_query_images)))
        #         for label, feat in query_images_dict.items():
        #             num_queries=len(accumulated_query_images[label])
        #             if num_queries >= cfg.VISION_QUERY.MAX_QUERY_NUMBER:
        #                 continue
        #             if num_queries==0:
        #                 accumulated_query_images[label] = feat.to(device)
        #             else:
        #                 accumulated_query_images[label] = torch.cat([accumulated_query_images[label].to(device), feat.to(device)])

        #     save_name = 'MODEL/{}_query_{}_pool{}_{}{}_multi-node.pth'.format(cfg.VISION_QUERY.DATASET_NAME if cfg.VISION_QUERY.DATASET_NAME else cfg.DATASETS.TRAIN[0].split('_')[0] , cfg.VISION_QUERY.MAX_QUERY_NUMBER, cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION ,'sel' if cfg.VISION_QUERY.SELECT_FPN_LEVEL else 'all', cfg.VISION_QUERY.QUERY_ADDITION_NAME)
        #     print('saving to ', save_name)
        #     torch.save(accumulated_query_images, save_name)
        if cfg.VISION_QUERY.QUERY_BANK_SAVE_PATH != '':
            raise NotImplementedError
        global_rank = get_rank()
        save_name = 'MODEL/{}_query_{}_pool{}_{}{}_rank{}.pth'.format(cfg.VISION_QUERY.DATASET_NAME if cfg.VISION_QUERY.DATASET_NAME else cfg.DATASETS.TRAIN[0].split('_')[0] , cfg.VISION_QUERY.MAX_QUERY_NUMBER, cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION ,'sel' if cfg.VISION_QUERY.SELECT_FPN_LEVEL else 'all', cfg.VISION_QUERY.QUERY_ADDITION_NAME, global_rank)
        print('saving to ', save_name)
        torch.save(query_images, save_name)
    else:
        if cfg.VISION_QUERY.QUERY_BANK_SAVE_PATH != '':
            save_name = cfg.VISION_QUERY.QUERY_BANK_SAVE_PATH
        else:
            save_name = 'MODEL/{}_query_{}_pool{}_{}{}.pth'.format(cfg.VISION_QUERY.DATASET_NAME if cfg.VISION_QUERY.DATASET_NAME else cfg.DATASETS.TRAIN[0].split('_')[0] , cfg.VISION_QUERY.MAX_QUERY_NUMBER, cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION ,'sel' if cfg.VISION_QUERY.SELECT_FPN_LEVEL else 'all', cfg.VISION_QUERY.QUERY_ADDITION_NAME)
        print('saving to ', save_name)
        torch.save(query_images, save_name)
    # if cfg.num_gpus > 1:
    #     # 
    #     world_size = torch.distributed.dist.get_world_size()
    #     if is_main_process():
    #         query_images_list = []
    #         for r in range(world_size):
    #             saved_path = 'MODEL/{}_query_{}_pool{}_{}{}_rank{}.pth'.format(cfg.VISION_QUERY.DATASET_NAME if cfg.VISION_QUERY.DATASET_NAME else cfg.DATASETS.TRAIN[0].split('_')[0] , cfg.VISION_QUERY.MAX_QUERY_NUMBER, cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION ,'sel' if cfg.VISION_QUERY.SELECT_FPN_LEVEL else 'all', cfg.VISION_QUERY.QUERY_ADDITION_NAME, r)
    #             query_images_list.append(torch.load(saved_path, map_location='cpu'))
            
    #         for s in query_images_list

    
def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )

    parser.add_argument("--use-tensorboard",
                        dest="use_tensorboard",
                        help="Use tensorboardX logger (Requires tensorboardX installed)",
                        action="store_true",
                        default=False
                        )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument("--save_original_config", action="store_true")
    parser.add_argument("--disable_output_distributed", action="store_true")
    parser.add_argument("--override_output_dir", default=None)
    parser.add_argument("--custom_shot_and_epoch_and_general_copy", default=None, type=str)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--extract_query", action="store_true", default=False)
    parser.add_argument(
        "--task_config",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--additional_model_config",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )



    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        import datetime
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
            timeout=datetime.timedelta(0, 7200)
        )
    
    if args.disable_output_distributed:
        setup_for_distributed(args.local_rank <= 0)

    cfg.local_rank = args.local_rank
    cfg.num_gpus = num_gpus

    cfg.merge_from_file(args.config_file)
    if args.task_config:
        cfg.merge_from_file(args.task_config)
    if args.additional_model_config:
        cfg.merge_from_file(args.additional_model_config)
    cfg.merge_from_list(args.opts)
    # specify output dir for models
    if args.override_output_dir:
        cfg.OUTPUT_DIR = args.override_output_dir
    tuning_highlevel_override(cfg)
    cfg.freeze()

    seed = cfg.SOLVER.SEED + args.local_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info(args)
    logger.info("Using {} GPUs".format(num_gpus))

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    if args.save_original_config:
        import shutil
        shutil.copy(args.config_file, os.path.join(cfg.OUTPUT_DIR, 'config_original.yml'))
    
    save_config(cfg, output_config_path)

    if args.extract_query:
        extract_query(cfg)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()