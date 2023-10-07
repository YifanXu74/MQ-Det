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
from maskrcnn_benchmark.engine.trainer import do_train
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


def train(cfg, local_rank, distributed, use_tensorboard=False, resume=False):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    if cfg.GROUNDINGDINO.enabled:
        if cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE == "vision_query":
            for key, p in model.named_parameters():
                if not ('pre_select' in key or 'qv_layer' in key):
                    p.requires_grad = False
    else:
        if cfg.SOLVER.TUNING_HIGHLEVEL_OVERRIDE == "vision_query":
            if model.language_backbone is not None:
                for key, p in model.language_backbone.named_parameters():
                    if not ('pre_select' in key or 'qv_layer' in key):
                        p.requires_grad = False
            if cfg.VISION_QUERY.QUERY_FUSION:
                if model.rpn is not None:
                    for key, p in model.rpn.named_parameters():
                        if not ('pre_select' in key or 'qv_layer' in key):
                            p.requires_grad = False

    if cfg.MODEL.BACKBONE.RESET_BN:
        for name, param in model.named_buffers():
            if 'running_mean' in name:
                torch.nn.init.constant_(param, 0)
            if 'running_var' in name:
                torch.nn.init.constant_(param, 1)

    if cfg.SOLVER.GRAD_CLIP > 0:
        clip_value = cfg.SOLVER.GRAD_CLIP
        for p in filter(lambda p: p.grad is not None, model.parameters()):
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=0  # <TODO> Sample data from resume is disabled, due to the conflict with max_epoch
    )

    if cfg.TEST.DURING_TRAINING or cfg.SOLVER.USE_AUTOSTEP:
        data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
        data_loaders_val = data_loaders_val[0]
    else:
        data_loaders_val = None

    if cfg.GROUNDINGDINO.enabled:
        pass
    else:
        if cfg.MODEL.BACKBONE.FREEZE:
            for p in model.backbone.body.parameters():
                p.requires_grad = False

        if cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
            print("LANGUAGE_BACKBONE FROZEN.")
            for p in model.language_backbone.body.parameters():
                p.requires_grad = False

        if cfg.MODEL.FPN.FREEZE:
            for p in model.backbone.fpn.parameters():
                p.requires_grad = False
        if cfg.MODEL.RPN.FREEZE:
            for p in model.rpn.parameters():
                p.requires_grad = False
    
    # if cfg.SOLVER.PROMPT_PROBING_LEVEL != -1:
    #     if cfg.SOLVER.PROMPT_PROBING_LEVEL == 1:
    #         for p in model.parameters():
    #             p.requires_grad = False

    #         for p in model.language_backbone.body.parameters():
    #             p.requires_grad = True

    #         for name, p in model.named_parameters():
    #             if p.requires_grad:
    #                 print(name, " : Not Frozen")
    #             else:
    #                 print(name, " : Frozen")
    #     else:
    #         assert(0)

    optimizer = make_optimizer(cfg, model)
    print('Making scheduler')
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        print('Distributing model')
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=cfg.MODEL.BACKBONE.USE_BN,
            find_unused_parameters=cfg.SOLVER.FIND_UNUSED_PARAMETERS
        )
        print('Done')

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    print('Making checkpointer')
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    if resume and cfg.OUTPUT_DIR != "OUTPUT":
        if not os.path.exists(cfg.OUTPUT_DIR):
            load_weight=cfg.MODEL.WEIGHT
        else:
            checkpoint_list=[name for name in os.listdir(cfg.OUTPUT_DIR) if name.endswith('.pth') and 'final' not in name and 'resume' not in name]
            if len(checkpoint_list)==0:
                load_weight=cfg.MODEL.WEIGHT
                resume=False
            else:
                max_bits=len(checkpoint_list[0].split('.')[0].split('_')[-1])
                iter_list=[int(name.split('.')[0].split('_')[-1]) for name in checkpoint_list]
                max_iter=str(max(iter_list)).zfill(max_bits)
                resume_weight_name='model_'+max_iter+'.pth'
                load_weight=str(Path(cfg.OUTPUT_DIR, resume_weight_name))
    else:
        load_weight=cfg.MODEL.WEIGHT

    print('Loading checkpoint')
    extra_checkpoint_data = checkpointer.load(try_to_find(load_weight))
    arguments.update(extra_checkpoint_data)

    # enable resume
    data_loader.batch_sampler.start_iter = arguments["iteration"] + 1 if resume else 0


    # data_loader = make_data_loader(
    #     cfg,
    #     is_train=True,
    #     is_distributed=distributed,
    #     start_iter=arguments["iteration"] + 1 if resume else 0  # <TODO> Sample data from resume is disabled, due to the conflict with max_epoch
    # )

    # if cfg.TEST.DURING_TRAINING or cfg.SOLVER.USE_AUTOSTEP:
    #     data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    #     data_loaders_val = data_loaders_val[0]
    # else:
    #     data_loaders_val = None
    
    if cfg.DATASETS.FEW_SHOT:
        arguments["dataset_ids"] = data_loader.dataset.ids

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    if use_tensorboard:
        meters = TensorboardLogger(
            log_dir=cfg.OUTPUT_DIR,
            start_iter=arguments["iteration"],
            delimiter="  "
        )
    else:
        meters = MetricLogger(delimiter="  ")
    
    if is_main_process():
        for name, p in model.named_parameters():
            if p.requires_grad:
                print(name, " : Not Frozen")
            else:
                print(name, " : Frozen")
    do_train(
        cfg,
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        data_loaders_val,
        meters,
    )

    return model

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
    parser.add_argument("--local-rank", type=int, default=0)
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
        # if cfg.DATASETS.FEW_SHOT and cfg.VISION_QUERY.ENABLED:
        #     max_query_number = int(os.path.basename(cfg.VISION_QUERY.QUERY_BANK_PATH).split('_')[-2])
        #     assert cfg.DATASETS.FEW_SHOT == max_query_number, 'You should first extract corresponding few-shot query instances.'
        #     assert max_query_number >= cfg.VISION_QUERY.NUM_QUERY_PER_CLASS
        model = train(cfg=cfg,
                    local_rank=args.local_rank,
                    distributed=args.distributed,
                    use_tensorboard=args.use_tensorboard, 
                    resume=args.resume)


if __name__ == "__main__":
    main()
