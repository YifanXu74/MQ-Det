# MQ-Det: Multi-modal Queried Object Detection in the Wild
Official PyTorch implementation of "[MQ-Det: Multi-modal Queried Object Detection in the Wild](https://arxiv.org/abs/2305.18980)": the first multi-modal queried open-set object detector.

## Multi-modal Queried Object Detection
We introduce **MQ-Det**, an efficient architecture and pre-training strategy design to utilize both textual description with open-set generalization and visual exemplars with rich description granularity as category queries, namely, **M**ulti-modal **Q**ueried object **Det**ection, for real-world detection with both open-vocabulary categories and various granularity. 
<img src=".asset/method.png" width="800"> 

## Method
MQ-Det incorporates vision queries into existing well-established language-queried-only detectors.

**Features**:

- A plug-and-play gated class-scalable perceiver module upon the frozen detector.
- A vision conditioned masked language prediction strategy.
- Compatible with most language-queried object detectors.

## TODO 

- [x] Release zero-shot inference code.
- [x] Release checkpoints.
- [ ] Release fine-tuning code.
- [ ] Release modulated training code.
- [ ] More detailed instruction on applying MQ-Det to custom language-queried detectors.

## Preparation
**Data**  Prepare ``Objects365`` (for modulated pre-training), ``LVIS`` (for evaluation), and ``ODinW`` (for evaluation) benchmarks following [DATA.md](DATA.md).




**Environment** This repo requires Pytorch==1.9  and torchvision. 
Init the  environment:
```
bash init.sh
```

**Initial weight** MQ-Det is build upon frozen language-queried detector. To conduct modulated pre-training, download corresponding pre-trained model weights first.

We apply MQ-Det on GLIP and GroundingDINO:

```
GLIP-T:
wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_tiny_model_o365_goldg_cc_sbu.pth -O MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth
GLIP-L:
wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_large_model.pth -O MODEL/glip_large_model.pth
GroundingDINO-T:
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O MODEL/groundingdino_swint_ogc.pth
```

## Model Zoo
Model | LVIS MiniVal | LVIS Val v1.0 | ODinW-13 | ODinW-35 | Config  | Weight
-- | -- | -- | -- | -- | -- | --
MQ-GLIP-T | 30.4 | 22.6 | 45.6 | 20.8 | [config](configs/pretrain/mq-glip-t.yaml)  | [weight](https://drive.google.com/file/d/1n0_D-tisqN5v-IESUEIGzMuO-9wolXiu/view?usp=sharing)
MQ-GLIP-L | 43.4 | 34.7 | 54.1 | 23.9 | [config](configs/pretrain/mq-glip-l.yaml)  | [weight](https://drive.google.com/file/d/1O_eb1LrlNqpEsoxD23PAIxW8WB6sGoBO/view?usp=sharing)

## Vision Query Extraction

**Take MQ-GLIP-T as an example.**

If you wish to extract vision queries from custom dataset, specify the  ``DATASETS.TRAIN`` in the config file.
We provide some examples in our implementation in the following.

### Objects365 for modulated pre-training:

```
python tools/extract_vision_query.py --config_file configs/pretrain/mq-glip-t.yaml --dataset objects365 --add_name tiny 
```
This will generate a query bank file in ``MODEL/object365_query_5000_sel_tiny.pth``


Some paramters corresponding to the query extraction:

``DATASETS.FEW_SHOT``: if set ``k>0``, the dataset will be subsampled to k-shot for each category when initializing the dataset. This is completed before training. Not used during pre-training.

``VISION_QUERY.MAX_QUERY_NUMBER``: the max number of vision queries for each category when extracting the query bank. Note that the query extraction is conducted before training and evaluation.

``VISION_QUERY.NUM_QUERY_PER_CLASS`` controls how many queries to provide for each category during one forward process in training and evaluation.

Usually, we set 

``VISION_QUERY.MAX_QUERY_NUMBER=5000``, ``VISION_QUERY.NUM_QUERY_PER_CLASS=5``, ``DATASETS.FEW_SHOT=0`` during pre-training. 

``VISION_QUERY.MAX_QUERY_NUMBER=5``, ``VISION_QUERY.NUM_QUERY_PER_CLASS=5``, ``DATASETS.FEW_SHOT=5`` during few-shot (5-shot) fine-tuning.



### LVIS for downstream tasks:
```
python tools/extract_vision_query.py --config_file configs/pretrain/mq-glip-t.yaml --dataset lvis --num_vision_queries 5 --add_name tiny
```
This will generate a query bank file in ``MODEL/lvis_query_5_pool7_sel_tiny.pth``.

``--num_vision_queries`` denotes number of vision queries for each category, and can be an arbitrary number. This will set both ``VISION_QUERY.MAX_QUERY_NUMBER`` and ``DATASETS.FEW_SHOT`` to ``num_vision_queries``.
Note that here ``DATASETS.FEW_SHOT`` is only for accelerating the extraction process.

``--add_name`` is only a mark for different models.
For training/evaluating with MQ-GLIP-T/MQ-GLIP-L/MQ-GroundingDINO, we set ``--add_name`` to 'tiny'/'large'/'gd'.

### ODinW for downstream tasks:

```
python tools/extract_vision_query.py --config_file configs/pretrain/mq-glip-t.yaml --dataset odinw-13 --num_vision_queries 5 --add_name tiny
```
This will generate query bank files for each dataset in ODinW in  ``MODEL/{dataset}_query_5_pool7_sel_tiny.pth``.


## (Zero-Shot) Evaluation
**Take MQ-GLIP-T as an example.**

### LVIS Evaluation
```
python -m torch.distributed.launch --nproc_per_node=4 \
tools/test_grounding_net.py \
--config-file configs/pretrain/mq-glip-t.yaml \
--additional_model_config configs/vision_query_5shot/lvis_minival.yaml \
VISION_QUERY.QUERY_BANK_PATH MODEL/lvis_query_5_pool7_sel_tiny.pth \
MODEL.WEIGHT model_weight_path \
TEST.IMS_PER_BATCH 4 
```
If you wish to evaluate on Val 1.0, set ``--task_config`` to ``configs/vision_query_5shot/lvis_val.yaml``.
``VISION_QUERY.QUERY_BANK_PATH`` is the vision queries extracted via ``tools/extract_vision_query.py``. Please follow the above section to extract corresponding vision queries.

### ODinW / Custom Dataset Evaluation
```
python tools/eval_odinw.py --config_file configs/pretrain/mq-glip-t.yaml \
--opts 'MODEL.WEIGHT model_weight_path' \
--setting zero-shot \
--add_name tiny \
--log_path 'OUTPUT/odinw_log/'
```
The results are stored at ``OUTPUT/odinw_log/``.

If you wish to use custom vision queries or datasets, add ``'VISION_QUERY.QUERY_BANK_PATH custom_bank_path'`` to the ``--opts`` argment, and also modify the ``dataset_configs`` in the ``tools/eval_odinw.py``. 

## Single-Modal Evaluation

Here we provide introduction on utilizing single modal queries, such as visual exemplars or textual description.


Follow the command as in ``(Zero-Shot) Evaluation``. But set the following hyper-parameters.

To solely use vision queries, add hyper-parameters:
```
VISION_QUERY.MASK_DURING_INFERENCE True TEXT_DROPOUT 1.0
```

To solely use language queries, add hyper-parameters:
```
VISION_QUERY.ENABLED FALSE
```

## Citation

If you find our work useful in your research, please consider citing:
```
@article{mqdet,
  title={Multi-modal Queried Object Detection in the Wild},
  author={Xu, Yifan and Zhang, Mengdan and Fu, Chaoyou and Chen, Peixian and Yang, Xiaoshan and Li, Ke and Xu, Changsheng},
  journal={arXiv preprint arXiv:2305.18980},
  year={2023}
}
```

