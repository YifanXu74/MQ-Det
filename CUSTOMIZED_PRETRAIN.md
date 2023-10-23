MQ-Det supports modulated training on any datasets in COCO format. Let's take COCO for example.

To conduct customized modulating, you can follow these steps.


**1.  Add customized dataset infomation in two places of the [code](maskrcnn_benchmark/config/paths_catalog.py)**

1. L120: [Add DatasetCatalog](https://github.com/YifanXu74/MQ-Det/blob/bbacce45f8223d136ceb2be13dd18208cdc9b3db/maskrcnn_benchmark/config/paths_catalog.py#L120)

2. L394: [Add to factory](https://github.com/YifanXu74/MQ-Det/blob/bbacce45f8223d136ceb2be13dd18208cdc9b3db/maskrcnn_benchmark/config/paths_catalog.py#L394)

**NOTE:** make sure to add "_grounding" in dataset name.

Here we add a new dataset ``coco_grounding_train_for_obj365``.


**2.  Acquire customized config files**

You can modified upon the [official pretraining config file](configs/pretrain/mq-glip-t.yaml) to get a customized config file. Here we provide an [example](configs/pretrain/mq-glip-t_coco.yaml), which is modified upon [``mq-glip-t.yaml``](configs/pretrain/mq-glip-t.yaml) with"``NOTE``" on all modifications.  You can customize your own needs following the "``NOTE``" in the file.

Make sure to use correct ``DATASETS.TRAIN`` and ``VISION_QUERY.QUERY_BANK_PATH``.

Here we use a new config file ``configs/pretrain/mq-glip-t_coco.yaml``.


**3.  Extract vision queries**
```
python tools/train_net.py \
--config-file configs/pretrain/mq-glip-t_coco.yaml \
--extract_query \
VISION_QUERY.QUERY_BANK_PATH "" \
VISION_QUERY.QUERY_BANK_SAVE_PATH MODEL/coco_query_5000_sel_tiny.pth
```

Here we can get a new query bank ``MODEL/coco_query_5000_sel_tiny.pth``. Make sure the ``VISION_QUERY.QUERY_BANK_PATH`` in the config file to be this query bank path.

You can specify ``VISION_QUERY.MAX_QUERY_NUMBER`` (number of queries for each category in the bank, default 5000) to any number to control the bank size.


**4.  Conduct modulated pretraining**
```
python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --config-file configs/pretrain/mq-glip-t_coco.yaml --use-tensorboard OUTPUT_DIR 'OUTPUT/MQ-GLIP-TINY-COCO/'
```

You can specify ``VISION_QUERY.NUM_QUERY_PER_CLASS`` (default 5) to control the number of vision queries for each category in one forward process during training.
