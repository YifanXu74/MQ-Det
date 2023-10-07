We provide guidance for preparing the data used by MQ-DET. Note that not all data are needed for a specific experiments. Please check the `` Required Data`` fields in [README](README.md) to download necessary data. All data should by placed under the ``DATASET`` folder.

The data should be organized in the following format:
```
DATASET/
    coco/
        annotations/
            lvis_od_train.json
            lvis_od_val.json
            lvis_v1_minival_inserted_image_name.json
        train2017/
        val2017/
        test2017/
    Objects365/
        images/
        zhiyuan_objv2_train.json
    odinw/
       AerialMaritimeDrone/
       ...
       WildfireSmoke/
```



#### ``Objects365``
We found that the Objects365 v1 is unavailable now. Please try to download v2 as follows.

Download the [Objects365](https://www.objects365.org/overview.html) dataset from [YOLOv5](https://github.com/ultralytics/yolov5/blob/master/data/Objects365.yaml). 

You can also use custom datasets for modulated pre-training as long as they are in COCO format.


#### ``LVIS``
LVIS use the same images as COCO. Thus prepare the COCO images and annoations first and place them at ``DATASET/coco/``.

**All processed LVIS annotation files can be downloaded through:**

|train|minival|val 1.0|
|-----|-------|-------|
|[link](https://drive.google.com/file/d/1UpLRWfvXnGrRrhniKuiX_E1bkT90yZVE/view?usp=sharing)|[link](https://drive.google.com/file/d/1lLN9wole5yAsatFpYLnlnFEgcbDLXTfH/view?usp=sharing)|[link](https://drive.google.com/file/d/1BxlNOXEkcwsY2w2QuKdA2bdrrKCGv08J/view?usp=sharing)|

And place them at ``DATASET/coco/annotations/``.


**If you want to process by yourself rather than using the pre-processed files**, please follow the [instruction in GLIP](https://github.com/microsoft/GLIP/blob/main/DATA.md), summarized as following.

Download the following annotation files:
```
    wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/coco/annotations/lvis_v1_minival_inserted_image_name.json -O DATASET/coco/annotations/lvis_v1_minival_inserted_image_name.json

    wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/coco/annotations/lvis_od_val.json -O coco/annotations/lvis_od_val.json"
```
Also download the training set for extracting vision queries:
 ```   
    wget https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip -O coco/annotations/lvis_v1_train.json.zip
```
Unpack the .zip file to ``coco/annotations/lvis_v1_train.json``, and convert it to coco format:
```

python utils/add_file_name.py
```



#### ``Object Detection in the Wild (ODinW)``

**Download ODinW**
```
python odinw/download_datasets.py
```

``configs/odinw_35`` contain all the meta information of the datasets. ``configs/odinw_13`` are the datasets used by GLIP. Each dataset follows the coco detection format.

Please refer to [GLIP](https://github.com/microsoft/GLIP/tree/main) for more details.