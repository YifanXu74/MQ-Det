import json

ann_file_path = 'DATASET/coco/annotations/lvis_v1_train.json'
output_path = 'DATASET/coco/annotations/lvis_od_train.json'

ann_file = json.load(open(ann_file_path, 'r'))

for image in ann_file['images']:
    image['file_name'] = image['coco_url'].split('http://images.cocodataset.org/')[-1]
for ann in ann_file['annotations']:
    ann['iscrowd'] = 0

json.dump(ann_file, open(output_path, 'w'))


