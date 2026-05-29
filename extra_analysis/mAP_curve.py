import json
import matplotlib.pyplot as plt

log_path = '/data/fxy_projects/mmrazor-main/work_dirs/distillation_uod/20260526_215937/vis_data/scalars.json'
# 修改1: log_path 定位到对应的日志 .json 文件上
maps = []
epochs = []

with open(log_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        if 'coco/bbox_mAP' in data:
            maps.append(data['coco/bbox_mAP'])
            epochs.append(data['step']) # 或者用 data['epoch']

plt.plot(epochs, maps)
plt.title('mAP Curve')
plt.xlabel('Epoch')
plt.ylabel('mAP')
plt.savefig('/data/fxy_projects/mmdetection-main/graduate_result/fcos_distill_mAP.png')
# 修改2: 存储地址的文件名称依据模型修改 (例如: faster_rcnn, cascade_rcnn)