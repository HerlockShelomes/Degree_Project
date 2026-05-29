import torch

file_path = "/data/fxy_projects/mmdetection-main/work_dirs/"
# 加载原算法的全量权重
checkpoint = torch.load(file_path + "fcos_bs_deblur/best_coco_bbox_mAP_epoch_12.pth", map_location='cpu')
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

# 写入文本文档
with open('/data/fxy_checkpoint_read/fcos_improved_checkpoint.txt', 'w') as f:
    for key, value in state_dict.items():
        # 记录：层名 - 形状 - 数据类型
        line = f'Layer: {key} | Shape: {list(value.shape)} | ID: {value.dtype}\n'
        f.write(line)

# 创建新的字典，只保留 neck 和 head 部分的相关参数
new_state_dict = {}
for k, v in state_dict.items():
    # 过滤掉所有以 backbone 开头的键
    if not k.startswith('backbone'):
        new_state_dict[k] = v

# 保存更新处理后的权重
# checkpoint['state_dict'] = new_state_dict
# torch.save(checkpoint, file_path + 'fcos_no_backbone.pth')