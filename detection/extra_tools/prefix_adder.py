import torch

# 加载原算法的全量权重
checkpoint = torch.load("/data/fxy_projects/mmrazor-main/work_dirs/distillation_uod/epoch_12.pth", map_location='cpu')
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

new_state_dict={}

# 假设要去除 'module.' 前缀，可以这样操作
for k, v in state_dict.items():
    if k.startswith('architecture'):
        new_state_dict[k.replace('architecture.', '')] = v


# 写入文本文档
with open('/data/fxy_checkpoint_read/fcos_light_checkpoint.txt', 'w') as f:
    for key, value in new_state_dict.items():
        # 记录：层名 - 形状 - 数据类型
        line = f'Layer: {key} | Shape: {list(value.shape)} | ID: {value.dtype}\n'
        f.write(line)


checkpoint['state_dict'] = new_state_dict
# 或者保存修改后的 state_dict
torch.save(new_state_dict, '/data/fxy_projects/mmrazor-main/work_dirs/distillation_uod/fcos_student.pth')