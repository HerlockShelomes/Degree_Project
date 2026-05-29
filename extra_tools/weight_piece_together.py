import torch

# 1. 加载各个权重文件
backbone_dict = torch.load('/data/fxy_checkpoints/fcos_cspnext_backbone.pth', map_location='cpu')
head_dict = torch.load('/data/fxy_checkpoints/fcos_no_backbone.pth', map_location='cpu')

if 'state_dict' in backbone_dict:
    backbone_dict = backbone_dict['state_dict']
else:
    backbone_dict = backbone_dict
    
if 'state_dict' in head_dict:
    head_dict = head_dict['state_dict']
else:
    head_dict = head_dict

# 2. 为每个组件添加对应的前缀
new_state_dict = {}
for k, v in backbone_dict.items():
    new_state_dict[k] = v   # 添加 backbone. 前缀

for k, v in head_dict.items():
    new_state_dict[k] = v       # 添加 head. 前缀

# 如果有更多组件（如 neck），继续添加
# neck_dict = torch.load('neck.pth', ...)
# for k, v in neck_dict.items():
#     new_state_dict[f'neck.{k}'] = v

# 3. 保存合并后的权重
checkpoint = {}
checkpoint['state_dict'] = new_state_dict

with open('/data/fxy_checkpoint_read/fcos_cspnext_improved_checkpoint.txt', 'w') as f:
    for key, value in new_state_dict.items():
        # 记录：层名 - 形状 - 数据类型
        line = f'Layer: {key} | Shape: {list(value.shape)} | ID: {value.dtype}\n'
        f.write(line)
        
torch.save(checkpoint, '/data/fxy_checkpoints/fcos_cspnext.pth')