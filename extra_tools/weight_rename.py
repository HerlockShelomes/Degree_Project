import torch

def add_prefix_to_checkpoint(input_path, output_path, prefix="teacher"):
    print(f"⏳ 正在加载原始权重: {input_path}")
    
    # 1. 加载权重文件到 CPU 内存
    checkpoint = torch.load(input_path, map_location='cpu')
    
    # 2. 兼容不同的预训练权重格式
    # 官方的 pth 通常是一个包含了 'state_dict', 'meta' 等信息的字典
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        is_dict_format = True
    else:
        # 有些简单的 pth 直接就是一个 state_dict
        state_dict = checkpoint
        is_dict_format = False

    new_state_dict = {}
    
    # 3. 遍历并修改所有的 Key
    for key, value in state_dict.items():
        # 安全防御：检查是否已经带有该前缀，避免变成 'teacher.teacher.xxx'
        if not key.startswith(prefix):
            new_key = prefix + '.' + key
        else:
            new_key = key
            
        new_state_dict[new_key] = value
        
    # 4. 将修改后的 state_dict 重新组装回去
    if is_dict_format:
        checkpoint['state_dict'] = new_state_dict
    else:
        checkpoint = new_state_dict
        
    # 写入文本文档
    with open('/data/fxy_checkpoint_read/fcos_improved_checkpoint.txt', 'w') as f:
        for key, value in new_state_dict.items():
            # 记录：层名 - 形状 - 数据类型
            line = f'Layer: {key} | Shape: {list(value.shape)} | ID: {value.dtype}\n'
            f.write(line)
            
    # 5. 保存为新的 pth 文件
    print(f"💾 正在保存新权重: {output_path}")
    torch.save(checkpoint, output_path)
    print("✅ 转换成功！")

# ==========================================
# 在这里填入你的文件路径
# ==========================================
if __name__ == '__main__':
    # 你从官方下载的、未修改的原始权重路径
    OLD_WEIGHT_PATH = '/data/fxy_projects/mmdetection-main/work_dirs/fcos_bs_deblur/best_coco_bbox_mAP_epoch_12.pth' 
    
    # 你希望保存的新权重路径
    NEW_WEIGHT_PATH = '/data/fxy_projects/mmdetection-main/work_dirs/fcos_bs_deblur/best_mAP_epoch_12_teacher.pth'
    
    add_prefix_to_checkpoint(OLD_WEIGHT_PATH, NEW_WEIGHT_PATH)