import json

log_path = '/data/fxy_projects/mmrazor-main/work_dirs/distillation_uod/20260526_215937/vis_data/scalars.json' # 替换为你的路径
best_map = {"val": -1, "epoch": -1}
best_map_s = {"val": -1, "epoch": -1}

with open(log_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        # 寻找包含验证指标的记录
        if 'coco/bbox_mAP' in data:
            epoch = data['step'] # MMDetection 3.x 中 val 记录的 step 通常对应 epoch
            
            if data['coco/bbox_mAP'] > best_map['val']:
                best_map = {"val": data['coco/bbox_mAP'], "epoch": epoch}
            
            if data['coco/bbox_mAP_s'] > best_map_s['val']:
                best_map_s = {"val": data['coco/bbox_mAP_s'], "epoch": epoch}

print(f"最高 mAP: {best_map['val']:.4f} (出现在第 {best_map['epoch']} 轮)")
print(f"最高 mAP_s: {best_map_s['val']:.4f} (出现在第 {best_map_s['epoch']} 轮)")