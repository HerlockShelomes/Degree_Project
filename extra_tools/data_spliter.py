import json
import random
import os
import shutil

def split_coco_json(json_path, image_dir, target_dir, train_ratio=0.8):
    os.makedirs(os.path.join(os.path.dirname(target_dir), os.path.dirname('train')), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(target_dir), os.path.dirname('test')), exist_ok=True)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    
    random.shuffle(images)
    num_train = int(len(images) * train_ratio)
    
    train_images = images[:num_train]
    test_images = images[num_train:]
    
    def get_anns(image_list):
        img_ids = {img['id'] for img in image_list}
        return [ann for ann in annotations if ann['image_id'] in img_ids]
    
    # 构建训练集JSON
    train_data = {
        'images':train_images,
        'annotations': get_anns(train_images),
        'categories': categories
    }
    
    test_data = {
        'images': test_images,
        'annotations': get_anns(test_images),
        'categories': categories
    }
    
    with open(os.path.join(target_dir, 'train_annotations.json'), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4)
    with open(os.path.join(target_dir, 'test_annotations.json'), 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4)
        
    def move_file_list(image_list, target):
        for img in image_list:
            file_name = img['file_name']
            src_path = os.path.join(image_dir, file_name)
            dst_path = os.path.join(target, file_name)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
            else:
                print(f"Warning: Picture {src_path} cannot find, skipped.")
    
    print("Moving Training Pictures...")
    move_file_list(train_images, os.path.join(target_dir, 'train'))
    print("Moving Testing Pictures...")
    move_file_list(test_images, os.path.join(target_dir, 'test'))
    
    print("Completed!")

def check_dataset_alignment(json_path, image_dir):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. 提取 JSON 中的数据
    json_image_dict = {img['id']: img['file_name'] for img in data['images']}
    json_image_filenames = set(json_image_dict.values())
    json_image_ids = set(json_image_dict.keys())
    
    ann_image_ids = {ann['image_id'] for ann in data['annotations']}
    
    # 2. 提取本地磁盘中的实际图片
    local_images = set(os.listdir(image_dir))

    print(f"--- 原始数据统计 ---")
    print(f"本地磁盘图片数: {len(local_images)}")
    print(f"JSON 中注册的图片数: {len(json_image_filenames)}")
    print(f"JSON 中含有的标注框总数: {len(data['annotations'])}")
    print(f"最大 Image ID: {max(ann_image_ids) if ann_image_ids else '无'}")
    print(f"最小 Image ID: {min(ann_image_ids) if ann_image_ids else '无'}")
    print(f"Image ID 长度: {len(ann_image_ids) if ann_image_ids else '无'}")
    print("-" * 20)

    # 3. 核心对齐检查
    # 错误 A：JSON 里有，但本地缺失的图片
    missing_local_files = json_image_filenames - local_images
    # 错误 B：本地有，但 JSON 里没注册的图片
    unregistered_local_files = local_images - json_image_filenames
    # 错误 C：标注框引用了不存在的 Image ID
    invalid_ann_ids = ann_image_ids - json_image_ids

    print(f"❌ 错误 A (有标注无图片): 有 {len(missing_local_files)} 张图在 JSON 中，但本地磁盘不存在。")
    print(f"❌ 错误 B (有图片无标注): 有 {len(unregistered_local_files)} 张本地图片，未在 JSON 中注册。")
    print(f"❌ 错误 C (死脑筋标注): 有 {len(invalid_ann_ids)} 个 image_id 在 annotations 中，但 images 列表里没有。")

    return missing_local_files, unregistered_local_files, invalid_ann_ids

def sanitize_coco_dataset(json_path, image_dir, output_json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    local_images = set(os.listdir(image_dir))
    
    # 1. 仅保留本地实际存在的图像结构
    clean_images = []
    valid_image_ids = set()
    
    for img in data['images']:
        if img['file_name'] in local_images:
            clean_images.append(img)
            valid_image_ids.add(img['id'])
            
    # 2. 仅保留属于这些有效图像的标注框
    clean_annotations = [
        ann for ann in data['annotations'] 
        if ann['image_id'] in valid_image_ids
    ]
    
    # 重置 ID 的核心逻辑示例
    id_mapping = {}
    for new_id, img in enumerate(clean_images, start=1):
        id_mapping[img['id']] = new_id
        img['id'] = new_id # 更新 image 里面的 id

    for ann in clean_annotations:
        ann['image_id'] = id_mapping[ann['image_id']] # 更新 annotation 里面的 image_id
        # 同时也可以顺便重置一下 annotation 自身的 id

    # 3. 组装清洗后的新数据
    clean_data = {
        "images": clean_images,
        "annotations": clean_annotations,
        "categories": data['categories'] # 类别通常保持不变
    }
    
    # 4. 保存新文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, indent=4)
        
    print(f"\n清洗完成！已将完美对齐的数据集保存至: {output_json_path}")
    print(f"新数据集包含图片: {len(clean_images)} 张，标注框: {len(clean_annotations)} 个。")
    


def remove_duplicate_bbox_annotations(json_path, output_json_path):
    print(f"正在读取并清洗文件: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    annotations = data.get('annotations', [])
    original_count = len(annotations)
    
    clean_annotations = []
    # 用于记录已经处理过的标注特征，结构为: (image_id, category_id, (x, y, w, h))
    seen_annotations = set()
    duplicate_count = 0

    for ann in annotations:
        img_id = ann['image_id']
        cat_id = ann['category_id']
        # 将 bbox 列表转换为元组，方便进行哈希比较
        bbox_tuple = tuple(round(coord, 1) for coord in ann['bbox'])
        
        # 组合成一个唯一的特征标识
        ann_feature = (img_id, cat_id, bbox_tuple)
        
        if ann_feature in seen_annotations:
            # 如果这个特征已经存在，说明是完全重复的框，选择丢弃
            print(f"Index {ann['id']} Annotation: Replicate, Drop.")
            duplicate_count += 1
            continue
        else:
            # 如果是第一次见到，记录特征并保留
            seen_annotations.add(ann_feature)
            clean_annotations.append(ann)

    # 核心修正：为清洗后的标注重新分配连续且唯一的全局 ID
    for new_id, ann in enumerate(clean_annotations, start=1):
        ann['id'] = new_id

    # 重新组装完整的 COCO 数据
    cleaned_data = {
        "images": data.get('images', []),
        "annotations": clean_annotations,
        "categories": data.get('categories', [])
    }

    # 写入新文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=4)

    # 打印清洗报告
    print("\n--- 数据清洗报告 ---")
    print(f"原始标注框总数: {original_count} 个")
    print(f"❌ 发现并剔除的完全重复框: {duplicate_count} 个")
    print(f"✅ 最终保留的有效标注框: {len(clean_annotations)} 个")
    print(f"新文件已安全保存至: {output_json_path}")
    


import json
import os
import shutil

def find_and_remove_unannotated_images(json_path, image_dir, mode='dry_run', backup_dir=None):
    """
    找出未被 JSON 标注的本地图片并进行处理。
    
    参数:
    - json_path: 你的 COCO 格式 JSON 文件路径
    - image_dir: 本地存放图片的文件夹路径
    - mode: 处理模式。
            'dry_run' (默认): 只打印不合规的图片列表，不删除任何文件（最安全）。
            'move': 将未标注的图片移动到 backup_dir 目录中（推荐，方便检查）。
            'delete': 直接从磁盘上彻底删除（请务必确认后使用）。
    - backup_dir: 当 mode='move' 时，图片被移入的备份目录。
    """
    
    # 1. 读取 JSON 标注文件
    print(f"正在读取标注文件: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 2. 提取 JSON 中所有注册过的图片文件名（转换为 set 提高查找速度）
    registered_images = {img['file_name'] for img in data.get('images', [])}
    print(f"JSON 文件中注册的图片总数: {len(registered_images)}")
    
    # 3. 扫描本地磁盘中的实际图片文件
    local_images = set(os.listdir(image_dir))
    print(f"本地磁盘中的图片总数: {len(local_images)}")
    
    # 4. 求差集：找出本地存在、但 JSON 中没有的图片
    unannotated_images = local_images - registered_images
    print(f"⚠️ 发现未标注（未注册）的图片数量: {len(unannotated_images)}")
    
    if not unannotated_images:
        print("🎉 太棒了！本地所有图片都在 JSON 中有标注记录，无需清理。")
        return

    # 5. 根据模式执行具体的清理操作
    if mode == 'dry_run':
        print("\n[当前为 dry_run 预览模式，未对文件做任何改动]")
        print("部分未标注图片示例（前10张）:")
        for img_name in list(unannotated_images)[:10]:
            print(f"  - {img_name}")
            
    elif mode == 'move':
        if not backup_dir:
            print("❌ 错误：选择 'move' 模式时必须指定 backup_dir 参数！")
            return
        os.makedirs(backup_dir, exist_ok=True)
        print(f"\n[移动模式] 正在将未标注图片移至: {backup_dir}")
        for img_name in unannotated_images:
            src_path = os.path.join(image_dir, img_name)
            dst_path = os.path.join(backup_dir, img_name)
            if os.path.exists(src_path):
                shutil.move(src_path, dst_path)
        print(f"✅ 成功移动 {len(unannotated_images)} 张图片。你可以去备份目录检查它们是否真的不需要。")
        
    elif mode == 'delete':
        # 安全二次确认
        confirm = input(f"\n❗ 警告 ❗ 您选择了直接删除。确认要永久删除这 {len(unannotated_images)} 张图片吗？(yes/no): ")
        if confirm.lower() == 'yes':
            print("\n[删除模式] 正在彻底从磁盘删除文件...")
            for img_name in unannotated_images:
                file_path = os.path.join(image_dir, img_name)
                if os.path.exists(file_path):
                    os.remove(file_path)
            print(f"🗑️ 已成功从磁盘删除 {len(unannotated_images)} 张未标注图片。")
        else:
            print("❌ 操作已取消，未删除任何文件。")

# ================== 使用示例 ==================
JSON_FILE = '/data/fxy_datasets/underwater_bs/train_annotations.json'
IMAGE_FOLDER = '/data/fxy_datasets/underwater_bs/train_images'
BACKUP_FOLDER = '/data/fxy_datasets/underwater_bs/unannotated_images_bak'

# 步骤 1：先运行 dry_run 看看有哪些图片会被影响（安全第一）
find_and_remove_unannotated_images(JSON_FILE, IMAGE_FOLDER, mode='dry_run')

# 步骤 2：如果确认这些图片确实是垃圾数据，建议先用 'move' 模式移走
# find_and_remove_unannotated_images(JSON_FILE, IMAGE_FOLDER, mode='move', backup_dir=BACKUP_FOLDER)

# 步骤 3：如果你百分之百确定，可以直接用 'delete' 模式（会有命令行二次确认）
# find_and_remove_unannotated_images(JSON_FILE, IMAGE_FOLDER, mode='delete')


# # 运行检查
missing_files, unreg_files, invalid_anns = check_dataset_alignment('/data/fxy_datasets/underwater_bs/train_annotations_clean.json', '/data/fxy_datasets/underwater_bs/train/')    
missing_files, unreg_files, invalid_anns = check_dataset_alignment('/data/fxy_datasets/underwater_bs/test_annotations_clean.json', '/data/fxy_datasets/underwater_bs/test/') 
# # 运行清洗
# sanitize_coco_dataset('/data/fxy_datasets/Merged/merged_dataset_final.json', '/data/fxy_datasets/Merged/merged_images/', '/data/fxy_datasets/Merged/merged_final.json')


# split_coco_json(
#     json_path='/data/fxy_datasets/Merged/merged_final.json',
#     image_dir='/data/fxy_datasets/Merged/merged_images/',
#     target_dir='/data/fxy_datasets/underwater_bs/',
#     train_ratio=0.8)

# ================== 执行去重清洗 ==================
# remove_duplicate_bbox_annotations(
#     json_path='/data/fxy_datasets/underwater_bs/train_annotations.json',
#     output_json_path='/data/fxy_datasets/underwater_bs/train_annotations_clean.json'
# )