import json
import os
from collections import Counter, defaultdict

ANNO_PATH = r"/data/fxy_datasets/underwater_bs/train_annotations_clean.json"
IMAGE_DIR = r"/data/fxy_datasets/underwater_bs/train/"

with open(ANNO_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

images = data['images']
annotations = data['annotations']
categories = data['categories']

img_id_to_info = {img['id']: img for img in images}
cat_id_to_name = {cat['id']: cat['name'] for cat in categories}
cat_name_to_id = {cat['name']: cat['id'] for cat in categories}

print("=" * 60)
print("Basic Info of the Dataset")
print("=" * 60)
print(f"Total Number of Images: {len(images)}")
print(f"Total Number of Annotations: {len(annotations)}")
print(f"Total Number of Categories: {len(categories)}")
print("\nCategory List:")
for cat in categories:
    print(f"  id={cat['id']:2d}, name='{cat['name']}'")

cat_counter = Counter()
for ann in annotations:
    cat_counter[cat_id_to_name[ann['category_id']]] += 1

print("\n" + "=" * 60)
print("The Number of Instances per Category")
print("=" * 60)
for name in [c['name'] for c in categories]:
    count = cat_counter.get(name, 0)
    print(f"  {name:25s}: {count:6d}")
print(f"\n  {'Total':25s}: {sum(cat_counter.values()):6d}")

img_ann_count = Counter()
img_has_ann = set()
for ann in annotations:
    img_ann_count[ann['image_id']] += 1
    img_has_ann.add(ann['image_id'])

imgs_without_ann = len(images) - len(img_has_ann)

print("\n" + "=" * 60)
print("Statistics of Image Annotation Quantity")
print("=" * 60)
if img_has_ann:
    counts = list(img_ann_count.values())
    print(f"Min: {min(counts)}")
    print(f"Max: {max(counts)}")
    avg_all = sum(counts) / len(images)
    avg_ann = sum(counts) / len(img_has_ann)
    print(f"Average: {avg_all:.2f}")
    print(f"Average (labeled images only): {avg_ann:.2f}")
else:
    print("No labels")

bins = defaultdict(int)
bins[0] = imgs_without_ann
for cnt in counts:
    if cnt == 1:
        bins[1] += 1
    elif cnt == 2:
        bins[2] += 1
    elif cnt == 3:
        bins[3] += 1
    elif cnt == 4:
        bins[4] += 1
    elif cnt == 5:
        bins[5] += 1
    elif 6 <= cnt <= 10:
        bins['6-10'] += 1
    elif cnt >= 11:
        bins['11+'] += 1

print("\nDistribution of Image Annotations:")
order = [0, 1, 2, 3, 4, 5, '6-10', '11+']
for key in order:
    print(f"  {str(key):5s}: {bins.get(key, 0):5d} images")

print("\n" + "=" * 60)
print("Image File Verification")
print("=" * 60)
missing_images = []
for img in images:
    file_path = os.path.join(IMAGE_DIR, img['file_name'])
    if not os.path.isfile(file_path):
        missing_images.append(img['file_name'])

if missing_images:
    print(f"Missing files: {len(missing_images)}")
    if len(missing_images) <= 20:
        for m in missing_images:
            print(f"  - {m}")
    else:
        for m in missing_images[:10]:
            print(f"  - {m}")
        print(f"  ... total {len(missing_images)} missing, please check.")
else:
    print("All image files exist ✓")

print("\n" + "=" * 60)
print("Number of Images per Category")
print("=" * 60)
cat_to_imgset = defaultdict(set)
for ann in annotations:
    cat_to_imgset[cat_id_to_name[ann['category_id']]].add(ann['image_id'])

for name in [c['name'] for c in categories]:
    imgs = cat_to_imgset.get(name, set())
    print(f"  {name:25s}: {len(imgs):6d} images")

print("\n" + "=" * 60)
print("Annotation Size Distribution (pixel area)")
print("=" * 60)

areas = []
rel_areas = []
for ann in annotations:
    img_info = img_id_to_info.get(ann['image_id'])
    if img_info is None:
        continue
    img_w = img_info['width']
    img_h = img_info['height']
    bbox = ann['bbox']
    w, h = bbox[2], bbox[3]
    area = w * h
    areas.append(area)
    rel_areas.append(area / (img_w * img_h))

import statistics
if areas:
    print(f"Bounding Box Pixel Area Statistics (sample count: {len(areas)}):")
    print(f"  Min: {min(areas):.0f}")
    print(f"  Max: {max(areas):.0f}")
    print(f"  Mean: {statistics.mean(areas):.0f}")
    print(f"  Median: {statistics.median(areas):.0f}")

    area_bins = {
        "0-1K": (0, 1000),
        "1K-5K": (1000, 5000),
        "5K-10K": (5000, 10000),
        "10K-50K": (10000, 50000),
        "50K-100K": (50000, 100000),
        "100K+": (100000, float('inf'))
    }
    area_counts = defaultdict(int)
    for a in areas:
        for label, (low, high) in area_bins.items():
            if low <= a < high:
                area_counts[label] += 1
                break
    print("\nPixel Area Interval Distribution:")
    for label in area_bins:
        print(f"  {label:10s}: {area_counts[label]:6d} ({area_counts[label]/len(areas)*100:.1f}%)")

    print("\n" + "=" * 60)
    print("Annotation Size Distribution (percentage of image)")
    print("=" * 60)
    print(f"  Min: {min(rel_areas)*100:.2f}%")
    print(f"  Max: {max(rel_areas)*100:.2f}%")
    print(f"  Mean: {statistics.mean(rel_areas)*100:.2f}%")
    print(f"  Median: {statistics.median(rel_areas)*100:.2f}%")

    rel_bins = {
        "0-1%": (0, 0.01),
        "1%-5%": (0.01, 0.05),
        "5%-10%": (0.05, 0.10),
        "10%-25%": (0.10, 0.25),
        "25%-50%": (0.25, 0.50),
        "50%+": (0.50, 1.0)
    }
    rel_counts = defaultdict(int)
    for r in rel_areas:
        for label, (low, high) in rel_bins.items():
            if low <= r < high:
                rel_counts[label] += 1
                break
    print("\nRelative Area Interval Distribution:")
    for label in rel_bins:
        print(f"  {label:10s}: {rel_counts[label]:6d} ({rel_counts[label]/len(areas)*100:.1f}%)")
else:
    print("No annotations available for calculation.")

print("\nAnalysis complete.")