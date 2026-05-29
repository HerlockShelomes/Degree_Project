import os
from glob import glob
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from fire import Fire
from tqdm import tqdm

from aug import get_normalize
from models.networks import get_generator

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:128"

class Predictor:
    def __init__(self, weights_path: str, model_name: str = ''):
        with open('config/config.yaml',encoding='utf-8') as cfg:
            config = yaml.load(cfg, Loader=yaml.FullLoader)
        model = get_generator(model_name or config['model'])
        model.load_state_dict(torch.load(weights_path)['model'])
        self.model = model.cuda()
        self.model.train(True)
        # GAN inference should be in train mode to use actual stats in norm layers,
        # it's not a bug
        self.normalize_fn = get_normalize()

    @staticmethod
    def _array_to_batch(x):
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return torch.from_numpy(x)

    def _preprocess(self, x: np.ndarray, mask: Optional[np.ndarray]):
        x, _ = self.normalize_fn(x, x)
        if mask is None:
            mask = np.ones_like(x, dtype=np.float32)
        else:
            mask = np.round(mask.astype('float32') / 255)

        h, w, _ = x.shape
        block_size = 32
        min_height = (h // block_size + 1) * block_size
        min_width = (w // block_size + 1) * block_size

        pad_params = {'mode': 'constant',
                      'constant_values': 0,
                      'pad_width': ((0, min_height - h), (0, min_width - w), (0, 0))
                      }
        x = np.pad(x, **pad_params)
        mask = np.pad(mask, **pad_params)

        return map(self._array_to_batch, (x, mask)), h, w

    @staticmethod
    def _postprocess(x: torch.Tensor) -> np.ndarray:
        x, = x
        x = x.detach().cpu().float().numpy()
        x = (np.transpose(x, (1, 2, 0)) + 1) / 2.0 * 255.0
        return x.astype('uint8')

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray], ignore_mask=True) -> np.ndarray:
        (img, mask), h, w = self._preprocess(img, mask)
        with torch.no_grad():
            inputs = [img.cuda()]
            if not ignore_mask:
                inputs += [mask]
            pred = self.model(*inputs)
        return self._postprocess(pred)[:h, :w, :]

def process_video(pairs, predictor, output_dir):
    for video_filepath, mask in tqdm(pairs):
        video_filename = os.path.basename(video_filepath)
        output_filepath = os.path.join(output_dir, os.path.splitext(video_filename)[0]+'_deblur.mp4')
        video_in = cv2.VideoCapture(video_filepath)
        fps = video_in.get(cv2.CAP_PROP_FPS)
        width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frame_num = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
        video_out = cv2.VideoWriter(output_filepath, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))
        tqdm.write(f'process {video_filepath} to {output_filepath}, {fps}fps, resolution: {width}x{height}')
        for frame_num in tqdm(range(total_frame_num), desc=video_filename):
            res, img = video_in.read()
            if not res:
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred = predictor(img, mask)
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            video_out.write(pred)

def main(img_pattern: str,
         mask_pattern: Optional[str] = None,
         weights_path='fpn_inception.h5',
         out_dir='submit/',
         side_by_side: bool = False,
         video: bool = False):
    def sorted_glob(pattern):
        return sorted(glob(pattern))

    all_imgs = sorted_glob(img_pattern)
    all_masks = sorted_glob(mask_pattern) if mask_pattern is not None else [None for _ in all_imgs]

    os.makedirs(out_dir, exist_ok=True)

    imgs = []
    masks = []
    names = []

    for img_path, mask_path in zip(all_imgs, all_masks):
        name = os.path.basename(img_path)
        target_output = os.path.join(out_dir, name)

        # print(img_path)
        # print(mask_path)
        # print(name)

        if os.path.exists(target_output):
            continue

        imgs.append(img_path)
        masks.append(mask_path)
        names.append(name)

    pairs = zip(imgs, masks)

    total_all = len(all_imgs)
    total_remain = len(imgs)

    print(f"Detected: {total_all} pictures, Remaining: {total_remain} pictures.")

    
    predictor = Predictor(weights_path=weights_path)

    os.makedirs(out_dir, exist_ok=True)
    if not video:
        for name, pair in tqdm(zip(names, pairs), total=len(names)):
            f_img, f_mask = pair
            img, mask = map(cv2.imread, (f_img, f_mask))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w = img.shape[:2]
            max_side = 1280
            scale_factor = 1.0

            if max(h, w) > max_side:
                scale_factor = max_side/max(h, w)
                new_w = int(w*scale_factor)
                new_h = int(h*scale_factor)

                img = cv2.resize(img, (new_w, new_h),  interpolation=cv2.INTER_AREA)
                if mask is not None:
                    mask = cv2.resize(mask, (new_w, new_h),  interpolation=cv2.INTER_NEAREST)

            pred = predictor(img, mask)

            if scale_factor != 1.0:
                pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_CUBIC)
                if side_by_side:
                    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

            if side_by_side:
                pred = np.hstack((img, pred))
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_dir, name),
                        pred)
            torch.cuda.empty_cache()
    else:
        process_video(pairs, predictor, out_dir)

# def getfiles():
#     filenames = os.listdir(r'.\dataset1\blur')
#     print(filenames)
def get_files():
    list=[]
    for filepath,dirnames,filenames in os.walk(r'.\dataset1\blur'):
        for filename in filenames:
            list.append(os.path.join(filepath,filename))
    return list

if __name__ == '__main__':
#   #  Fire(main)
# #增加批量处理图片：
#     img_path=get_files()
#     for i in img_path:
#         main(i)
#     # main('test_img/tt.mp4')
    input_pattern = r'/data/fxy_datasets/underwater_bs/train/*'
    
    output_directory = r'/data/fxy_datasets/underwater_bs_deblur/train'
    
    # input_pattern = r'/data/fxy_projects/DeblurGANv2-master/submit/*'
    
    # output_directory = r'/data/fxy_projects/DeblurGANv2-master/output_results'

    main(
        img_pattern=input_pattern,
        out_dir=output_directory,
        weights_path='/data/fxy_checkpoints/fpn_inception.h5',
        side_by_side=False
    )
