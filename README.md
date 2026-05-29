# Degree Project

## Basic Components

- **Dataset**: `underwater_bs.zip/underwater_bs_deblur.zip`
- **Project**: `MMDetection/MMRazor/MMDeploy/DeblurGANv2`
  - `MMDetection`: https://github.com/open-mmlab/mmdetection
  - `MMRazor`: https://github.com/open-mmlab/mmrazor
  - `MMDeploy`: https://github.com/open-mmlab/mmdeploy
  - `DeblurGANv2`: https://github.com/VITA-Group/DeblurGANv2
- **Hardware**: 6 of `NVIDIA TITAN Xp` at one remote server.

## Installation

```bash
# Create the environment
conda create --name openmmlab python=3.8 -y
conda activate openmmlab

# If you are using GPUs
conda install pytorch torchvision -c pytorch

# If you are using CPU
conda install pytorch torchvision cpuonly -c pytorch

# Install the dependencies
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

# Install MMDetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
# -e: editable mode, any modification to the code will come into effect automatically without reinstall the mmdetection.

# Validation of installation
python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cpu

# Install MMRazor
git clone https://github.com/open-mmlab/mmrazor.git
cd mmrazor
pip install -v -e .
# -e: editable mode, any modification to the code will come into effect automatically without reinstall the mmdetection.

# Install MMDeploy in Linux-x86_64

# 1. install MMDeploy model converter
pip install mmdeploy==1.3.1

# 2. install MMDeploy sdk inference
# you can install one to install according whether you need gpu inference
# 2.1 support onnxruntime
pip install mmdeploy-runtime==1.3.1
# 2.2 support onnxruntime-gpu, tensorrt
pip install mmdeploy-runtime-gpu==1.3.1

# 3. install inference engine
# 3.1 install TensorRT
# !!! If you want to convert a tensorrt model or inference with tensorrt,
# download TensorRT-8.2.3.0 CUDA 11.x tar package from NVIDIA, and extract it to the current directory
pip install TensorRT-8.2.3.0/python/tensorrt-8.2.3.0-cp38-none-linux_x86_64.whl
pip install pycuda
export TENSORRT_DIR=$(pwd)/TensorRT-8.2.3.0
export LD_LIBRARY_PATH=${TENSORRT_DIR}/lib:$LD_LIBRARY_PATH
# !!! Moreover, download cuDNN 8.2.1 CUDA 11.x tar package from NVIDIA, and extract it to the current directory
export CUDNN_DIR=$(pwd)/cuda
export LD_LIBRARY_PATH=$CUDNN_DIR/lib64:$LD_LIBRARY_PATH

# 3.2 install ONNX Runtime
# you can install one to install according whether you need gpu inference
# 3.2.1 onnxruntime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz
tar -zxvf onnxruntime-linux-x64-1.8.1.tgz
export ONNXRUNTIME_DIR=$(pwd)/onnxruntime-linux-x64-1.8.1
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
# 3.2.2 onnxruntime-gpu
pip install onnxruntime-gpu==1.8.1
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-gpu-1.8.1.tgz
tar -zxvf onnxruntime-linux-x64-gpu-1.8.1.tgz
export ONNXRUNTIME_DIR=$(pwd)/onnxruntime-linux-x64-gpu-1.8.1
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH


git clone -b main https://github.com/open-mmlab/mmdeploy.git
cd mmdeploy
pip install -v -e .
# -e: editable mode, any modification to the code will come into effect automatically without reinstall the mmdetection.

# DeblurGANv2 installation
git clone https://github.com/VITA-Group/DeblurGANv2.git

# For other requirements for this environment, you could refer to the file requirements.txt
```

If you have finished all steps above, then the project of mmdetection has been installed successfully.

> [!CAUTION]
>
> If you are **seeing errors** related to **the version of mmcv**, it **does not indicate** that you have installed wrong version of mmcv. Instead, you could fix this by commenting out the following code in `/mmdetection-main/mmdet/init.py`

```python
# assert (mmcv_version >= digit_version(mmcv_minimum_version)
#         and mmcv_version < digit_version(mmcv_maximum_version)), \
#     f'MMCV=={mmcv.__version__} is used but incompatible. ' \
#     f'Please install mmcv>={mmcv_minimum_version}, <{mmcv_maximum_version}.'
```

## Usage

The main structure of the whole project is like below (**<u>only the folders that will be used are listed</u>**):

```python
- Dataset (Two datasets share similar structuress)
  # This folder is the training and test data utilized
  - test
  - train
  - test_annotations_clean.json
  - train_annotations_clean.json
- mmdetection
  # This folder is for the mmdetection project
  - tools
    - extra_analysis
    - extra_tools
  - work_dirs
- mmdeploy
  # This folder is for the mmdeploy project
- mmrazor
  # This folder is for the mmrazor project
  - work_dirs

- DeblurGANv2
  # This folder is for the mmdetection project. The two files are from 'deblur' folder.
  - predict.py
  - predict_test.py
```

There are several modifications that you must complete before training or testing.

```python
# First, find the folder work_dirs/ , which stores all config files and training results.
# You shall see all models involved in this project. Select one and enter its folder.
# Then you could see one config file, one weight file and one folder with training date.

# We have to alter some parameters within each config file
# Main two parts: 1. The path to load data: data root；2. The path to load the weight: load_from
data_root = 'path/to/FER/'

load_from = "path/to/the/checkpoint/weight.pth"
# This file could be the weight within the same folder, or the weight file from the folder checkpoints
# The former one is the fine-tuned weight for FER, and the latter one is a pretrained-weight for fine-tune.

# There are usually 3 dataloaders (train_dataloader, test_dataloader, val_dataloader)
..._dataloader = dict(
	...
	ann_file='based on data_root, locate to corresponding annotation file(.json)',
	data_prefix=dict(img='based on data_root, locate to corresponding image folder'),
	data_root=data_root,
	...
)

# example
..._dataloader = dict(
	...
	ann_file='test_annotations.coco.json',
	data_prefix=dict(img='test/'),
	data_root=data_root,
	...
)

# There are usually two evaluators (test_evaluator, val_evaluator)
..._evaluator = dict(
	...
	ann_file='path/to/your/test/annotation/.json/file',
	...
)

# work_dir alternation.
work_dir = "/path/to/current/model/folder"
```

### Train

If you want to train right from the start, you should...

```python
# Find the resume parameter, set it to False.
# So the program will not try to restore the previous training state.
resume = False

# Load the weight from the pretrained weight, for example
load_from = '/data/checkpoints/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth'
```

If you want to train more or less epochs...

```python
# max_epochs: parameter used to control the number of training epochs
# val_interval: parameter used to control the validation interval.
#               5 means that after each 5 turns of training, there will be one round of measurement of 
#               model's current performance.
train_cfg = dict(
    dynamic_intervals=[
        (
            280,
            1,
        ),
    ],
    max_epochs=100, # This means that it will be trained for 100 epochs
    type='EpochBasedTrainLoop',
    val_interval=5  # This means that the model's performance will be measured for every 5 epochs)
```

If your GPU is out of memory...

```python
# You could save some GPU memory space by decreasing the batch_size,
# and use Amp + accumulative gradient to achieve the effect of original training.

# If the original batch_size is...
train_dataloader = dict(
    ...
    batch_size=16,
    ...
    )

# Then we could...
train_dataloader = dict(
    ...
    batch_size=1,
    ...
    )

optim_wrapper = dict(
	type='AmpOptimWrapper',
    accumulative_counts=8,
	)

# and run the training on 2 GPUs.


auto_scale_lr = dict(base_batch_size=16, enable=False)
# We'd better turn off the function of adjusting learning rate based on batch_size, since we have modified other parameters to simulate the original training batch_size
```

The altering of batch_size must follow the rule:
$$
\text{Effective Batch Size} = \text{batch size} \times \text{gpu count} \times \text{accumulative counts}
$$
For the above example, it is
$$
16 = 1 \times 2 \times 8
$$

```bash
# You could use this command to find out your available GPU(s)
nvidia-smi
# The returned list shall show the GPUs along with their index.
Sat Apr 25 11:11:02 2026       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.65.01    Driver Version: 515.65.01    CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA TITAN X ...  Off  | 00000000:04:00.0 Off |                  N/A |
| 23%   19C    P8     8W / 250W |      0MiB / 12288MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA TITAN Xp     Off  | 00000000:05:00.0 Off |                  N/A |
| 23%   22C    P8     8W / 250W |      0MiB / 12288MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   2  NVIDIA TITAN Xp     Off  | 00000000:08:00.0 Off |                  N/A |
| 23%   22C    P8     9W / 250W |      0MiB / 12288MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   3  NVIDIA TITAN Xp     Off  | 00000000:09:00.0 Off |                  N/A |
| 51%   80C    P2   171W / 250W |   9505MiB / 12288MiB |     99%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   4  NVIDIA TITAN Xp     Off  | 00000000:84:00.0 Off |                  N/A |
| 28%   49C    P2    62W / 250W |   4955MiB / 12288MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   5  NVIDIA TITAN Xp     Off  | 00000000:85:00.0 Off |                  N/A |
| 31%   51C    P2    62W / 250W |   4427MiB / 12288MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   6  NVIDIA TITAN Xp     Off  | 00000000:88:00.0 Off |                  N/A |
| 44%   69C    P2   224W / 250W |   9505MiB / 12288MiB |     99%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    3   N/A  N/A     20437      C   ...envs/openmmlab/bin/python     9503MiB |
|    6   N/A  N/A     20438      C   ...envs/openmmlab/bin/python     9503MiB |
+-----------------------------------------------------------------------------+

# To start the training process, first, you have to be in the mmdetection folder,
# and ensure that the 'openmmlab' environment has been activated.

# Then you could start the training process with the following command

CUDA_VISIBLE_DEVICES=0,1 PORT=29500 ./tools/dist_train.sh ${Config File Path} 2

# Example:
CUDA_VISIBLE_DEVICES=0,1 PORT=29500 ./tools/dist_train.sh /home/xyfang/projects/mmdetection-main/work_dirs/cascade-rcnn_r50_fpn_1x_coco/cascade-rcnn_r50_fpn_1x_coco.py 2
# This command means that you will be using 2 GPUs(GPU 0 and GPU 1) for the training process.
# The model Cascade RCNN will be trained.
# Before execute the command, make sure that the port for training is not be occupied.
```

You could adjust the `CUDA_VISIBLE_DEVICES` and the integer at the end according to GPU(s) available.

There is **one problem unsolved**. Although I have assigned GPU 0 and GPU 1 in above command, the actual GPUs used are GPU 1 and GPU 2. To express more clearly,
$$
\text{Actual Used GPU} = \text{Assigned GPU} + 1
$$

### Test

Once the training is finished, you could use multiple metrics for result analysis.

Most tests are based on the training log file. Its relative path is like:

`dino-4scale_r50_8xb2-12e_coco/20260423_104008/vis_data/scalars.json`

#### Loss

```bash
python tools/analysis_tools/analyze_logs.py plot_curve ${scalar.json Path} --keys loss --legend loss --out ${Output Path}

# Example:
python tools/analysis_tools/analyze_logs.py plot_curve /data/fxy_projects/mmdetection-main/work_dirs/dino-4scale_r50_8xb2-12e_coco/20260423_104008/vis_data/scalars.json --keys loss --legend loss --out /data/fxy_projects/mmdetection-main/result_curves/dino_fer_loss.png
# Draw the loss curve for DINO, store it as dino_fer_loss.png in result_curves folder.
```

You could use the `analyze_logs.py` to draw the **Loss Curve** during training process to see if the model converges at last.

#### mAP_50

To analyze model's mAP_50, there are two python scripts in `tools/extra_analysis/` folder.

One for **mAP_50 curve generation**, the other is to **extract the greatest mAP and mAP_50** of all epochs evaluated during training. The instructions to use these two scripts have been written into the two files.

#### FPS/Memory Usage

```bash
CUDA_VISIBLE_DEVICES=1 python tools/analysis_tools/benchmark.py ${Config File Path} --checkpoint ${Checkpoint Path} --repeat-num 1 --task inference

# Example:
CUDA_VISIBLE_DEVICES=1 python tools/analysis_tools/benchmark.py /data/fxy_projects/mmdetection-main/work_dirs/dino-4scale_r50_8xb2-12e_coco/dino-4scale_r50_8xb2-12e_coco.py --checkpoint /data/fxy_projects/mmdetection-main/work_dirs/dino-4scale_r50_8xb2-12e_coco/epoch_24.pth --repeat-num 1 --task inference
# Load the weight file from 24th epoch for DINO, then use GPU 1 to run benchmark.py
# '--repeat-num' defines how many times the test will be executed.
# Repeated tests could make the measurement more accurate,
# since the extreme values will cause less effect.
# '--task inference' indicates that the test measures model's time consumption to output the prediction.
```

### Deblur

Using DeblurGANv2 for image deblurring: You have to update the path to the dataset.

```python
# First modify the path for input and output in 'predict' file, then run the command below.
python DeblurGANv2/predict.py
```

### Distillation

First move into the mmrazor folder, then configure the file just like files in MMDetection. Run the command like below

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 --master_port=28740 tools/train.py ${Config File Path} --launcher pytorch

# Example:
CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 --master_port=28740 tools/train.py /data/fxy_projects/mmrazor-main/work_dirs/distillation_uod/config_file.py --launcher pytorch
```

### Deploy

Move into the mmdeploy folder, then run the command like below.

```bash
CUDA_VISIBLE_DEVICES=4 PORT=28732 python tools/deploy.py configs/mmdet/detection/detection_onnxruntime_dynamic.py ${Config File Path} ${Checkpoint Path} /data/fxy_projects/mmdetection-main/demo/demo.jpg --work-dir your/work/dir --device cuda --dump-info

# Example:
CUDA_VISIBLE_DEVICES=4 PORT=28732 python tools/deploy.py configs/mmdet/detection/detection_onnxruntime_dynamic.py /data/fxy_projects/mmdetection-main/work_dirs/fcos_bs_light/fcos_cspnext_fpn_gn-head_1x_coco.py /data/fxy_projects/mmdetection-main/work_dirs/fcos_bs_light/best_coco_bbox_mAP_epoch_12.pth /data/fxy_projects/mmdetection-main/demo/demo.jpg --work-dir mmdeploy_model/fcos_light --device cuda --dump-info
```





## Environment Setup

- Python 3.8+
- Install required packages: `pip install onnxruntime opencv-python numpy flask flask-socketio psutil matplotlib pandas`
- (Optional) Docker for resource‑constrained simulation.

## Repository Structure

Edge_Deployment_Simulation_and_Visualization_System/
├── src/
│ ├──app.py # Flask visualization server
│ ├── detector.py # ONNX model inference wrapper (FCOSDetector)
│ ├── tracker.py # IoU‑based multi‑object tracker (SimpleTracker)
│ ├── eval_edge.py # Benchmarking script for edge deployment evaluation
│ └── templates/
│  └── index.html # Web dashboard
├── classes.txt # Class names (one per line)
├── represent_images/ # Example images for the visualization demo
├── test_images/
├── Dockerfile.x86 # Docker image for x86 simulation
├── end2end.onnx
├── end2end_light.onnx
├── end2end_distill.onnx
└── README.md

### All commands are executed by default in the path \Edge_Deployment_Simulation_and_Visualization_System

## Model Performance Evaluation (eval_edge.py)

This script benchmarks an ONNX model and reports end‑to‑end latency, pure inference latency, CPU utilisation, and peak memory.

### Basic Usage (Local, Unconstrained)

`python src\eval_edge.py --model end2end_distill.onnx --img_dir test_images --runs 100`

Parameters:

| Argument    | Default              | Description                        |
| ----------- | -------------------- | ---------------------------------- |
| `--model`   | `end2end_light.onnx` | Path to ONNX model                 |
| `--img_dir` | `None`               | Folder with test images (optional) |
| `--runs`    | `50`                 | Number of inference iterations     |

If `--img_dir` is omitted, random noise is used as input.


### Docker‑based Edge Simulation

Build the x86 image: `docker build -f Dockerfile.x86 -t fcos-edge-x86 .`

Run with resource limits (e.g., 6 cores, 1 GB RAM) and mount result/output directories: `docker run --cpus=6 --memory=1g --rm -v %cd%\results:/app/results -v %cd%\test_images:/app/test_images fcos-edge-x86 python3 eval_edge.py --model end2end_distill.onnx --img_dir /app/test_images --runs 50`

For ARM64 emulation, build and run similarly using `Dockerfile.arm` with `--platform linux/arm64`.

**Note:** On Windows, replace `%cd%` with `%CD%` in PowerShell or use absolute paths.

## Real‑Time Visualization System

A Flask‑based web application that loads an ONNX model and displays detection results on a sequence of images, simulating a live underwater feed.

### Configuration

Edit the following constants in `app.py`:

`MODEL_PATH = 'end2end.onnx'`  
`CLASSES_PATH = 'classes.txt'`  
`IMAGE_FOLDER = 'represent_images'`  
`IMAGE_EXTS = ['*.jpg', '*.png', '*.jpeg']`  
`FRAME_INTERVAL = 1   # seconds per image`

The default alarm classes can be customised through the web UI (see below).

### Starting the Server

`python src\app.py`

Open `http://127.0.0.1:5000` or `http://localhost:5000` in a browser. Click **Pause**/**Resume** to control processing.

### Dynamic Alarm Configuration

The web dashboard allows on‑the‑fly adjustment of:

- **Density threshold**: number of total objects that triggers a high‑density alarm.
- **Species selection**: check any combination of the 13 marine classes (e.g., starfish, fish). Only checked classes that exceed the confidence threshold (0.5) generate alerts.

Changes take effect immediately after clicking **“Apply Settings”**.

### Interface Features

- Live MJPEG video feed with bounding boxes, track IDs, and crosshair highlights for small objects.
- Real‑time statistics (frame count, current targets, FPS).
- Bar chart of cumulative detections per class (Chart.js).
- Alarm list showing recent events.
- CPU and memory usage monitors.

## Performance Highlights

*Local baseline (unconstrained x86)*:

- Distilled model: ~74 ms (13.5 FPS), ~200 MB peak memory.

*Docker 6‑core / 1 GB*:

- Distilled model: ~187 ms (5.3 FPS).

*Docker 4‑core / 512 MB*:

- Distilled model: ~346 ms (2.9 FPS).

