import onnxruntime as ort
import numpy as np
import cv2
import time
import psutil
import os
import argparse
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
os.makedirs('/app/results', exist_ok=True)

class ModelEvaluator:
    def __init__(self, model_path, img_folder=None):
        # 显式指定 providers 以避免 ONNX Runtime 报错
        self.sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.sess.get_inputs()[0].name
        # 模型输入尺寸（应与 MMDeploy 导出时一致）
        self.input_size = (256, 256)   # (H, W)
        # 成员 B 配置中的均值和标准差
        self.mean = np.array([102.9801, 115.9465, 122.7717], dtype=np.float32)
        self.std  = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        if img_folder and os.path.isdir(img_folder):
            self.img_paths = [str(p) for p in Path(img_folder).glob('*.jpg')] + \
                             [str(p) for p in Path(img_folder).glob('*.png')]
            if self.img_paths:
                print(f"Found {len(self.img_paths)} images in {img_folder}")
            else:
                print(f"No images found in {img_folder}, switching to random noise.")
                self.img_paths = None
        else:
            self.img_paths = None
            if img_folder:
                print(f"Image folder '{img_folder}' not found. Using random noise.")
            else:
                print("No image folder provided. Using random noise for performance test.")

    def preprocess(self, img_bgr):
        h, w = img_bgr.shape[:2]
        # keep ratio resize + pad
        scale = min(self.input_size[1] / w, self.input_size[0] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img_bgr, (new_w, new_h))
        # 创建画布并填充
        canvas = np.zeros((*self.input_size, 3), dtype=np.float32)
        canvas[:new_h, :new_w, :] = resized
        # 减均值除标准差
        canvas = (canvas - self.mean) / self.std
        # 转为 NCHW
        canvas = np.transpose(canvas, (2, 0, 1))[np.newaxis, ...]
        return canvas.astype(np.float32), scale

    def run_inference(self, img_bgr):
        input_tensor, _ = self.preprocess(img_bgr)
        t0 = time.perf_counter()
        outputs = self.sess.run(None, {self.input_name: input_tensor})
        inference_time = (time.perf_counter() - t0) * 1000  # ms
        return outputs, inference_time

    def benchmark(self, num_runs=100):
        print(f"Running benchmark with {num_runs} iterations...")
        # 准备测试图像
        if self.img_paths:
            test_img = cv2.imread(self.img_paths[0])
            if test_img is None:
                raise FileNotFoundError(f"Cannot read {self.img_paths[0]}")
        else:
            test_img = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)

        # 预热
        for _ in range(5):
            self.run_inference(test_img)

        latencies = []
        inference_latencies = []   # 纯推理时间
        cpu_usages = []
        mem_usages = []
        process = psutil.Process(os.getpid())

        for i in range(num_runs):
            cpu_before = process.cpu_percent(interval=None)
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            start = time.perf_counter()
            _, inf_time = self.run_inference(test_img)
            end = time.perf_counter()
            cpu_after = process.cpu_percent(interval=None)
            mem_after = process.memory_info().rss / 1024 / 1024

            latencies.append((end - start) * 1000)        # 端到端延迟 (ms)
            inference_latencies.append(inf_time)          # 纯推理延迟 (ms)
            cpu_usages.append(cpu_after)                  # 瞬时CPU%
            mem_usages.append(mem_after)

        # 汇总结果
        df = pd.DataFrame({
            'end_to_end_latency_ms': latencies,
            'inference_latency_ms': inference_latencies,
            'cpu_percent': cpu_usages,
            'memory_mb': mem_usages
        })

        avg_e2e = np.mean(latencies)
        avg_inf = np.mean(inference_latencies)

        stats = {
            'avg_end_to_end_latency_ms': avg_e2e,
            'std_end_to_end_latency_ms': np.std(latencies),
            'fps_end_to_end': 1000.0 / avg_e2e if avg_e2e > 0 else 0,
            'avg_inference_latency_ms': avg_inf,
            'std_inference_latency_ms': np.std(inference_latencies),
            'fps_inference': 1000.0 / avg_inf if avg_inf > 0 else 0,
            'avg_cpu_percent': np.mean(cpu_usages),
            'peak_memory_mb': np.max(mem_usages)
        }

        print("\n=== Performance Summary ===")
        print(f"End-to-end avg latency: {avg_e2e:.2f} ms ({1000/avg_e2e:.2f} FPS)" if avg_e2e > 0 else "N/A")
        print(f"Pure inference avg latency: {avg_inf:.2f} ms ({1000/avg_inf:.2f} FPS)" if avg_inf > 0 else "N/A")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"{k}: {v:.2f}")
        return df, stats

    def plot_results(self, df, save_path='performance.png'):
        # 创建1行4列子图，分别显示端到端延迟、推理延迟、CPU、内存
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # 端到端延迟直方图
        axes[0].hist(df['end_to_end_latency_ms'], bins=30, color='skyblue')
        axes[0].set_title('End-to-End Latency (ms)')
        axes[0].set_xlabel('ms')

        # 纯推理延迟直方图
        axes[1].hist(df['inference_latency_ms'], bins=30, color='lightgreen')
        axes[1].set_title('Inference Latency (ms)')
        axes[1].set_xlabel('ms')

        # CPU 占用率曲线
        axes[2].plot(df['cpu_percent'], color='orange')
        axes[2].set_title('CPU %')
        axes[2].set_ylim(0, None)

        # 内存占用曲线
        axes[3].plot(df['memory_mb'], color='green')
        axes[3].set_title('Memory (MB)')

        plt.tight_layout()
        plt.savefig(save_path)
        #print(f"Performance plot saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='end2end_light.onnx', help='Path to ONNX model')
    parser.add_argument('--img_dir', type=str, default=None, help='Directory of test images (optional)')
    parser.add_argument('--runs', type=int, default=100, help='Number of inference runs')
    args = parser.parse_args()

    evaluator = ModelEvaluator(args.model, args.img_dir)
    df, stats = evaluator.benchmark(args.runs)
    evaluator.plot_results(df)

    # 保存统计数据到 CSV 和 TXT
    os.makedirs('/app/results', exist_ok=True)
    df.to_csv('/app/results/benchmark_log.csv', index=False)
    with open('benchmark_summary.txt', 'w') as f:
        for k, v in stats.items():
            f.write(f"{k}: {v:.2f}\n")