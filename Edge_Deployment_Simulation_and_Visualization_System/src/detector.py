# fcos_detector.py
import onnxruntime as ort
import numpy as np
import cv2

class FCOSDetector:
    def __init__(self, model_path='end2end.onnx', classes_path='classes.txt'):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 2        # 与 Docker 限制的核心数匹配
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.sess = ort.InferenceSession(
            model_path,
            sess_options,
            providers=['CPUExecutionProvider']   # 强制使用 CPU
        )
        self.input_name = self.sess.get_inputs()[0].name
        self.input_size = (512, 512)  # 模型输入尺寸 (H, W) 根据配置文件
        # 读取类别名
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        # 其他预处理参数（如 mean/std）可根据实际情况调整
        self.mean = np.array([102.9801, 115.9465, 122.7717], dtype=np.float32)
        self.std = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    def preprocess(self, image_bgr):
        # 图像预处理： resize, 减均值除标准差，转CHW
        h, w = image_bgr.shape[:2]
        scale = min(self.input_size[0] / h, self.input_size[1] / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image_bgr, (new_w, new_h))
        # 填充到目标尺寸
        canvas = np.zeros((*self.input_size, 3), dtype=np.float32)
        canvas[:new_h, :new_w, :] = resized
        canvas = (canvas - self.mean) / self.std
        canvas = np.transpose(canvas, (2, 0, 1))[np.newaxis, ...]  # 1,3,H,W
        return canvas.astype(np.float32), scale, (h, w)

    def postprocess(self, outputs, scale, original_shape):
        # 这里根据成员C给出的ONNX输出格式解析
        # 假设输出包含三个数组： bboxes (N,4), scores (N), labels (N)
        # 实际输出格式请与成员C确认，此处为示例
        bboxes = outputs[0]  # 归一化坐标 xyxy
        scores = outputs[1]
        labels = outputs[2]
        # 缩放到原始图像尺寸
        bboxes /= scale
        # 过滤低置信度
        keep = scores > 0.3
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        return bboxes, scores, labels

    def detect(self, image_bgr):
        # 预处理，假设 preprocess 返回三个值：input_tensor, scale, _
        input_tensor, scale, _ = self.preprocess(image_bgr)
        outputs = self.sess.run(None, {self.input_name: input_tensor})
        
        # 模型输出：
        # outputs[0] = dets: shape (1, num_detections, 5) -> [x1, y1, x2, y2, score]
        # outputs[1] = labels: shape (1, num_detections) -> 类别ID
        dets = outputs[0]      # (1, N, 5)
        labels = outputs[1]    # (1, N)
        
        # 去除 batch 维度
        dets = dets[0]         # (N, 5)
        labels = labels[0]     # (N,)
        
        results = []
        for i in range(dets.shape[0]):
            x1, y1, x2, y2, score = dets[i]
            cls_id = int(labels[i])
            
            # 过滤无效检测：得分过低或坐标全为零
            if score < 0.1 or (x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0):
                continue
            
            # 坐标缩放回原始图像尺寸
            x1, y1, x2, y2 = int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)
            results.append({
                'bbox': [x1, y1, x2, y2],
                'score': float(score),
                'class_id': cls_id,
                'class_name': self.class_names[cls_id] if cls_id < len(self.class_names) else 'unknown'
            })
        
        return results