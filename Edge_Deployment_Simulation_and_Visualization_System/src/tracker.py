# tracker.py
import numpy as np

class SimpleTracker:
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []   # 每个 track 是 dict: id, bbox, age, total_hits, trails
        self.next_id = 0

    def iou(self, boxA, boxB):
        # 计算两个矩形的 IoU
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou_val = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou_val

    def update(self, detections):
        # detections: list of dicts with key 'bbox' (list of 4)
        # 预测已有 track 的新位置（这里使用匀速模型，简化：直接用上一帧位置）
        for trk in self.tracks:
            trk['pred_bbox'] = trk['bbox']   # 简单假设不移动

        # 匹配
        assigned_dets = []
        assigned_trks = []
        if self.tracks and detections:
            iou_matrix = np.zeros((len(self.tracks), len(detections)))
            for t, trk in enumerate(self.tracks):
                for d, det in enumerate(detections):
                    iou_matrix[t, d] = self.iou(trk['pred_bbox'], det['bbox'])
            # 贪心匹配
            while True:
                if iou_matrix.size == 0:
                    break
                max_iou = np.max(iou_matrix)
                if max_iou < self.iou_threshold:
                    break
                idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                t, d = idx
                assigned_trks.append(t)
                assigned_dets.append(d)
                iou_matrix[t, :] = -1
                iou_matrix[:, d] = -1

        # 更新匹配的 track
        for t_idx, d_idx in zip(assigned_trks, assigned_dets):
            self.tracks[t_idx]['bbox'] = detections[d_idx]['bbox']
            self.tracks[t_idx]['age'] = 0
            self.tracks[t_idx]['total_hits'] += 1
            self.tracks[t_idx]['class_name'] = detections[d_idx].get('class_name', '')
            self.tracks[t_idx]['score'] = detections[d_idx].get('score', 0)
            # 记录中心点轨迹
            center = ((detections[d_idx]['bbox'][0] + detections[d_idx]['bbox'][2])//2,
                      (detections[d_idx]['bbox'][1] + detections[d_idx]['bbox'][3])//2)
            self.tracks[t_idx]['trails'].append(center)
            if len(self.tracks[t_idx]['trails']) > 30:  # 只保留最近30个点
                self.tracks[t_idx]['trails'].pop(0)

        # 处理未匹配的 track：age 增加
        for t_idx, trk in enumerate(self.tracks):
            if t_idx not in assigned_trks:
                trk['age'] += 1

        # 处理未匹配的 detection：创建新 track
        for d_idx, det in enumerate(detections):
            if d_idx not in assigned_dets:
                new_trk = {
                    'id': self.next_id,
                    'bbox': det['bbox'],
                    'age': 0,
                    'total_hits': 1,
                    'class_name': det.get('class_name', ''),
                    'score': det.get('score', 0),
                    'trails': [((det['bbox'][0] + det['bbox'][2])//2,
                                (det['bbox'][1] + det['bbox'][3])//2)]
                }
                self.tracks.append(new_trk)
                self.next_id += 1

        # 删除过期的 track
        self.tracks = [trk for trk in self.tracks if trk['age'] <= self.max_age]

        # 返回有效的 tracks（total_hits >= min_hits）
        return [trk for trk in self.tracks if trk['total_hits'] >= self.min_hits]