import os, glob
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def list_images(folder: str) -> List[str]:
    files = []
    for ext in IMG_EXTS:
        files += glob.glob(os.path.join(folder, f"*{ext}"))
    files.sort()
    return files

class YoloTxtDataset(Dataset):
    """
    读取 YOLO 标签（每行: cls cx cy w h，全部是0~1）
    返回:
      image: FloatTensor [3,H,W]，范围[0,1]
      targets: FloatTensor [N,5] -> [cls, cx, cy, w, h]（全归一化）
      path: 原图路径（用于调试/可视化）
    """
    def __init__(self,
                 img_dir: str,
                 label_dir: str,
                 img_size: int = 640):
        super().__init__()
        self.img_paths = list_images(img_dir)
        assert len(self.img_paths) > 0, f"no images found in {img_dir}"
        self.label_dir = label_dir
        self.img_size  = img_size

    def __len__(self): return len(self.img_paths)

    def _load_labels(self, label_path: str) -> np.ndarray:
        if not os.path.exists(label_path):
            return np.zeros((0, 5), dtype=np.float32)
        rows = []
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) != 5:  # 忽略格式异常行
                    continue
                cls, cx, cy, w, h = map(float, parts)
                rows.append([cls, cx, cy, w, h])
        if len(rows) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        return np.array(rows, dtype=np.float32)

    def __getitem__(self, idx: int):
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB").resize((self.img_size, self.img_size))
        img = np.asarray(img, dtype=np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC->CHW

        # 对应标签路径（同名 txt）
        name = os.path.splitext(os.path.basename(path))[0] + ".txt"
        label_path = os.path.join(self.label_dir, name)
        targets = self._load_labels(label_path)  # [N,5]

        return torch.from_numpy(img), torch.from_numpy(targets), path
