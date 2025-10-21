import os, argparse, glob
import numpy as np
import cv2
import torch
from model import YoloLite
from utils import xywhn_to_xyxy_abs, nms_boxes, draw_boxes

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--source",  type=str, required=True)  # 单张图片或文件夹
    ap.add_argument("--imgsz",   type=int, default=640)
    ap.add_argument("--nc",      type=int, default=3)
    ap.add_argument("--conf",    type=float, default=0.25)
    ap.add_argument("--iou",     type=float, default=0.5)
    ap.add_argument("--device",  type=str, default="mps")
    ap.add_argument("--save_dir", type=str, default="runs_yololite/predict")
    return ap.parse_args()

@torch.no_grad()
def predict_one(model, img_bgr, names, conf=0.25, iou=0.5):
    H, W = img_bgr.shape[:2]
    inp = cv2.resize(img_bgr, (640, 640))
    inp = torch.from_numpy(inp[:, :, ::-1].transpose(2,0,1)).float()/255.0  # BGR->RGB->CHW
    inp = inp.unsqueeze(0).to(next(model.parameters()).device)

    out = model(inp)  # [1,C,H',W']
    B, C, Hf, Wf = out.shape
    obj = out[:, 0:1].sigmoid()            # [1,1,H',W']
    cls = out[:, 1:1+len(names)].sigmoid() # [1,nc,H',W']
    box = out[:, 1+len(names):].sigmoid()  # [1,4,H',W'] -> cx,cy,w,h (0~1)

    # 将每个格的预测映射回原图尺度
    obj = obj[0,0]  # [H',W']
    cls = cls[0]    # [nc,H',W']
    box = box[0]    # [4,H',W']

    scores, cls_ids = cls.max(0)  # [H',W']
    scores = scores * obj         # 与 obj 相乘作为最终置信度

    boxes_list, scores_list, cls_list = [], [], []
    for gj in range(Hf):
        for gi in range(Wf):
            sc = scores[gj, gi].item()
            if sc < conf:
                continue
            cx, cy, w, h = box[:, gj, gi].tolist()
            x1,y1,x2,y2 = xywhn_to_xyxy_abs(cx,cy,w,h,W,H)
            boxes_list.append([x1,y1,x2,y2])
            scores_list.append(sc)
            cls_list.append(int(cls_ids[gj,gi].item()))

    if len(boxes_list) == 0:
        return np.array([]), np.array([]), np.array([])

    boxes = torch.tensor(boxes_list, dtype=torch.float32)
    scores = torch.tensor(scores_list, dtype=torch.float32)
    cls_ids = torch.tensor(cls_list, dtype=torch.int64)
    keep = nms_boxes(boxes, scores, iou_thres=iou, conf_thres=conf)
    return boxes[keep].numpy(), scores[keep].numpy(), cls_ids[keep].numpy()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    model = YoloLite(nc=args.nc).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    names = [f"cls{i}" for i in range(args.nc)]
    paths = [args.source] if os.path.isfile(args.source) else glob.glob(os.path.join(args.source, "*"))
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        b, s, c = predict_one(model, img, names, conf=args.conf, iou=args.iou)
        vis = draw_boxes(img, b, s, c, names)
        out_p = os.path.join(args.save_dir, os.path.basename(p))
        cv2.imwrite(out_p, vis)
        print("saved:", out_p)

if __name__ == "__main__":
    main()
