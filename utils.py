import torch
import torchvision
import numpy as np
import cv2

def xywhn_to_xyxy_abs(cx, cy, w, h, W, H):
    x1 = (cx - w/2) * W
    y1 = (cy - h/2) * H
    x2 = (cx + w/2) * W
    y2 = (cy + h/2) * H
    return x1, y1, x2, y2

def nms_boxes(boxes_xyxy, scores, iou_thres=0.5, conf_thres=0.25):
    m = scores > conf_thres
    boxes = boxes_xyxy[m]
    sc = scores[m]
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    keep = torchvision.ops.nms(boxes, sc, iou_thres)
    return m.nonzero().flatten()[keep]

def draw_boxes(img_bgr, boxes, scores=None, cls_ids=None, names=None):
    out = img_bgr.copy()
    for i, (x1,y1,x2,y2) in enumerate(boxes.astype(int)):
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
        if scores is not None:
            lbl = f"{scores[i]:.2f}"
            if names is not None and cls_ids is not None:
                lbl = f"{names[int(cls_ids[i])]} {lbl}"
            cv2.putText(out, lbl, (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return out
