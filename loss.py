import torch
import torch.nn as nn
import torch.nn.functional as F

class YoloLiteCriterion(nn.Module):
    """
    中心分配：把每个标注 (cx,cy) 对应到输出特征图的某个网格(gx, gy)
    只在该网格计算 obj=1、cls one-hot、box 的 SmoothL1
    其他网格 obj=0（负样本）
    """
    def __init__(self, nc: int, lambda_box=5.0, lambda_obj=1.0, lambda_cls=1.0):
        super().__init__()
        self.nc = nc
        self.lbox = lambda_box
        self.lobj = lambda_obj
        self.lcls = lambda_cls
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.smoothl1 = nn.SmoothL1Loss(reduction='mean')

    @torch.no_grad()
    def build_targets(self, targets, grid_size, device):
        """
        targets: List[[N_i, 5]] -> [cls, cx, cy, w, h]（0~1）
        返回：
          t_obj:  [B,1,H,W]
          t_cls:  [B,nc,H,W]
          t_box:  [B,4,H,W]（存储 cx,cy,w,h，归一化）
          mask:   [B,1,H,W]（哪些格是正样本）
        """
        B = len(targets)
        H = W = grid_size
        t_obj = torch.zeros((B,1,H,W), device=device)
        t_cls = torch.zeros((B,self.nc,H,W), device=device)
        t_box = torch.zeros((B,4,H,W), device=device)
        mask  = torch.zeros((B,1,H,W), device=device)

        for b in range(B):
            if targets[b].numel() == 0:
                continue
            # 每个 gt 分配到一个格
            for row in targets[b]:
                cls, cx, cy, w, h = row.tolist()
                gi = min(W-1, max(0, int(cx * W)))
                gj = min(H-1, max(0, int(cy * H)))
                t_obj[b,0,gj,gi] = 1.0
                t_cls[b,int(cls),gj,gi] = 1.0
                t_box[b,:,gj,gi] = torch.tensor([cx,cy,w,h], device=device)
                mask[b,0,gj,gi] = 1.0
        return t_obj, t_cls, t_box, mask

    def forward(self, pred, targets):
        """
        pred: [B, (1+nc+4), H, W]
        targets: list(len=B) of FloatTensor [Ni, 5]
        """
        B, C, H, W = pred.shape
        device = pred.device
        obj_logit = pred[:, 0:1]         # [B,1,H,W]
        cls_logit = pred[:, 1:1+self.nc] # [B,nc,H,W]
        box_reg   = pred[:, 1+self.nc:]  # [B,4,H,W]  -> 直接回归到 0~1 用 sigmoid

        t_obj, t_cls, t_box, mask = self.build_targets(targets, H, device)

        # 约束 box 到 0~1
        box_pred = torch.sigmoid(box_reg)

        # 损失
        loss_obj = self.bce(obj_logit, t_obj)
        # 只在正样本位置计算分类/回归
        pos = mask.expand_as(box_pred).bool()
        if pos.any():
            loss_box = self.smoothl1(box_pred[pos], t_box[pos])
            loss_cls = self.bce(cls_logit[mask.bool().expand_as(cls_logit)], t_cls[mask.bool().expand_as(t_cls)])
        else:
            loss_box = torch.tensor(0.0, device=device)
            loss_cls = torch.tensor(0.0, device=device)

        loss = self.lobj*loss_obj + self.lcls*loss_cls + self.lbox*loss_box
        metrics = {
            "loss": loss.detach().item(),
            "loss_obj": loss_obj.detach().item(),
            "loss_cls": loss_cls.detach().item(),
            "loss_box": loss_box.detach().item(),
        }
        return loss, metrics
