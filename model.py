import torch
import torch.nn as nn

class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c1, c2, k, s, p, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class C2f(nn.Module):
    """非常简化版 C2f（只做两层Conv的残差堆叠示意）"""
    def __init__(self, c1, c2, n=2):
        super().__init__()
        self.cv1 = ConvBNAct(c1, c2, 1, 1, 0)
        self.blocks = nn.Sequential(*[ConvBNAct(c2, c2, 3, 1, 1) for _ in range(n)])
    def forward(self, x):
        x = self.cv1(x)
        return self.blocks(x)

class YoloLite(nn.Module):
    """
    下采样 1/16 的单尺度检测头：
    输出通道 = nc + 5（obj, cx, cy, w, h 全部是logits/线性，loss里做sigmoid/约束）
    """
    def __init__(self, nc=3, ch=3):
        super().__init__()
        self.nc = nc
        # Backbone (stride: 2*2*2*2=16)
        self.stem = ConvBNAct(ch, 32, 3, 2, 1)   # 1/2
        self.stage2 = C2f(32, 64);   self.down2 = ConvBNAct(64, 64, 3, 2, 1)   # 1/4
        self.stage3 = C2f(64, 128);  self.down3 = ConvBNAct(128,128,3, 2, 1)   # 1/8
        self.stage4 = C2f(128,256);  self.down4 = ConvBNAct(256,256,3, 2, 1)   # 1/16

        # Neck（这里直接用最后一层；想进阶可加 FPN/PAN）
        self.neck = C2f(256, 256, n=2)

        # Detect Head：每格输出 (obj + cls*nc + box*4)
        self.head = nn.Conv2d(256, 1 + nc + 4, 1, 1, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.down2(self.stage2(x))
        x = self.down3(self.stage3(x))
        x = self.down4(self.stage4(x))
        f = self.neck(x)
        out = self.head(f)  # B, (1+nc+4), H', W'
        return out
