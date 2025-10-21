# 🧠 simple_yolo: 基于 YOLOv8 的大豆叶病检测

本项目实现了一个轻量级的 **YOLOv8** 模型，用于 **大豆叶片病害的检测与识别**，适合初学者学习 YOLO 框架的基本原理与实现方式。

---

## 📁 项目结构

```text
simple_yolo/
│
├── dataset.py      # 数据加载与预处理逻辑
├── model.py        # YOLO 模型结构定义
├── loss.py         # 自定义损失函数
├── train.py        # 模型训练入口
├── infer.py        # 推理与结果可视化
├── utils.py        # 常用工具函数
├── runs_yololite/  # 训练结果输出目录
└── README.md       # 项目说明文件

建议使用python3.8及以上
