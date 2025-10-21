# 🧠 simple_yolo: 基于 YOLOv8 的大豆叶病检测

本项目实现了一个轻量级的 **YOLOv8** 模型，用于 **大豆叶片病害的检测与识别**，适合初学者学习 YOLO 框架的基本原理与实现方式。
数据集链接：通过网盘分享的文件：大豆叶片病害数据集.zip
链接: https://pan.baidu.com/s/1BNx-tfLg7lF7LQr1PGDC9A?pwd=hyhh 提取码: hyhh 

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
```
⚙️ 环境依赖

建议使用 Python >= 3.8
在项目根目录下运行以下命令安装依赖：
```bash
pip install -r requirements.txt
```

若还没有 requirements.txt，可通过以下命令生成：
```bash
pip freeze > requirements.txt
```
🚀 使用说明
1️⃣ 数据集准备

将大豆叶片病害数据集放入项目根目录下的 data/ 文件夹中，结构如下：
```yaml
data/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```
```yaml
data.yaml 示例：

train: data/images/train
val: data/images/val

nc: 3
names: ['healthy', 'bacterial_spot', 'rust']
```
```python

2️⃣ 训练模型
python train.py --data data/data.yaml --epochs 100 --batch-size 16


训练结果与模型权重将保存到：

runs_yololite/weights/best.pt

3️⃣ 推理使用
python infer.py --weights runs_yololite/weights/best.pt --source test_images/


预测结果会自动保存在：

runs_yololite/inference/
```
📊 示例结果（可选）
模型版本	数据集	mAP@0.5	参数量	FPS
YOLOv8-lite	SoybeanLeaf	92.4%	4.1M	85
🧩 TODO

 增加模型可视化（Feature Map / Grad-CAM）

 支持多类别病害分类

 Jetson Nano / Edge 设备部署

 优化训练与推理速度

🤝 引用与致谢

Ultralytics YOLOv8

PyTorch 官方教程

大豆叶病害公开数据集

📄 License

本项目基于 MIT License
 开源。

💬 联系作者

GitHub: HYHH-OPS

Email: yhhh07128@gmail.com

⭐ 如果本项目对你有帮助，请点一个 Star 支持一下！
