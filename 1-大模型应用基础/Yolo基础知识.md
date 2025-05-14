### YOLO（You Only Look Once）目标检测算法

首先学习的时候先别看原理,直接看微调实操就可以了

要细研究等你成大牛了再说(lol,就是各种卷积块组成的)

#### 微调教程

印章检测

1. 首先需要有训练数据集:

图片+印章的位置
```目录
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
 ```   

2. 准备 YAML 配置文件内容包括：
   
nc--类别

示例
```yaml
train: /mnt/data/llch/my_lm_log/no_git_oic/stamp/datas/train/images
val: /mnt/data/llch/my_lm_log/no_git_oic/stamp/datas/val/images
nc: 1
names: ['stamp']
```

3. YOLO 标签文件格式
```
每个 `.txt` 文件与对应的图像文件同名，例如：
   - 图像文件：`image1.jpg`
   - 标签文件：`image1.txt`

每一行表示一个目标，行内包含以下 5 个值，用空格分隔：
`<class_id> <x_center> <y_center> <width> <height>`

参数说明：
   - `<class_id>`: 目标的类别索引（从 0 开始）。例如，`cago` 对应 0，`ki` 对应 1。
   - `<x_center>`: 目标边界框中心点的 x 坐标，归一化到 [0, 1] 范围。
   - `<y_center>`: 目标边界框中心点的 y 坐标，归一化到 [0, 1] 范围。
   - `<width>`: 目标边界框的宽度，归一化到 [0, 1] 范围。
   - `<height>`: 目标边界框的高度，归一化到 [0, 1] 范围。
```

4. 微调

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO("no_git_oic/yolo11n.pt")
print(model)
"""
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
50/50      20.5G    0.07874    0.08839     0.7664         58        640: 100%|██████████| 24/24 [00:08<00:00,  2.86it/s]
         Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 2/2 [00:00<00:00,  2.17it/s]
         all        298        597          1      0.997      0.995      0.995
"""
# uv run stamp/train_yolov11.py
results = model.train(
    data="stamp/data.yaml",
    epochs=50,
    imgsz=640,
    batch=128,
    device=0,
    pretrained=False,
    optimizer="auto",
    verbose=True,  # 已启用详细日志
)
```

5. 使用
   
```python
model = YOLO("runs/detect/train2/weights/best.pt")

im1 = Image.open("no_git_oic/stamp1.png")

results = model.predict(source=im1, save=True, save_txt=True)
```

之后可以转成其他格式,边缘设备使用
