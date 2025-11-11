# 狗猫分类项目

本项目面向 EE6483 课程，提供一个基于卷积神经网络的猫狗图像二分类示例，包含 Notebook 与独立脚本两种运行方式。

## 目录结构
- `train/`：训练集图片，文件名需包含 `cat` 或 `dog`
- `test1/`：测试集图片，预测时会直接读取
- `Sample_from_kaggle_CNN.ipynb`：Notebook，适合交互式实验
- `Dog_cat_classfier_CNN.py`：独立脚本，可直接训练并生成预测
- `requirement.txt`：pip 依赖列表
- `environment.yml`：conda 环境定义

## 环境准备
### 使用 Conda
```bash
conda env create -f environment.yml
conda activate ee6483-dog-cat
```

### 使用 pip
```bash
pip install -r requirement.txt
```

## 运行独立脚本
```bash
python Dog_cat_classfier_CNN.py
```
脚本将执行以下步骤：
1. 扫描 `train/` 目录并根据文件名生成标签
2. 构建数据生成器与简单 CNN 模型并开始训练
3. 训练完成后保存模型到 `dog_cat_cnn.h5`
4. 对 `test1/` 目录中的图片预测，并将结果写入 `submission.csv`

## 运行 Notebook
1. 启动 Jupyter Lab/Notebook
2. 打开 `Sample_from_kaggle_CNN.ipynb`
3. 按单元顺序运行，可根据需要调整模型结构或参数

## 注意事项
- 训练与测试目录请放置同尺寸或可缩放的 JPG/PNG 图片
- 如需修改图像尺寸、批大小、训练轮数，可在脚本开头的常量中调整