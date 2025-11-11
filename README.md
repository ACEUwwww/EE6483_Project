# 狗猫分类项目

本项目用于 EE6483 课程，训练一个卷积神经网络对猫狗图片进行分类。

## 目录结构
- `train/`：训练集图片（已在 `.gitignore` 中忽略）
- `test1/`：测试集图片（已在 `.gitignore` 中忽略）
- `dog-and-cat-classifier-cnn.ipynb`：主要的 Jupyter Notebook，包含数据处理、模型训练与预测流程

## 使用方法
1. 安装必要依赖（建议使用 `conda` 或 `pip`，确保已安装 `tensorflow`、`torchvision` 等深度学习库）。
2. 将训练集与测试集图片放入对应目录。
3. 打开 `dog-and-cat-classifier-cnn.ipynb`，按照 Notebook 步骤执行。

## 备注
- Kaggle API 需要在本地配置 `kaggle.json` 才能下载数据集，可参考官方文档。
- 若需提交 Kaggle 结果，请将生成的预测文件上传至 Kaggle 平台。