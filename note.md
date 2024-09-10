### 9.10
+ 实现`modules.text_embedding.py`中的`TextEmbeddingModule`类
  + 用于从一段bits映射到224*224的黑白棋盘图像
  + 棋盘的大小为超参数
+ Dataset: 3\*224\*224, 在dataset目录下
  + Dataset
  + dataloader_map: {"train": train_dataloader: DataLoader, "val": val_dataloader: DataLoader, "test": DataLoader}
    + 迭代元素image
+ 