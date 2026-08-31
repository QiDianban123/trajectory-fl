# D1 模型输入输出草案

**负责人：** C；**状态：** D2 评审候选，未冻结。

| 项目 | 草案 |
|---|---|
| 输入 | `history: FloatTensor[B, T_h, 2]`，训练时为归一化平面坐标 |
| 输出 | `pred_future: FloatTensor[B, T_f, 2]`，与输入采用同一归一化坐标系 |
| 标签 | `future: FloatTensor[B, T_f, 2]` |
| 基础模型 | LSTM Encoder-Decoder；模型不得依赖 Client、Server 或文件路径 |
| 训练损失 | 逐坐标 MSE（D5 评估是否加入位移增量建模） |
| 推理 | `eval()` + `no_grad()`，结果由评价层反归一化后计算 ADE/FDE |
| 序列化 | 使用模型 `state_dict`；配置保存结构和 `T_h/T_f` |

实现入口见 `src/models/base.py`。D2 必须明确 batch 是否含 mask、是否预测绝对坐标或增量、默认 `T_h/T_f` 和 dtype/device 约定。
