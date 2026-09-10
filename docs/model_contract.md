# D1 模型输入输出草案

**负责人：** C；**状态：** D2 已冻结，S2-C-01 已实现。完整公共契约见 `docs/design.md`
的“模型与通用训练契约”；本文件保留 D1 决策来源并指向当前实现。

| 项目 | 草案 |
|---|---|
| 输入 | `history: FloatTensor[B, T_h, 2]`，训练时为归一化平面坐标 |
| 输出 | `pred_future: FloatTensor[B, T_f, 2]`，与输入采用同一归一化坐标系 |
| 标签 | `future: FloatTensor[B, T_f, 2]` |
| 基础模型 | LSTM Encoder-Decoder；模型不得依赖 Client、Server 或文件路径 |
| 训练损失 | 逐坐标 MSE（D5 评估是否加入位移增量建模） |
| 推理 | `eval()` + `no_grad()`，结果由评价层反归一化后计算 ADE/FDE |
| 序列化 | 使用模型 `state_dict`；配置保存结构和 `T_h/T_f` |

冻结契约入口见 `src/models/base.py`，具体网络见 `src/models/lstm_seq2seq.py`，共享训练实现见
`src/training/torch_trainer.py`。当前实现使用固定长度、无 mask、归一化绝对坐标；Trainer
负责 MSE、Adam、device、梯度裁剪和 checkpoint，评价层负责反归一化后的 ADE/FDE。
