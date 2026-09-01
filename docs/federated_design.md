# D2 联邦学习接口设计

**负责人：** D<br>
**状态：** D2 接口已冻结；本地训练、轮次循环和 FedAvg 数值实现安排在 D7—D9。

## 1. Non-IID 与数据边界

1. 先按车辆 ID 或场景划分 train/validation/test，禁止滑窗后再随机拆分。
2. 仅将训练集按道路纵向空间区域映射至 5 个模拟 RSU；每个客户端记录区域边界、车辆数、有效训练样本数与坐标统计。
3. 某区域样本不足时合并相邻区域，不得复制样本填充。车道或车辆分桶仅作次级诊断，不作为默认主划分。
4. `split_id` 由数据版本、车辆/场景切分 seed、RSU 边界和过滤规则共同确定。

## 2. Client、Server 与 Aggregator 责任

| 组件 | D2 冻结责任 | 明确禁止 |
|---|---|---|
| Client | 接收 `ClientTrainRequest`；未来委托共享 Trainer；返回一个 `ClientUpdate` 或 `ClientFailure` | 自行实现另一套训练循环；静默吞掉失败 |
| Server | 产生可追踪 `ClientSelection`；确保每个选中客户端恰有一个结果；记录失败；向 Aggregator 提交成功更新 | 把未选择、重复、过期或缺失结果送入聚合 |
| Aggregator | 校验请求；D8 对浮点状态按有效样本数加权；返回带审计元数据的新状态 | 在 D2 实现数值聚合；修改失败记录或补造样本数 |

客户端选择必须包含非负 round index 和唯一、非空客户端 ID；所选客户端必须来自该轮可用集合。Server 不得因部分客户端失败而静默改变参与集合：每个选中客户端都必须留下成功更新或显式失败记录。

## 3. ClientUpdate 与失败契约

`ClientUpdate` 固定字段为：

- `client_id`、`round_index`、`global_state_id`；
- `state`：本地训练后的完整 `state_dict`；
- `sample_count`：本轮实际参与本地训练的有效样本数，必须大于 0；
- `stats`：有限数值的训练统计，不包含原始轨迹。

`ClientFailure` 固定记录 client、round、失败阶段、异常类型、可定位消息和是否可重试。失败更新不参与 FedAvg，但必须进入轮次日志和最终运行记录。

## 4. 状态与 FedAvg 请求校验

客户端状态必须与下发的全局状态满足：

1. key 集合完全相同，不允许缺失或多余 key；
2. 每个值均为 Tensor，shape 和 dtype 与全局值一致；
3. 所有浮点 Tensor 只含有限值；
4. update 的 round 与 `global_state_id` 必须匹配该次下发；
5. 同一聚合请求中客户端 ID 唯一、有效样本数为正。

第 r 轮未来的数值实现只能对验证通过的成功更新计算：

$$w_{r+1}^{(k)}=\sum_{i\in S_r}\frac{n_i}{\sum_{j\in S_r}n_j}w_{r+1,i}^{(k)}$$

其中 `n_i` 是实际有效训练样本数，不是 batch 数、客户端声明容量或数据集原始行数。D2 的 `AggregationRequest` 仅提供校验和 `total_sample_count` 分母，不执行上述逐元素计算。

## 5. 非浮点 buffer 策略

非浮点 Tensor 固定采用 `preserve_global`：Aggregator 忽略客户端上传值并复制本轮全局状态中的原值。整数计数器、布尔 mask 和枚举值没有合理的加权平均语义；强制平均会引入类型转换、取整歧义和不可复现行为。客户端非浮点值仍必须保持 key、shape 和 dtype 兼容，以便状态结构可验证。

## 6. Local-only 公平性与隔离

实验编排器以同一 seed 生成一份基线初始 `state_dict`。每个 Local-only 客户端必须：

1. 创建全新的模型实例；
2. 接收基线状态的深拷贝；
3. 仅用自己的训练数据调用共享 Trainer；
4. 不得将任一客户端训练后的状态作为另一客户端初始状态。

`run_local_only` 在进入 Trainer 前深拷贝 initial state；新模型实例仍由实验编排器负责。Centralized、Local-only 和 Federated 复用相同 split、模型契约、初始状态、seed 与评价实现。

## 7. D8 实现前验收门禁

- 用人工状态字典验证 key、shape、dtype、NaN/Inf、样本数与过期状态拒绝路径。
- 用非浮点 buffer 人工例验证输出保留全局值。
- 用至少一个 `ClientFailure` 验证失败被记录且不静默跳过。
- D8 实现的 FedAvg 必须另加精确数值单测；本文件和 D2 类型契约不构成数值实现。
