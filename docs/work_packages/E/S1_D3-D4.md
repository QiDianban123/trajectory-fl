# S1-E-01：数据诊断与真值轨迹图

- 阶段：S1（D3—D4）
- 角色：E，评价与可视化开发
- 分支：`feature/s1-e-data-plots`
- 主评审：C；协作：B
- 关联：F2、AT-02

## 目标

用真实/合法小样例验证数据清洗、坐标和 scaler，不实现预测模型图。

## D3

- 生产代码：实现原始/清洗后轨迹快照、异常计数和基础数据诊断图。
- 测试：空轨迹、NaN、错误 shape、输出目录和确定性文件名。
- 建议提交：`feat(evaluation): add data diagnostic plots [S1-E-01][F2]`。

## D4

- 生产代码：实现 history/future 真值轨迹图与 inverse-transform 抽检。
- 测试/联调：坐标轴单位米、图例、路径与 B scaler 往返一致。
- 建议提交：`test(evaluation): verify truth trajectory plots [S1-E-01][AT-02]`。

## 交付与验收

- 图中明确区分 history/future，含坐标、单位、标题和图例。
- 所有图片由脚本重建，不提交大批运行图片。
- C/B 复核坐标语义与单位。
