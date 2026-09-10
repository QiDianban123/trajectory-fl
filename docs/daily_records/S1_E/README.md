# S1-E-01：数据诊断与真值轨迹图

S1-E 已实现原始/清洗轨迹、异常计数、history/future 真值轨迹和 scaler
inverse-transform 抽检图。所有图由 `python scripts/plot_data_diagnostics.py`
重建，不提交运行图片。

自动化覆盖空轨迹、NaN/Inf、错误 shape、物理坐标单位、history/future 图例、
确定性文件名、输出目录和 B scaler 往返。2026-09-10 的 S1 综合验收实际生成
四张 PNG；最终证据与人工 C/B 审批状态见
[MS2 S1 准出报告](../../milestones/MS2_S1_exit_report.md)。
