# 测试样例数据约定

D1 不随仓库分发 highD 原始数据，也不在这里提交大规模数据副本。

S1 使用以下命令按需生成匿名 highD 风格小样例，并通过生产 CLI 验证完整数据闭环：

```powershell
python scripts\run_s1_smoke.py
```

生成数据和产物位于被 Git 忽略的 `outputs/s1-smoke-<UTC time>/`。样例包含
`Track ID`、`Frame ID`、`x Position` 和 `y Position`，不来自 highD 原始数据，
仅用于接口、无泄漏 split、scaler、5-RSU 和 manifest 验收。
