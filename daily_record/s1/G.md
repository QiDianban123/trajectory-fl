\# S1-G-01 工作记录



\## 成员

G（实验运行与复现开发）



\## D3 工作（2026-09-06）



\### 已完成

\- \[x] 实现 `RunContext` 核心类

&#x20; - run\_id 自动生成（时间戳+UUID）与验证

&#x20; - 路径逃逸检查（防止 `../../../` 攻击）

&#x20; - 重复 run 拦截（已存在则抛 FileExistsError）

&#x20; - git SHA 自动获取

&#x20; - 配置 JSON 序列化检查

&#x20; - 输出目录自动创建

&#x20; - config\_snapshot.json 自动保存

\- \[x] 编写 pytest 测试（6 个用例全部通过）

&#x20; - 正常创建、非法 run\_id、重复 run、路径逃逸、非序列化配置、manifest 导出



\### 提交

\- `feat(experiments): add run context \[S1-G-01]\[M2]`


## D4 工作（2026-09-06）

### 已完成
- [x] 实现数据文件 checksum 计算（SHA256）
- [x] 实现 `record_data_file` 自动记录文件及校验值
- [x] 实现 `export_data_profile` 数据画像 JSON 导出
- [x] 编写 D4 测试（5 个用例全部通过）
- [x] 修复 Windows 路径分隔符问题

### 提交
- `feat(experiments): add data manifest [S1-G-01][AT-01]`


\## 评审

\- 主评审：D

\- 状态：待评审

