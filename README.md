# astrbot_plugin_rutodistill 人格蒸馏与克隆插件

`astrbot_plugin_rutodistill` 是一个基于 **AstrBot v4.x** 最新 API（建议兼容 `≥4.11.4`）的「人格蒸馏与克隆」插件。

通过多轮交互事件监听、结构化增量特征提取与参数化 Profile 更新，对特定用户的语言风格、认知模式、价值观与行为习惯进行高精度数字蒸馏，并在不同模式下实现基于 System Prompt / extra_user_content 的自适应拟态对话。

---

## ✨ 核心特性

- **多模式独立切换**：
  - `distill`（蒸馏学习模式）：边对话边由后台异步任务萃取用户语言风格与价值观。
  - `chat`（拟态对话模式）：暂停提取，完全以当前克隆 Profile 浸入拟态对话。
  - `safe`（静态锚定模式）：锁定 Profile 为只读。
- **全异步无感解耦**：蒸馏分析逻辑通过 `asyncio.create_task` 提交后台执行，**绝不阻塞主回复链路**。
- **降级保护机制**：LLM 提取 JSON 失败或网络波动时自动静默降级，不中断主流程对话。
- **数据安全持久化**：数据全部存储在 `data/plugin_data/astrbot_plugin_rutodistill/`，带原子落盘与并发锁保护。

---

## 🛠️ 安装方法

1. 使用本仓库地址https://github.com/RinCynar/astrbot_plugin_rutodistill 进行安装即可

---

## 🎮 指令说明

| 指令 | 说明 |
| --- | --- |
| `/distill` | 切换至「蒸馏学习」模式 |
| `/chat` | 切换至「拟态对话」模式 |
| `/safe` | 切换至「静态锚定」模式 |
| `/status` | 查看当前蒸馏轮数、收敛度及 Profile 特征卡片 |
| `/distill_model` | 查看当前 Provider 启用的模型列表并设置/清除蒸馏模型（支持序号或模型名） |
| `/distill_reset` | 重置当前会话的 Profile 与指标 |
| `/distill_export` | 导出 Profile 结构化数据与 System Prompt 预览 |

---

## ⚙️ 配置项 (`_conf_schema.json`)

在 AstrBot 插件配置面板中可调节以下参数：
- `enable_auto_probes`: 蒸馏模式下是否生成引导探针（默认 `false`）
- `probe_cooldown_turns`: 两次探针提问间的最小消息轮数（默认 `5`）
- `convergence_threshold`: 判定收敛阈值 `0.0 ~ 1.0`（默认 `0.85`）
- `decay_weight`: 历史特征衰减权重（默认 `0.7`）
- `distill_model`: 蒸馏模型指定。配置面板中会渲染为**下拉选择框**（AstrBot 启动后自动从当前 Provider 启用的模型中填充选项）；留空则跟随 Provider 默认模型。也可在聊天中使用 `/distill_model` 指令从当前 Provider 启用的模型中直接选择。

> 蒸馏模型的生效优先级：**会话级 `/distill_model` 设置 > 全局配置 `distill_model` 首个候选 > Provider 默认模型**。

---

## 🔒 隐私与数据管理

- 所有数据完全保存在本地 `data/plugin_data/astrbot_plugin_rutodistill/` 目录中。
- 用户可随时使用 `/distill_reset` 清空个人存储数据。

---

## ✅ 如何验证插件正常工作

### 1. 基础验证（安装与指令）

1. 在 AstrBot 管理面板确认插件已**启用**、无报错；查看 AstrBot 日志确认无 `Failed to load plugin` 记录。
2. 依次发送 `/distill`、`/chat`、`/safe`、`/status`，应分别得到对应的模式切换/状态卡片回复，**说明指令路由与状态持久化正常**。
3. 发送 `/distill_model`，应列出**当前 Provider 启用的模型**，并能用 `/distill_model 序号` 或 `/distill_model 模型名` 设置、`/distill_model clear` 清除。

### 2. 核心功能验证（蒸馏）

1. 确保处于 `/distill` 模式。
2. 连续发送 **10~20 条风格鲜明的消息**（不同句式、口头禅、观点），等待数秒。
3. 发送 `/status`：`蒸馏轮数` 应随消息增长，`收敛度` 应整体走高。
4. 检查数据落盘文件：
   `data/plugin_data/astrbot_plugin_rutodistill/<会话ID>.json`
   `profile` 下的 `style`/`cognition`/`values` 等字段应出现与你的表达风格匹配的内容，`metrics.turns_count` 应为正数。
5. 发送 `/distill_export`，确认能导出结构化的 Profile JSON 与拟态 System Prompt 预览。

> 蒸馏为**后台异步任务**，回复本身不等待蒸馏结果，因此请稍等片刻再查 `/status`。

### 3. 拟态效果验证（chat 模式）

1. 完成步骤 2 积累一定特征后，发送 `/chat` 切换到拟态对话模式。
2. 与机器人闲聊几轮，**观察其语气、用词、句式是否逐渐贴近你的表达习惯**（例如口头禅、标点风格、句长）。
3. 对比 `chat` 与 `safe` 模式的回复差异：`safe` 模式锁定 Profile 只读，`chat` 模式仍会注入人格 Prompt。

### 4. 注入与降级验证

- 在 `chat`/`distill` 模式下，AstrBot 发出的 LLM 请求应带有插件注入的 System Prompt（可通过 AstrBot 的调试/请求日志观察）。
- 临时把 Provider 密钥改错再发消息：**主对话不应崩溃**，插件只会在日志中记录 `[rutodistill]` 的 debug 级错误并自动跳过蒸馏。

### 5. 持久化与健壮性

- 重启 AstrBot 后再次发送 `/status`，蒸馏轮数、Profile 应**完整保留**。
- 发送 `/distill_reset` 后，Profile 与指标应被清空，随后可重新开始蒸馏。

### 常见问题排查

| 现象 | 排查方向 |
| --- | --- |
| `/distill_model` 无模型列表 | 未配置可用的对话 Provider，或 Provider 不支持 `get_models()`；可改用模型名直接指定 |
| 蒸馏轮数不增长 | 确认在 `distill` 模式；消息以 `/` 开头会被视为指令跳过；查看日志中 `[rutodistill]` debug 输出 |
| `status` 中 Profile 为空 | 蒸馏 LLM 输出未通过 JSON 解析时会被静默丢弃（降级保护），多积累几条风格消息后再观察 |
| 重启后数据丢失 | 确认插件运行目录下存在 `data/plugin_data/astrbot_plugin_rutodistill/` 且文件可写 |
