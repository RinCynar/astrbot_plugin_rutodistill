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

1. 将本仓库克隆或下载解压到 AstrBot 的 `data/plugins/astrbot_plugin_rutodistill` 目录。
2. 插件会自动从 `requirements.txt` 安装所需依赖（如 `pydantic>=2.0.0`）。
3. 在 AstrBot 管理面板中启用插件即可。

---

## 🎮 指令说明

| 指令 | 说明 |
| --- | --- |
| `/distill` | 切换至「蒸馏学习」模式 |
| `/chat` | 切换至「拟态对话」模式 |
| `/safe` | 切换至「静态锚定」模式 |
| `/status` | 查看当前蒸馏轮数、收敛度及 Profile 特征卡片 |
| `/distill_reset` | 重置当前会话的 Profile 与指标 |
| `/distill_export` | 导出 Profile 结构化数据与 System Prompt 预览 |

---

## ⚙️ 配置项 (`_conf_schema.json`)

在 AstrBot 插件配置面板中可调节以下参数：
- `enable_auto_probes`: 蒸馏模式下是否生成引导探针（默认 `false`）
- `probe_cooldown_turns`: 两次探针提问间的最小消息轮数（默认 `5`）
- `convergence_threshold`: 判定收敛阈值 `0.0 ~ 1.0`（默认 `0.85`）
- `decay_weight`: 历史特征衰减权重（默认 `0.7`）

---

## 🔒 隐私与数据管理

- 所有数据完全保存在本地 `data/plugin_data/astrbot_plugin_rutodistill/` 目录中。
- 用户可随时使用 `/distill_reset` 清空个人存储数据。
