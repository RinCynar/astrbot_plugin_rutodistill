import asyncio
from typing import Dict, Any
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api.provider import ProviderRequest
from astrbot.api import logger

# AstrBot v4.27+ 中 EventMessageType 位于 astrbot.api.event.filter；
# 旧版本（<4.27）则直接由 astrbot.api.event 导出，这里做兼容回退。
try:
    from astrbot.api.event.filter import EventMessageType
except ImportError:
    from astrbot.api.event import EventMessageType  # 旧版 AstrBot 兼容

# extra_user_content_parts 元素要求为 ContentPart/TextPart 对象。
try:
    from astrbot.core.agent.message import TextPart
except ImportError:
    TextPart = None  # 旧版 AstrBot 无该能力，注入逻辑自动降级

from .storage.json_store import JSONStore
from .core.state_machine import SessionState, PersonaProfile, SessionMetrics
from .core.distiller import DistillerEngine, EXTRACTION_SYSTEM_PROMPT
from .core.prompt_builder import PromptBuilder


@register(
    "astrbot_plugin_rutodistill",
    "RinCynar",
    "人格蒸馏与克隆插件：多轮交互高精度蒸馏用户语言风格、认知与价值观，并支持拟态对话",
    "1.0.0",
)
class PersonaDistillerPlugin(Star):
    def __init__(self, context: Context, config: dict = None):
        super().__init__(context)
        self.config = config or {}
        self.store = JSONStore("astrbot_plugin_rutodistill")
        decay_weight = self.config.get("decay_weight", 0.7)
        convergence_threshold = self.config.get("convergence_threshold", 0.85)
        self.engine = DistillerEngine(decay_weight=decay_weight, convergence_threshold=convergence_threshold)
        # 全局蒸馏模型候选（_conf_schema.json 中 distill_model，兼容字符串或列表）
        distill_model_cfg = self.config.get("distill_model") or []
        if isinstance(distill_model_cfg, str):
            distill_model_cfg = [distill_model_cfg] if distill_model_cfg else []
        self.distill_models = [m for m in distill_model_cfg if m and isinstance(m, str)]
        self.bg_tasks = set()

    def _get_session_id(self, event: AstrMessageEvent) -> str:
        if hasattr(event, "unified_msg_origin") and event.unified_msg_origin:
            return str(event.unified_msg_origin)
        return str(event.get_sender_id())

    async def _get_state(self, session_id: str) -> SessionState:
        raw_data = await self.store.get_session(session_id, default_factory=lambda: SessionState().to_dict())
        return SessionState.from_dict(raw_data)

    async def _save_state(self, session_id: str, state: SessionState):
        await self.store.save_session(session_id, state.to_dict())

    # --- 指令处理 ---

    @filter.command("distill")
    async def mode_distill(self, event: AstrMessageEvent):
        """切换至「蒸馏学习」模式"""
        session_id = self._get_session_id(event)
        state = await self._get_state(session_id)
        state.mode = SessionState.MODE_DISTILL
        await self._save_state(session_id, state)
        yield event.plain_result("已切换至【蒸馏学习】模式。系统将在后续对话中进行增量特征萃取与 Profile 更新。")

    @filter.command("chat")
    async def mode_chat(self, event: AstrMessageEvent):
        """切换至「拟态对话」模式"""
        session_id = self._get_session_id(event)
        state = await self._get_state(session_id)
        state.mode = SessionState.MODE_CHAT
        await self._save_state(session_id, state)
        yield event.plain_result("已切换至【拟态对话】模式。将暂停特征蒸馏，全力以当前 Profile 进行深度沉浸拟态。")

    @filter.command("safe")
    async def mode_safe(self, event: AstrMessageEvent):
        """切换至「静态锚定」模式"""
        session_id = self._get_session_id(event)
        state = await self._get_state(session_id)
        state.mode = SessionState.MODE_SAFE
        await self._save_state(session_id, state)
        yield event.plain_result("已切换至【静态锚定】模式。当前 Profile 已锁定为只读。")

    @filter.command("status")
    async def show_status(self, event: AstrMessageEvent):
        """查看当前蒸馏状态与 Profile 卡片"""
        session_id = self._get_session_id(event)
        state = await self._get_state(session_id)
        p = state.profile
        m = state.metrics

        status_text = (
            "📊 **人格蒸馏与克隆 - 状态卡片**\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"• **当前模式**：`{state.mode}`\n"
            f"• **蒸馏轮数**：{m.turns_count} 轮\n"
            f"• **特征收敛度**：{m.convergence_score * 100:.1f}%\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "🧬 **已提取特征**：\n"
            f"• 表达语癖：{p.style or '（暂无）'}\n"
            f"• 思维逻辑：{p.cognition or '（暂无）'}\n"
            f"• 价值观/立场：{p.values or '（暂无）'}\n"
            f"• 常用称谓：{p.salutation or '（暂无）'}\n"
            f"• 表达禁忌：{p.taboo or '（暂无）'}\n"
            f"• 金句示例：{len(p.examples)} 条"
        )
        yield event.plain_result(status_text)

    @filter.command("distill_reset")
    async def reset_profile(self, event: AstrMessageEvent):
        """重置当前会话的人格 Profile"""
        session_id = self._get_session_id(event)
        state = SessionState()
        await self._save_state(session_id, state)
        yield event.plain_result("已成功重置当前会话的人格 Profile 与统计指标。")

    @filter.command("distill_export")
    async def export_profile(self, event: AstrMessageEvent):
        """导出当前 Profile 格式数据"""
        session_id = self._get_session_id(event)
        state = await self._get_state(session_id)
        sys_prompt, _ = PromptBuilder.build_prompts(state.profile, state.mode)
        export_text = (
            "📦 **Profile 导出数据**\n"
            "```json\n"
            f"{self._json_dumps(state.profile.to_dict())}\n"
            "```\n\n"
            "📝 **拟态 System Prompt 预览**：\n"
            f"{sys_prompt or '（未形成系统提示词）'}"
        )
        yield event.plain_result(export_text)

    @filter.command("distill_model")
    async def cmd_distill_model(self, event: AstrMessageEvent, model: str = ""):
        """查看/设置当前会话的蒸馏模型；不带参数时列出当前 Provider 启用的模型供选择"""
        session_id = self._get_session_id(event)
        state = await self._get_state(session_id)

        provider = None
        try:
            umo = getattr(event, "unified_msg_origin", None)
            provider = self.context.get_using_provider(umo=umo)
            if asyncio.iscoroutine(provider):
                provider = await provider
        except Exception as e:
            logger.debug(f"[rutodistill] Could not fetch provider: {e}")

        available_models: list[str] = []
        if provider is not None:
            try:
                available_models = list(await provider.get_models() or [])
            except Exception as e:
                logger.debug(f"[rutodistill] Could not fetch provider models: {e}")

        arg = (model or "").strip()
        lowered = arg.lower()

        if not arg:
            # 展示当前设置与可用模型列表
            current = state.model or (self.distill_models[0] if self.distill_models else "")
            cfg_models = "、".join(f"`{m}`" for m in self.distill_models) or "（未配置）"
            lines = [
                "🧬 **蒸馏模型设置**",
                f"• 当前蒸馏模型：`{current or '（跟随 Provider 默认）'}`",
                f"• 全局配置候选：{cfg_models}",
                "",
                "📋 **当前 Provider 启用的模型：**",
            ]
            if available_models:
                for i, m in enumerate(available_models, 1):
                    marker = "✅" if m == current else "•"
                    lines.append(f"{i}. {marker} `{m}`")
                lines.append("")
                lines.append("用法：`/distill_model <序号或模型名>` 设置；`/distill_model clear` 恢复默认。")
            else:
                lines.append("（无法获取当前 Provider 的模型列表）")
                lines.append("可直接使用 `/distill_model <模型名>` 手动指定。")
            yield event.plain_result("\n".join(lines))
            return

        if lowered in ("clear", "reset", "off", "none"):
            state.model = ""
            await self._save_state(session_id, state)
            yield event.plain_result("已清除会话级蒸馏模型设置，蒸馏将跟随全局配置 / Provider 默认模型。")
            return

        target = ""
        if arg.isdigit():
            idx = int(arg)
            if available_models and 1 <= idx <= len(available_models):
                target = available_models[idx - 1]
            else:
                yield event.plain_result(
                    f"序号 {idx} 无效，当前 Provider 共列出 {len(available_models)} 个模型。"
                )
                return
        else:
            target = arg

        if available_models and target not in available_models:
            yield event.plain_result(
                f"`{target}` 不在当前 Provider 的模型列表中（共 {len(available_models)} 个）。"
                "请输入 `/distill_model` 查看列表后重试，或确认模型名拼写。"
            )
            return

        state.model = target
        await self._save_state(session_id, state)
        yield event.plain_result(f"已将当前会话的蒸馏模型设置为：`{target}`")

    def _json_dumps(self, data: dict) -> str:
        import json
        return json.dumps(data, ensure_ascii=False, indent=2)

    # --- 事件钩子 ---

    @filter.event_message_type(EventMessageType.ALL)
    async def on_user_message(self, event: AstrMessageEvent):
        """监听用户消息并在蒸馏模式下后台触发特征提取"""
        msg_str = event.message_str.strip()
        if not msg_str or msg_str.startswith("/"):
            return

        session_id = self._get_session_id(event)
        state = await self._get_state(session_id)

        if state.mode == SessionState.MODE_DISTILL:
            task = asyncio.create_task(self._async_distill_worker(session_id, msg_str, event))
            self.bg_tasks.add(task)
            task.add_done_callback(self.bg_tasks.discard)

    async def _async_distill_worker(self, session_id: str, msg_str: str, event: AstrMessageEvent):
        """后台异步特征蒸馏 Task，绝不阻塞主聊天回复链路"""
        try:
            state = await self._get_state(session_id)
            if state.mode != SessionState.MODE_DISTILL:
                return

            provider = None
            try:
                umo = getattr(event, "unified_msg_origin", None)
                # v4.27.2 中 Context.get_using_provider 为同步方法，不能 await
                provider = self.context.get_using_provider(umo=umo)
                if asyncio.iscoroutine(provider):
                    provider = await provider
            except Exception as e:
                logger.debug(f"[rutodistill] Could not fetch provider: {e}")

            if not provider:
                return

            user_prompt = (
                f"【当前已有 Profile 摘要】\n"
                f"语癖: {state.profile.style}\n"
                f"思维: {state.profile.cognition}\n"
                f"价值观: {state.profile.values}\n\n"
                f"【用户最新输入】\n{msg_str}"
            )

            # 蒸馏模型优先级：会话级覆盖 > 全局配置首个候选 > Provider 默认
            distill_model = state.model or (self.distill_models[0] if self.distill_models else "")
            text_chat_kwargs: Dict[str, Any] = {
                "prompt": user_prompt,
                "system_prompt": EXTRACTION_SYSTEM_PROMPT,
            }
            if distill_model:
                text_chat_kwargs["model"] = distill_model

            res = await provider.text_chat(**text_chat_kwargs)

            raw_text = getattr(res, "completion_text", str(res))
            patch = self.engine.parse_patch_json(raw_text)

            if patch:
                updated_state = self.engine.merge_patch(state, patch)
                await self._save_state(session_id, updated_state)
                logger.debug(f"[rutodistill] Session {session_id} updated profile patch successfully.")
        except Exception as e:
            logger.debug(f"[rutodistill] Error during async background distillation worker: {e}")

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """在发起对话 LLM 请求前，自适应注入克隆人格 System Prompt 与 extra_user_content"""
        try:
            session_id = self._get_session_id(event)
            state = await self._get_state(session_id)

            sys_prompt, extra_content = PromptBuilder.build_prompts(state.profile, state.mode)

            if sys_prompt:
                if req.system_prompt:
                    req.system_prompt = f"{req.system_prompt}\n\n{sys_prompt}"
                else:
                    req.system_prompt = sys_prompt

            if extra_content and hasattr(req, "extra_user_content_parts"):
                if isinstance(req.extra_user_content_parts, list) and TextPart is not None:
                    # v4.27.2 要求元素为 TextPart/ContentPart 对象，而非纯字符串
                    req.extra_user_content_parts.append(TextPart(text=extra_content))
        except Exception as e:
            logger.debug(f"[rutodistill] Error in on_llm_request prompt injection: {e}")

    async def terminate(self):
        """销毁插件：取消所有后台异步 Task 并刷新保存数据"""
        for task in list(self.bg_tasks):
            if not task.done():
                task.cancel()
        await self.store.flush_all()
        logger.info("[rutodistill] Terminated successfully and flushed store.")
