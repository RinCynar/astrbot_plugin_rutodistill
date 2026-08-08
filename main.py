import asyncio
import random
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

# 首次进入蒸馏模式时抛出的引导话题（开放式、易引出个人表达风格）
ICE_BREAKER_TOPICS = [
    "如果明天可以完全自由地安排一整天，没有任何限制，你会怎么过？",
    "你最近单曲循环的一首歌是什么？它为什么打动你？",
    "用一个词形容你此刻的状态，并说说为什么选它。",
    "你做过最疯狂的一件事是什么？",
    "如果只能带三样东西去荒岛生活，你会带什么？",
    "你最近一次发自内心的开心是因为什么？",
    "你最想改掉的一个小毛病是什么？",
    "如果人生可以重来一次，你最想改变哪个决定？",
    "你理想中的完美一天是怎么样的？",
    "哪部电影、书或游戏对你的影响最大？为什么？",
    "你觉得自己最鲜明的性格标签是什么？",
    "如果突然得到一笔意外之财，你第一时间会做什么？",
    "你最近在为什么事情烦恼？",
    "描述一下你最喜欢的食物带给你的感受。",
    "你觉得什么样的人最有魅力？",
    "如果可以和任何人共进晚餐，你会选谁？",
    "你坚持最久的一个习惯是什么？",
    "你更相信直觉还是逻辑？为什么？",
]


@register(
    "astrbot_plugin_rutodistill",
    "RinCynar",
    "世另我：通过多轮交互高精度蒸馏用户语言风格、认知与价值观，自动学习并拟态用户的表达方式。",
    "1.0.4",
)
class PersonaDistillerPlugin(Star):
    def __init__(self, context: Context, config: dict = None):
        super().__init__(context)
        # 保留 AstrBotConfig 实例引用（其 .schema 属性会被配置面板动态读取，
        # 用于运行时注入「蒸馏模型指定」下拉框的模型选项）
        self.config = config if config is not None else {}
        self.store = JSONStore("astrbot_plugin_rutodistill")
        # 收敛度基于最近 N 轮客观变化幅度计算（DistillerEngine 内部参数）
        self.engine = DistillerEngine()
        # 全局蒸馏模型候选（_conf_schema.json 中 distill_model，兼容字符串或列表）
        distill_model_cfg = self.config.get("distill_model") or []
        if isinstance(distill_model_cfg, str):
            distill_model_cfg = [distill_model_cfg] if distill_model_cfg else []
        self.distill_models = [m for m in distill_model_cfg if m and isinstance(m, str)]
        self.bg_tasks = set()
        # 启动后台任务填充配置面板「蒸馏模型指定」下拉框选项。
        # on_astrbot_loaded 仅在 AstrBot 启动时触发一次，运行中安装/重载插件时
        # 不会触发，因此额外启动带重试的后台任务，保证下拉框能取到模型列表。
        try:
            task = asyncio.create_task(self._background_populate_options())
            self.bg_tasks.add(task)
            task.add_done_callback(self.bg_tasks.discard)
        except Exception as e:
            logger.debug(f"[rutodistill] Could not start model options background task: {e}")

    def _get_session_id(self, event: AstrMessageEvent) -> str:
        if hasattr(event, "unified_msg_origin") and event.unified_msg_origin:
            return str(event.unified_msg_origin)
        return str(event.get_sender_id())

    async def _get_state(self, session_id: str) -> SessionState:
        raw_data = await self.store.get_session(session_id, default_factory=lambda: SessionState().to_dict())
        return SessionState.from_dict(raw_data)

    async def _save_state(self, session_id: str, state: SessionState):
        await self.store.save_session(session_id, state.to_dict())

    async def _get_provider(self, event: AstrMessageEvent = None):
        """获取当前使用的对话 Provider（兼容同步/异步调用与新旧 API）"""
        try:
            if event is not None:
                umo = getattr(event, "unified_msg_origin", None)
                provider = self.context.get_using_provider(umo=umo)
            else:
                provider = self.context.get_using_provider()
            if asyncio.iscoroutine(provider):
                provider = await provider
            return provider
        except Exception as e:
            logger.debug(f"[rutodistill] Could not fetch provider: {e}")
            return None

    async def _generate_icebreaker(self, state: SessionState = None) -> str:
        """用 LLM 生成更接近人类口吻的引导话题；失败或生成质量不佳时回退到内置模板。"""
        provider = await self._get_provider()
        if provider:
            prompt = (
                "请用一句口语化、自然的开放式问题开启一段轻松的闲聊。"
                "目的是引导对方放松地打开话匣子，自然地流露出个人表达习惯。\n"
                "要求：\n"
                "- 像真实朋友随口问出来的话，不要像问卷调查或面试题；\n"
                "- 避免「如果…你会怎么」「描述一下…」「你怎么看待…」等套路句式；\n"
                "- 若对方有已知偏好，请结合偏好让问题更贴合本人；\n"
            )
            if state and (state.profile.values or state.profile.style):
                prompt += f"\n对方已知偏好：{state.profile.values or ''} {state.profile.style or ''}\n"
            prompt += "\n请只输出这句问题本身，不要解释、不要前缀、不要引号。"
            try:
                res = await provider.text_chat(
                    prompt=prompt,
                    system_prompt="你是一个擅长自然闲聊的真人朋友。",
                )
                text = getattr(res, "completion_text", str(res))
                text = text.strip().strip('"“”').strip()
                if text and 2 <= len(text) <= 80:
                    return text
            except Exception as e:
                logger.debug(f"[rutodistill] Icebreaker generation failed: {e}")
        return random.choice(ICE_BREAKER_TOPICS)

    # --- 指令处理（r- 前缀简洁结构） ---

    @filter.command("r-start")
    async def r_start(self, event: AstrMessageEvent, action: str = ""):
        """进入「蒸馏学习」模式：初次使用自动抛出随机话题，已有数据时提示是否初始化（/r-start reset 确认）"""
        session_id = self._get_session_id(event)
        state = await self._get_state(session_id)
        has_data = (
            state.metrics.turns_count > 0
            or bool(state.profile.style or state.profile.cognition or state.profile.values)
        )
        act = (action or "").strip().lower()

        # 确认初始化：清空现有数据重新开始
        if act in ("reset", "init", "yes", "y"):
            state = SessionState(mode=SessionState.MODE_DISTILL)
            await self._save_state(session_id, state)
            topic = await self._generate_icebreaker(state)
            yield event.plain_result(
                "已重新初始化并进入【蒸馏学习】模式，旧的蒸馏数据已清空。\n\n"
                "🎯 **先来聊聊这个吧**：\n"
                f"{topic}"
            )
            return

        # 已有数据：提示是否初始化
        if has_data:
            state.mode = SessionState.MODE_DISTILL
            await self._save_state(session_id, state)
            yield event.plain_result(
                f"已切换至【蒸馏学习】模式（已有 {state.metrics.turns_count} 轮蒸馏数据）。\n"
                "若想清空现有数据重新开始，请发送 `/r-start reset` 确认初始化。"
            )
            return

        # 首次进入：主动抛出随机话题引导对话
        state.mode = SessionState.MODE_DISTILL
        await self._save_state(session_id, state)
        topic = await self._generate_icebreaker(state)
        yield event.plain_result(
            "已进入【蒸馏学习】模式。系统将在后续对话中增量萃取你的表达风格。\n\n"
            "🎯 **先来聊聊这个吧**：\n"
            f"{topic}"
        )

    @filter.command("r-lock")
    async def r_lock(self, event: AstrMessageEvent):
        """切换至「静态锚定-拟态对话」模式：Profile 锁定只读，持续以当前特征拟态对话"""
        session_id = self._get_session_id(event)
        state = await self._get_state(session_id)
        state.mode = SessionState.MODE_SAFE
        await self._save_state(session_id, state)
        yield event.plain_result(
            "已切换至【静态锚定-拟态对话】模式。当前 Profile 已锁定为只读，对话将以已蒸馏特征持续拟态。"
        )

    @filter.command("r-status")
    async def r_status(self, event: AstrMessageEvent):
        """查看当前蒸馏状态与 Profile 卡片"""
        session_id = self._get_session_id(event)
        state = await self._get_state(session_id)
        p = state.profile
        m = state.metrics

        status_text = (
            "📊 **世另我 - 状态卡片**\n"
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

    @filter.command("r-export")
    async def r_export(self, event: AstrMessageEvent, format: str = ""):
        """导出当前 Profile：默认输出可直接粘贴进 AstrBot 人格设定的 markdown；/r-export json 输出原始数据"""
        session_id = self._get_session_id(event)
        state = await self._get_state(session_id)
        fmt = (format or "").strip().lower()
        if fmt == "json":
            export_text = (
                "📦 **Profile 原始数据（JSON）**\n"
                "```json\n"
                f"{self._json_dumps(state.profile.to_dict())}\n"
                "```"
            )
        else:
            persona = PromptBuilder.build_persona_markdown(state.profile)
            export_text = (
                "🧬 **可粘贴进 AstrBot 人格设定的人设 Prompt**\n\n"
                f"{persona}"
            )
        yield event.plain_result(export_text)

    @filter.command("r-info")
    async def r_info(self, event: AstrMessageEvent, model: str = ""):
        """查看当前 Provider 启用的模型列表并设置/清除蒸馏模型（支持序号或模型名）"""
        session_id = self._get_session_id(event)
        state = await self._get_state(session_id)

        # 拉取模型列表（含 get_models 失败时的当前模型兜底），并同步刷新配置面板下拉框选项
        available_models: list[str] = await self._refresh_model_options()

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
                lines.append("用法：`/r-info <序号或模型名>` 设置；`/r-info clear` 恢复默认。")
            else:
                lines.append("（无法获取当前 Provider 的模型列表）")
                lines.append("可直接使用 `/r-info <模型名>` 手动指定。")
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

        # 允许设置任意模型名；若不在当前已配置的模型列表中则给出提示
        state.model = target
        await self._save_state(session_id, state)
        if available_models and target not in available_models:
            yield event.plain_result(
                f"已将当前会话的蒸馏模型设置为：`{target}`"
                f"（注意：该模型不在当前已配置的模型列表中，请确认拼写无误）"
            )
        else:
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

        # 跳过指令消息：AstrBot 在指令 handler 匹配成功时会向事件写入
        # handlers_parsed_params。这在带唤醒前缀（如 "bot status" 被剥成
        # "status"、不带 "/" 前缀）导致 startswith("/") 检查失效时尤为关键。
        try:
            if event.get_extra("handlers_parsed_params", {}):
                return
        except Exception as e:
            logger.debug(f"[rutodistill] check handlers_parsed_params failed: {e}")

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

            provider = await self._get_provider(event)
            if not provider:
                return

            user_prompt = (
                f"【当前已有 Profile 摘要】\n"
                f"语癖: {state.profile.style}\n"
                f"思维: {state.profile.cognition}\n"
                f"价值观: {state.profile.values}\n"
                f"称谓: {state.profile.salutation}\n"
                f"禁忌: {state.profile.taboo}\n"
            )
            if state.profile.examples:
                user_prompt += (
                    "金句示例:\n" + "\n".join(f"- {ex}" for ex in state.profile.examples) + "\n"
                )
            user_prompt += f"\n【用户最新输入（逐字原文，含原始空格与缺失标点）】\n{msg_str}"

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
                # 拟态特征前置注入：放在已有 system_prompt 之前，提高模型遵循度
                if req.system_prompt:
                    req.system_prompt = f"{sys_prompt}\n\n{req.system_prompt}"
                else:
                    req.system_prompt = sys_prompt

            if extra_content and hasattr(req, "extra_user_content_parts"):
                if isinstance(req.extra_user_content_parts, list) and TextPart is not None:
                    # v4.27.2 要求元素为 TextPart/ContentPart 对象，而非纯字符串
                    req.extra_user_content_parts.append(TextPart(text=extra_content))
        except Exception as e:
            logger.debug(f"[rutodistill] Error in on_llm_request prompt injection: {e}")

    def _update_schema_options(self, models: list) -> None:
        """将模型列表写入插件配置 schema 的 options，使 WebUI 渲染为下拉选择框。

        self.config 与 AstrBot 的 plugin_md.config 是同一实例，
        配置面板 API (get_plugin_config) 会动态读取其 .schema，因此修改立即可见。
        """
        schema = getattr(self.config, "schema", None)
        if isinstance(schema, dict) and isinstance(schema.get("distill_model"), dict):
            schema["distill_model"]["options"] = [m for m in models if m]

    async def _refresh_model_options(self) -> list:
        """收集用户实际在 AstrBot「模型提供商」中启用/勾选的模型，写入 schema 的 options 并返回。

        数据来源：
        1. 每个已启用（enable=True）的对话 Provider 实例配置的 model（provider.get_model()）；
        2. provider_settings.fallback_chat_models（用户显式配置的回退模型）。

        不使用 get_models()：它返回的是 API key 支持的全部模型，
        而非用户在「模型提供商」中勾选启用的模型。
        """
        available: list[str] = []
        providers: list = []
        try:
            providers = list(self.context.get_all_providers() or [])
        except Exception as e:
            logger.debug(f"[rutodistill] get_all_providers failed: {e}")
        if not providers:
            try:
                provider = self.context.get_using_provider()
                if provider:
                    providers = [provider]
            except Exception as e:
                logger.debug(f"[rutodistill] get_using_provider failed: {e}")

        fallback_models: list[str] = []
        for provider in providers:
            # 1) 该 Provider 配置中勾选启用的模型
            try:
                m = provider.get_model()
                if m and m not in available:
                    available.append(m)
            except Exception as e:
                logger.debug(f"[rutodistill] get_model failed: {e}")
            # 2) 用户显式配置的回退模型（所有 Provider 共享同一 provider_settings）
            try:
                settings = getattr(provider, "provider_settings", None)
                if settings and not fallback_models:
                    fallback_models = list(settings.get("fallback_chat_models") or [])
            except Exception as e:
                logger.debug(f"[rutodistill] read fallback_chat_models failed: {e}")

        for m in fallback_models:
            if m and m not in available:
                available.append(m)

        self._update_schema_options(available)
        return available

    async def _background_populate_options(self):
        """带重试的后台任务：等待 Provider 就绪后填充配置面板「蒸馏模型指定」下拉框选项。"""
        for attempt in range(6):
            try:
                available = await self._refresh_model_options()
                if available:
                    logger.info(f"[rutodistill] 配置面板模型选项已填充（{len(available)} 个）。")
                    return
            except Exception as e:
                logger.debug(f"[rutodistill] populate model options attempt {attempt} failed: {e}")
            await asyncio.sleep(5)
        logger.debug("[rutodistill] 未能获取到模型列表，配置面板模型下拉框保持为空。")

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self):
        """AstrBot 启动加载完成后，将当前 Provider 启用的模型列表填充到配置面板「蒸馏模型指定」下拉框"""
        try:
            available = await self._refresh_model_options()
            logger.info(f"[rutodistill] 已向配置面板注入 {len(available)} 个模型选项。")
        except Exception as e:
            logger.debug(f"[rutodistill] on_astrbot_loaded error: {e}")

    async def terminate(self):
        """销毁插件：取消所有后台异步 Task 并刷新保存数据"""
        for task in list(self.bg_tasks):
            if not task.done():
                task.cancel()
        await self.store.flush_all()
        logger.info("[rutodistill] Terminated successfully and flushed store.")
