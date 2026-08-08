import json
import re
import time
from difflib import SequenceMatcher
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from astrbot.api import logger
from .state_machine import PersonaProfile, SessionMetrics, SessionState

EXTRACTION_SYSTEM_PROMPT = """你是一个高精度的用户表达模式分析与人格蒸馏专家。
请结合【当前已有 Profile 摘要】与【用户最新输入】，为每个维度输出**更新后的完整表述**（不是增量补丁）。

返回结果必须为且仅为合法标准的 JSON 格式，包含以下字段：
{
  "style": "结合已有特征与最新输入重新表述后的语言风格（语癖、口头禅、标点/空格/括号表情等格式习惯、句式习惯）。无需更新则传空字符串 \\"\\"",
  "cognition": "重新表述后的思维方式、逻辑倾向、表达节奏。无需更新则传空字符串 \\"\\"",
  "values": "重新表述后的价值观、态度倾向、兴趣偏好。无需更新则传空字符串 \\"\\"",
  "taboo": "重新表述后的禁忌、敏感话题或厌恶表达。无需更新则传空字符串 \\"\\"",
  "salutation": "重新表述后的常用称谓或人称代词偏好。无需更新则传空字符串 \\"\\"",
  "example_candidate": "如该消息极具用户个人特色，可提炼1句典型金句示例（无则传空字符串）",
  "change_magnitude": 0.15
}

注意：
- 每个字段都必须是**语法正确、结构清晰、用词精炼**的完整表述，可直接被下游直接使用；
  优先基于已有特征整体重写以保证一致性与连贯性，禁止用分号堆叠多个互相矛盾的描述。
- **格式保真（最重要）**：【用户最新输入】是目标用户的**逐字原文**。描述标点与格式习惯时，
  只描述原文中真实存在的现象（如"句间以空格分隔、不使用句末标点"），
  **严禁**将模型自动补全或猜测的标点、引号、空格等格式描述为目标用户的手笔，更不能为追求"完整"而改写原文格式。
- **example_candidate 必须逐字引用原文**：包括空格、缺失的标点、括号、语气词等一切字符，
  **严禁增删改任何字符**；无法逐字引用时传空字符串。
- 特征描述尽量保留可观测的个性化细节（具体语气词、口头禅、句式、引用习惯、独特用词等）；
  上下文有限时优先保留目标用户表达中最独特、最可还原的部分，避免泛化概括导致个性化流失。
- 区分稳定习惯与偶发用法：仅当某特征在目标用户的表达中**反复、自然地出现**时才标记为固定习惯；偶发的语气词、括号表情或感叹**不要**上升为固定特征，避免下游拟态过度。
- 若某维度不需要更新，对应字段传空字符串 ""，表示沿用已有表述。
- change_magnitude 表示本次更新相比已有特征的改动变化幅度（0.0 ~ 1.0），请客观评估实际重写量。
- 绝不编造或夸大，保持实事求是。只输出纯 JSON 数据，不要包含 markdown 代码块。"""


class ProfilePatch(BaseModel):
    style: str = Field(default="")
    cognition: str = Field(default="")
    values: str = Field(default="")
    taboo: str = Field(default="")
    salutation: str = Field(default="")
    example_candidate: str = Field(default="")
    change_magnitude: float = Field(default=0.0)


class DistillerEngine:
    def __init__(
        self,
        decay_weight: float = 0.7,
        convergence_threshold: float = 0.85,
        change_window: int = 5,
    ):
        self.decay_weight = decay_weight
        self.convergence_threshold = convergence_threshold
        self.change_window = max(1, int(change_window))

    def parse_patch_json(self, raw_text: str) -> Optional[ProfilePatch]:
        if not raw_text:
            return None
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned).strip()

        try:
            data = json.loads(cleaned)
            return ProfilePatch(**data)
        except Exception as e:
            logger.debug(f"[rutodistill] Failed to parse JSON patch from LLM output: {e}. Raw: {raw_text[:100]}")
            return None

    @staticmethod
    def _change_magnitude(current: str, new: str) -> float:
        """客观计算两个字段表述间的变化幅度（0.0 无变化 ~ 1.0 完全变化）。

        基于字符串相似度，不依赖 LLM 自报数值。
        """
        if not new:
            return 0.0
        if not current:
            return 1.0  # 从无到有视为大变化
        similarity = SequenceMatcher(None, current, new).ratio()
        return round(max(0.0, min(1.0, 1.0 - similarity)), 4)

    def merge_patch(self, state: SessionState, patch: ProfilePatch) -> SessionState:
        profile = state.profile
        metrics = state.metrics

        changes: list[float] = []

        def _apply(field: str, new_value: str) -> None:
            current = getattr(profile, field)
            if not new_value or new_value == current:
                return
            changes.append(self._change_magnitude(current, new_value))
            setattr(profile, field, new_value)

        _apply("style", patch.style)
        _apply("cognition", patch.cognition)
        _apply("values", patch.values)
        _apply("taboo", patch.taboo)
        _apply("salutation", patch.salutation)

        if patch.example_candidate and patch.example_candidate not in profile.examples:
            profile.examples.append(patch.example_candidate)
            if len(profile.examples) > 5:
                profile.examples.pop(0)
            changes.append(0.5)  # 新增金句示例视为中等变化

        metrics.turns_count += 1
        metrics.last_update_ts = time.time()

        # 本轮真实变化幅度 = 各字段变化幅度的平均值（无任何变化则为 0.0）
        round_magnitude = (sum(changes) / len(changes)) if changes else 0.0

        # 维护最近 N 轮变化历史
        history = list(metrics.change_history or [])
        history.append(round_magnitude)
        metrics.change_history = history[-self.change_window:]

        # 收敛度 = 1 - 最近 N 轮平均变化幅度：特征越稳定，收敛度越接近 1.0
        if metrics.change_history:
            avg_change = sum(metrics.change_history) / len(metrics.change_history)
        else:
            avg_change = 1.0
        metrics.convergence_score = round(max(0.0, min(1.0, 1.0 - avg_change)), 4)

        return state
