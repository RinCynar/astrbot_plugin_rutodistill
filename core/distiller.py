import json
import re
import time
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from astrbot.api import logger
from .state_machine import PersonaProfile, SessionMetrics, SessionState

EXTRACTION_SYSTEM_PROMPT = """你是一个高精度的用户表达模式分析与人格蒸馏专家。
请分析用户最新发送的消息，结合当前的 Persona Profile 摘要，提炼增量特征补丁（Patch）。

返回结果必须为且仅为合法标准的 JSON 格式，包含以下字段：
{
  "style_delta": "语气词、口头禅、标点习惯、句式结构的增量补充",
  "cognition_delta": "思维方式、逻辑倾向、表达节奏的增量补充",
  "values_delta": "价值观、态度倾向、兴趣偏好的增量补充",
  "taboo_delta": "发现的禁忌、敏感话题或厌恶表达",
  "salutation_delta": "发现的常用称谓或人称代词偏好",
  "example_candidate": "如该消息极具用户个人特色，可提炼1句典型金句示例（无则传空字符串）",
  "change_magnitude": 0.15
}

注意：
- 若无新增特征，对应 delta 字段传空字符串 ""。
- change_magnitude 表示本次更新相比已有特征的改动变化幅度（0.0 ~ 1.0）。
- 绝不编造或夸大，保持事实求是。只输出纯 JSON 数据，不要包含 markdown 代码块。"""


class ProfilePatch(BaseModel):
    style_delta: str = Field(default="")
    cognition_delta: str = Field(default="")
    values_delta: str = Field(default="")
    taboo_delta: str = Field(default="")
    salutation_delta: str = Field(default="")
    example_candidate: str = Field(default="")
    change_magnitude: float = Field(default=0.0)


class DistillerEngine:
    def __init__(self, decay_weight: float = 0.7, convergence_threshold: float = 0.85):
        self.decay_weight = decay_weight
        self.convergence_threshold = convergence_threshold

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

    def merge_patch(self, state: SessionState, patch: ProfilePatch) -> SessionState:
        profile = state.profile
        metrics = state.metrics

        def _merge_str(current: str, delta: str) -> str:
            if not delta or delta in current:
                return current
            if not current:
                return delta
            return f"{current}；{delta}"

        profile.style = _merge_str(profile.style, patch.style_delta)
        profile.cognition = _merge_str(profile.cognition, patch.cognition_delta)
        profile.values = _merge_str(profile.values, patch.values_delta)
        profile.taboo = _merge_str(profile.taboo, patch.taboo_delta)
        profile.salutation = _merge_str(profile.salutation, patch.salutation_delta)

        if patch.example_candidate and patch.example_candidate not in profile.examples:
            profile.examples.append(patch.example_candidate)
            if len(profile.examples) > 5:
                profile.examples.pop(0)

        metrics.turns_count += 1
        metrics.last_update_ts = time.time()

        # Update convergence score using moving average decay of change_magnitude
        current_mag = min(max(patch.change_magnitude, 0.0), 1.0)
        stability = 1.0 - current_mag
        metrics.convergence_score = round(
            self.decay_weight * metrics.convergence_score + (1.0 - self.decay_weight) * stability,
            4
        )

        return state
