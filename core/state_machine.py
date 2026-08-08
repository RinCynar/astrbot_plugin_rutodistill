from typing import Any, Dict
from pydantic import BaseModel, Field


class PersonaProfile(BaseModel):
    style: str = Field(default="", description="语癖、语气词、标点惯性与句式习惯")
    cognition: str = Field(default="", description="思维逻辑、决策倾向与表达节奏")
    values: str = Field(default="", description="核心价值观、态度与偏好")
    taboo: str = Field(default="", description="禁忌与敏感话题点")
    salutation: str = Field(default="", description="常用称谓与代词使用偏好")
    tone: str = Field(default="", description="语气/情绪色彩与意图特征（如戏谑、吐槽、认真论证、平静陈述、敷衍、兴奋等，可注明触发场景）")
    details: list[str] = Field(default_factory=list, description="逐条可观测的长期细节记忆（口头禅、句式、标点/空格习惯、话题偏好、具体立场等离散事实）")
    examples: list[str] = Field(default_factory=list, description="金句/典型对话/few-shot示例")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonaProfile":
        if not data:
            return cls()
        clean = dict(data)
        clean["examples"] = [str(x) for x in (data.get("examples") or [])]
        clean["details"] = [str(x) for x in (data.get("details") or [])]
        return cls(**clean)


class SessionMetrics(BaseModel):
    turns_count: int = Field(default=0, description="有效蒸馏轮数")
    convergence_score: float = Field(default=0.0, description="收敛度得分 0.0-1.0")
    last_update_ts: float = Field(default=0.0, description="最后更新时间戳")
    change_history: list[float] = Field(default_factory=list, description="最近 N 轮特征变化幅度历史（用于科学计算收敛度）")
    details_last_merge_ts: float = Field(default=0.0, description="细节库最近一次定期整理的时间戳（0 = 尚未整理过）")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionMetrics":
        if not data:
            return cls()
        # 防御旧版本/损坏数据：null 字段归一为默认值，避免下游算术/展示崩溃
        return cls(
            turns_count=int(data.get("turns_count", 0) or 0),
            convergence_score=float(data.get("convergence_score", 0.0) or 0.0),
            last_update_ts=float(data.get("last_update_ts", 0.0) or 0.0),
            change_history=[float(x) for x in (data.get("change_history") or [])],
            details_last_merge_ts=float(data.get("details_last_merge_ts", 0.0) or 0.0),
        )


class SessionState:
    MODE_DISTILL = "distill"
    MODE_CHAT = "chat"
    MODE_SAFE = "safe"

    def __init__(
        self,
        mode: str = MODE_DISTILL,
        profile: PersonaProfile = None,
        metrics: SessionMetrics = None,
        model: str = "",
        history: list = None,
    ):
        self.mode = mode if mode in (self.MODE_DISTILL, self.MODE_CHAT, self.MODE_SAFE) else self.MODE_DISTILL
        self.profile = profile or PersonaProfile()
        self.metrics = metrics or SessionMetrics()
        self.model = model or ""
        """会话级蒸馏模型覆盖；空字符串表示跟随全局配置 / Provider 默认模型"""
        self.history = list(history) if history else []
        """最近若干轮用户消息逐字原文（用于蒸馏时提供跨轮次上下文，防止特征被单轮输入覆盖丢失）"""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "profile": self.profile.to_dict(),
            "metrics": self.metrics.to_dict(),
            "model": self.model,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        if not data:
            return cls()
        return cls(
            mode=data.get("mode", cls.MODE_DISTILL),
            profile=PersonaProfile.from_dict(data.get("profile", {})),
            metrics=SessionMetrics.from_dict(data.get("metrics", {})),
            model=data.get("model", ""),
            history=data.get("history", []),
        )
