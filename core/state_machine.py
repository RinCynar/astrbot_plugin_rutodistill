from typing import Any, Dict
from pydantic import BaseModel, Field


class PersonaProfile(BaseModel):
    style: str = Field(default="", description="语癖、语气词、标点惯性与句式习惯")
    cognition: str = Field(default="", description="思维逻辑、决策倾向与表达节奏")
    values: str = Field(default="", description="核心价值观、态度与偏好")
    taboo: str = Field(default="", description="禁忌与敏感话题点")
    salutation: str = Field(default="", description="常用称谓与代词使用偏好")
    examples: list[str] = Field(default_factory=list, description="金句/典型对话/few-shot示例")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonaProfile":
        return cls(**data) if data else cls()


class SessionMetrics(BaseModel):
    turns_count: int = Field(default=0, description="有效蒸馏轮数")
    convergence_score: float = Field(default=0.0, description="收敛度得分 0.0-1.0")
    last_update_ts: float = Field(default=0.0, description="最后更新时间戳")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionMetrics":
        return cls(**data) if data else cls()


class SessionState:
    MODE_DISTILL = "distill"
    MODE_CHAT = "chat"
    MODE_SAFE = "safe"

    def __init__(
        self,
        mode: str = MODE_DISTILL,
        profile: PersonaProfile = None,
        metrics: SessionMetrics = None,
    ):
        self.mode = mode if mode in (self.MODE_DISTILL, self.MODE_CHAT, self.MODE_SAFE) else self.MODE_DISTILL
        self.profile = profile or PersonaProfile()
        self.metrics = metrics or SessionMetrics()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "profile": self.profile.to_dict(),
            "metrics": self.metrics.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        if not data:
            return cls()
        return cls(
            mode=data.get("mode", cls.MODE_DISTILL),
            profile=PersonaProfile.from_dict(data.get("profile", {})),
            metrics=SessionMetrics.from_dict(data.get("metrics", {})),
        )
