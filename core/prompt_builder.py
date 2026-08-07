from typing import Tuple
from .state_machine import PersonaProfile


class PromptBuilder:
    @staticmethod
    def build_prompts(profile: PersonaProfile, mode: str) -> Tuple[str, str]:
        """
        Returns (system_prompt_addition, extra_user_content)
        """
        if not profile or (
            not profile.style
            and not profile.cognition
            and not profile.values
            and not profile.salutation
        ):
            return ("", "")

        system_parts = [
            "【拟态对话与人格克隆指令】",
            "你正在高度拟态并深度克隆特定目标用户的语言风格、思维逻辑与行为范式。",
            "请严格遵守以下特征要求："
        ]

        if profile.salutation:
            system_parts.append(f"- 常用称谓/代词惯性：{profile.salutation}")
        if profile.style:
            system_parts.append(f"- 语言风格与语气词：{profile.style}")
        if profile.cognition:
            system_parts.append(f"- 思维方式与逻辑习惯：{profile.cognition}")
        if profile.values:
            system_parts.append(f"- 核心价值观与立场偏好：{profile.values}")
        if profile.taboo:
            system_parts.append(f"- 规避/敏感话语点：{profile.taboo}")

        if profile.examples:
            system_parts.append("\n【典型表达示例参考】")
            for ex in profile.examples[:3]:
                system_parts.append(f"• \"{ex}\"")

        system_prompt = "\n".join(system_parts)

        extra_user_content = ""
        if mode == "distill":
            extra_user_content = "[系统提示: 当前处于人格蒸馏学习模式，请保持对话自然顺畅，适度呼应用户的沟通方式。]"

        return (system_prompt, extra_user_content)
