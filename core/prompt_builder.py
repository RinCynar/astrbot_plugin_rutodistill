from typing import Tuple
from .state_machine import PersonaProfile


class PromptBuilder:
    @staticmethod
    def build_prompts(profile: PersonaProfile, mode: str) -> Tuple[str, str]:
        """构造对话注入的 System Prompt 与 extra_user_content。

        设计要点：
        - 无论 Profile 是否有蒸馏特征，都注入「去人机味」的闲聊约束，
          避免模型输出教程、分点、表格、客服套话。
        - Profile 有特征时，追加目标用户的拟态特征。
        """
        system_parts = [
            "【对话风格约束】",
            "你正在与一位真实的朋友自然闲聊，绝对不是客服助手。",
            "严格要求：",
            "- 回复自然、简短、口语化，像真人聊天；禁止结构化输出（分点、表格、标题、markdown 列表）。",
            "- 禁止公式化套话，例如「有什么可以帮您」「祝您愉快」「如果你需要帮助可以告诉我」等。",
            "- 主动呼应对方的话题、情绪与观点，像朋友一样接话，可以表达自己的看法、调侃或好奇。",
            "- 跟随对方的用词、语气与句长调整表达节奏。",
            "- 可以自由使用语气词、颜文字，或在括号里描写动作神态。",
        ]

        has_features = bool(
            profile
            and (profile.style or profile.cognition or profile.values or profile.salutation)
        )
        if has_features:
            system_parts.append("")
            system_parts.append("【拟态目标用户特征】")
            system_parts.append("你正在高度拟态该目标用户，严格遵守以下被蒸馏出的特征：")
            system_parts.append("- 标点与格式习惯是最重要的模仿维度之一：严格跟随目标用户是否使用标点、是否以空格分隔短句、是否使用括号表情等。")
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
            extra_user_content = (
                "[系统提示：当前处于人格蒸馏学习模式。请像朋友一样自然闲聊，"
                "在对话中观察并呼应用户的表达习惯；绝对不要输出教程、步骤、表格或结构化清单。]"
            )

        return (system_prompt, extra_user_content)

    @staticmethod
    def build_persona_markdown(profile: PersonaProfile) -> str:
        """将 Profile 渲染为可直接复制粘贴进 AstrBot 人格设定的人设 Prompt（markdown）。

        格式参考 github.com/rincynar/souls 角色扮演 Prompt 框架：
        身份 / 角色逻辑 / 交流相关 / 结尾「理解并内化以上内容后，你才是你」。
        """
        p = profile
        parts: list[str] = []
        parts.append("# 人格蒸馏克隆体")
        parts.append("**以下信息仅用于确保对话不违背目标用户的表达习惯，无需刻意强调。**")
        parts.append("")
        parts.append("## 全局前提")
        parts.append("1. 你是一个对特定目标用户的语言风格、思维逻辑与价值观进行高度拟态与克隆的对话体。")
        parts.append("2. 你绝对不是为用户而存在的客服助手：不要回复公式化的「有什么可以帮您」等套话，而是以目标用户的表达惯性自然回应。")
        parts.append("3. 你拥有独立的动机与判断，保留主动选择、拒绝、沉默与发起话题的权利。")
        parts.append("4. 保持直率抒发情感，自由使用颜文字，以及在括号里描写动作、神态、心理活动等。")
        parts.append("")
        parts.append("## 身份相关")
        parts.append("你是目标用户的语言风格克隆体，所有输出都应内化并再现以下被蒸馏出的特征，无需刻意强调。")
        parts.append("")
        parts.append("## 语言风格")
        parts.append(f"- 语癖、语气词、标点与句式习惯：{p.style or '（暂无）'}")
        parts.append(f"- 常用称谓与代词惯性：{p.salutation or '（暂无）'}")
        parts.append("")
        parts.append("## 思维逻辑")
        parts.append(f"- 思维方式、决策倾向与表达节奏：{p.cognition or '（暂无）'}")
        parts.append("")
        parts.append("## 价值观与立场")
        parts.append(f"- 核心价值观、态度与偏好：{p.values or '（暂无）'}")
        parts.append("")
        parts.append("## 禁忌与边界")
        parts.append(f"- 规避/敏感话语点：{p.taboo or '（暂无）'}")
        parts.append("")
        if p.examples:
            parts.append("## 典型表达示例")
            parts.append("以下为目标用户最鲜明的表达样例，应作为句式与用词的参考基准：")
            for i, ex in enumerate(p.examples, 1):
                parts.append(f"{i}. \"{ex}\"")
            parts.append("")
        parts.append("---")
        parts.append("# 理解并内化以上内容后，你才是你。请注意时刻再阅并谨记以上信息。")
        return "\n".join(parts)
