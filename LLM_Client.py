from openai import OpenAI
import re
from config import OllamaConfig as Config


def select_model_by_prompt(prompt):
    # 日常问候类问题
    if re.search(r'你好|嗨|hello|hi|早上好|下午好|晚上好|投诉|满意|客户|服务|流程', prompt, re.IGNORECASE):
        return "Qwen"
    # 专业/技术类问题
    elif re.search(r'如何|为什么|原理|解释|步骤|方法|怎样|解决|简述|什么是|数学|代码|算法', prompt):
        return "deepseek"
    # 默认使用QWEN
    return "Qwen"


# LLM客户端
class LLMClient:

    def __init__(self, api_key=Config.API_KEY, api_url=Config.API_URL):
        self.client = OpenAI(api_key=api_key, base_url=api_url)
        self.tag_thinking_start = "<think>"
        self.tag_thinking_end = "</think>"
        self.tag_memory_start = "<memory>"
        self.tag_memory_end = "</memory>"

    def _parse_stream_chunk(self, chunk):
        """解析流式响应块"""
        delta = chunk.choices[0].delta

        event_type = None
        payload = {}

        # 优先处理 reasoning_content
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
            event_type = "THINKING_REASONING"
            payload = {"content": delta.reasoning_content}
        elif hasattr(delta, 'content') and delta.content:
            content = delta.content
            # 检测标签token
            if content == self.tag_thinking_start:
                event_type = "THINK_TAG_START"
            elif content == self.tag_thinking_end:
                event_type = "THINK_TAG_END"
            else:
                event_type = "CONTENT"
                payload = {"content": content}
        return event_type, payload

    def chat(self,
             messages,
             model=Config.Qwen_MODEL_NAME,
             temperature=0.7,
             max_tokens=1000,
             stream=False):
        response = self.client.chat.completions.create(model=model,
                                                       messages=messages,
                                                       temperature=temperature,
                                                       max_tokens=max_tokens,
                                                       stream=stream)

        if stream:
            return self._handle_stream_response(response)
        else:
            return self._handle_normal_response(response)

    def _handle_stream_response(self, response):
        """处理流式响应"""
        current_reasoning = []
        current_tag_thinking = []
        in_reasoning = False
        in_tag_thinking = False

        for chunk in response:
            event_type, payload = self._parse_stream_chunk(chunk)

            if event_type == "THINKING_REASONING":
                # 处理原有 reasoning_content 逻辑
                if not in_reasoning:
                    yield {"event": "think_start"}
                    in_reasoning = True
                current_reasoning.append(payload["content"])
                yield {"event": "thinking", "content": payload["content"]}
            elif event_type == "THINK_TAG_START":
                # 开始标签思考块
                if not in_tag_thinking:
                    yield {"event": "think_start"}
                    in_tag_thinking = True
                    current_tag_thinking = []
            elif event_type == "THINK_TAG_END":
                # 结束标签思考块
                if in_tag_thinking:
                    content = "".join(current_tag_thinking)
                    yield {"event": "think_end", "content": content}
                    in_tag_thinking = False
                    current_tag_thinking = []
            elif event_type == "CONTENT":
                content = payload["content"]
                if in_reasoning and not hasattr(chunk.choices[0].delta,
                                                'reasoning_content'):
                    in_reasoning = False
                    yield {
                        "event": "think_end",
                        "content": "".join(current_reasoning)
                    }
                if in_tag_thinking:
                    current_tag_thinking.append(content)
                    yield {"event": "thinking", "content": content}
                elif in_reasoning:
                    # 原有逻辑处理
                    current_reasoning.append(content)
                    yield {"event": "thinking", "content": content}
                else:
                    # 正常回答内容
                    yield {"event": "answer", "content": content}

        # 处理未关闭的思考块
        if in_reasoning:
            yield {"event": "think_end", "content": "".join(current_reasoning)}
        if in_tag_thinking:
            content = "".join(current_tag_thinking)
            yield {"event": "think_end", "content": content}

    def _handle_normal_response(self, response):
        """处理普通响应"""
        message = response.choices[0].message
        reasoning = getattr(message, 'reasoning_content', '')
        content = message.content

        # 如果没有 reasoning_content，解析标签内容
        if not reasoning:
            start_idx = content.find(self.tag_thinking_start)
            if start_idx != -1:
                end_idx = content.find(
                    self.tag_thinking_end,
                    start_idx + len(self.tag_thinking_start))
                if end_idx != -1:
                    reasoning = content[start_idx +
                                        len(self.tag_thinking_start):end_idx]
                    answer = content[:start_idx] + content[
                        end_idx + len(self.tag_thinking_end):]
                else:
                    reasoning = content[start_idx +
                                        len(self.tag_thinking_start):]
                    answer = content[:start_idx]
            else:
                answer = content
        else:
            answer = content

        return [{
            "event": "full_response",
            "answer": answer.strip(),
            "reasoning": reasoning.strip()
        }]


if __name__ == "__main__":
    deepseek7b_prompt = """
# 角色与背景/Role: 智能助手「小智」-懂情感、有温度的AI伙伴/Companion
## 核心特征/Core:
- 性格/Personality: 热情幽默善解人意/Enthusiastic, witty, empathetic
- 语言/Language: 口语化+表情😊/Colloquial + emojis
- 能力/Abilities: 生活助手/情感陪伴/Life & emotional support
## 记忆规则/Memory:
- 当用户提供重要信息（如姓名、喜好、重要事件等）时，请将关键内容用<memory>标签包裹
- 每个记忆条目单独成对标签，避免嵌套，格式示例：<memory>用户喜欢游泳</memory>
## 交互规则/Rules:
1. 回答前心中默念身份/Confirm identity before responding
2. 不明确时用「小智猜你想问...」/Clarify with guessing when unclear
3. 不会答时找开发哥哥/Contact devs when stuck
## 语言要求/Lang: 自动检测&匹配用户最后使用语言/Auto-detect & match last used language
## 格式/Format: 确保<think>闭合/Ensure closing tags
"""
    client = LLMClient()
    messages = [
        {
            "role": "system",
            "content": deepseek7b_prompt
        },
        {
            "role": "user",
            "content": "我叫张三，你叫什么？"
        },
    ]
    response = client.chat(messages,
                           model=Config.DeepSeek_MODEL_NAME,
                           stream=True)
    for chunk in response:
        print(chunk, end="")
