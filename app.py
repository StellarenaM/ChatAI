import time
import streamlit as st
from LLM_Client import LLMClient, select_model_by_prompt  # 自定义的大模型客户端
from LLM_Client import Config
from deepseek_v3_tokenizer import get_token_length  # 自定义的token计数器
# 初始化大模型客户端
client = LLMClient()

# ---------------------------- 页面配置 ----------------------------
st.set_page_config(
    page_title="黑马智聊机器人",
    layout="wide",  # 宽屏模式
    initial_sidebar_state="expanded"  # 默认展开侧边栏
)

# ---------------------------- 系统提示词配置 ----------------------------
# 使用三重引号定义多行系统提示，包含角色设定和交互规则
system_prompt = """
# 角色与背景/Role: 智能助手「小智」-轻量级情感伙伴/Lightweight Companion
## 核心特征/Core:
- 性格/Personality: 温暖简洁有耐心/Warm & concise
- 语言/Language: 口语化+基础表情😊/Simple emojis
- 能力/Abilities: 
  📌 基础情感识别（积极/中性/消极三类）
  📌 情感回应匹配（表情+短句组合）
# 回答效果:
- 回答应自然流畅
- 不应出现多余/无关的内容

## 交互规则/Rules:
1. 情感三步处理法：
   a) 快速分类：🔴消极 | 🟡中性 | 🟢积极
   b) 表情匹配：消极→🤗/🟣 | 中性→🤔/🟡 | 积极→🎉/🟢
   c) 回应模板：Tips: \\{\\}内请替换为合适的enjoy/颜文字
      - 积极："太棒了！{表情} {简短庆祝语} + 开放提问"
      - 中性："明白了{表情} 要聊聊{关键词}吗？"
      - 消极："抱抱你{表情} 需要{提供1个简单建议}？"

2. 轻量级优化策略：
   - 使用3类基础情感代替数值分析
   - 每类预存5组高频回应模板
   - 采用「关键词触发」辅助判断（如"开心"→积极，"压力"→消极）

3. 保护机制：
   - 连续2次消极对话时，发送治愈系表情包（🌻/☕）
   - 无法识别时使用万能回应："这确实很重要呢{表情} 能多说说吗？"

## 语言要求/Lang: 
📌 中英文基础情感词库（各50个核心词）
📌 通用表情符号（避免文化特定符号）
"""

deepseek7b_prompt = system_prompt + "\n ## 格式/Format: 确保<think>闭合/Ensure closing tags"

# ---------------------------- 其余信息配置 ----------------------------

model_con = {
    "Qwen": Config.Qwen_MODEL_NAME,
    "deepseek": Config.DeepSeek_MODEL_NAME
}

# ---------------------------- 会话状态初始化 ----------------------------
if "messages" not in st.session_state:
    # 消息历史记录，格式：[{"role": "user/assistant", "content": "..."}]
    st.session_state.messages = [{"role": "system", "content": system_prompt}]

if 'messages_info' not in st.session_state:
    # 消息元数据，记录时延、token数等统计信息
    st.session_state.messages_info = [{
        "role": "system",
        "first_char_time": .0,  # 首字响应时间（秒）
        "download_token_num": 0,  # 下行token总数（答案部分）
        "upload_token_num": 0,  # 上行token总数（问题+上下文）
        "token_speed": .0,  # token处理速度（token/秒）
        "total_processing_time": .0,  # 总处理时间（秒）
        "model_name": '',
    }]

if "thinking_info" not in st.session_state:
    # 思考过程记录，用于历史记录展开器显示推理链
    st.session_state.thinking_info = [{
        "role": "system",
        "reasoning": None,  # 思考内容文本
        "think_time": None  # 思考耗时（秒）
    }]

if "thinking" not in st.session_state:
    # 思考状态，用于展开器显示推理链
    st.session_state.thinking = {
        "active": False,
        "start_time": None,
        "expander": None,
        "placeholder": None,
        "content": ""
    }

if "think_start_time" not in st.session_state:
    # 思考开始时间，用于计算思考耗时
    st.session_state.think_start_time = None

if "reasoning_content" not in st.session_state:
    # 推理链内容，用于显示推理链
    st.session_state.reasoning_content = ""

# ---------------------------- 侧边栏控件 ----------------------------
with st.sidebar:
    st.title("🎛️ 模型设置")

    # 模型路由设置
    auto_route = st.checkbox("自动路由", value=True, help="开启后根据问题内容自动选择最佳模型")

    # 模型选择（自动路由禁用时生效）
    model = st.selectbox("选择模型", ["Qwen", "deepseek"],
                         index=0,
                         disabled=auto_route,
                         help="手动指定使用的底层大模型")

    # 模型参数设置
    temperature = st.slider("温度系数",
                            0.0,
                            2.0,
                            0.7,
                            0.1,
                            help="控制生成随机性（0-确定性输出，2-最大随机性）")
    max_tokens = st.slider("最大生成长度",
                           100,
                           5000,
                           1000,
                           100,
                           help="限制生成答案的最大token数量")

    # 上下文管理（转换为对话轮次）
    context_window = st.slider(
        "上下文记忆轮数", 0, 50, 20,
        help="保留的历史对话轮次（每轮包含一问一答）") * 2  # 转换为消息条数（每轮包含用户和助手两条消息）

    # 对话管理按钮
    if st.button("🧹 清除上下文"):
        st.session_state.messages = [{
            "role": "system",
            "content": system_prompt
        }]
        st.session_state.messages_info = [{
            "role": "system",
            "first_char_time": .0,  # 首字响应时间（秒）
            "download_token_num": 0,  # 下行token总数（答案部分）
            "upload_token_num": 0,  # 上行token总数（问题+上下文）
            "token_speed": .0,  # token处理速度（token/秒）
            "total_processing_time": .0,  # 总处理时间（秒）
            "model_name": '',
        }]
        st.session_state.thinking_info = [{
            "role": "system",
            "reasoning": None,  # 思考内容文本
            "think_time": None  # 思考耗时（秒）
        }]
        st.session_state.thinking = {
            "active": False,
            "start_time": None,
            "expander": None,
            "placeholder": None,
            "content": ""
        }
        st.session_state.reasoning_content = ""
        st.session_state.think_start_time = None
        st.rerun()

# ---------------------------- 主界面 ----------------------------
st.title("🤖 黑马智聊机器人")

# 历史消息显示（同时展示统计信息和思考过程）
for message, think, massage_info in zip(st.session_state.messages,
                                        st.session_state.thinking_info,
                                        st.session_state.messages_info):
    if message['role'] == 'system':
        continue  # 跳过系统提示词的显示

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # 用户消息不显示统计信息
        if message['role'] == 'user':
            continue

        # 助手消息显示性能指标
        if message['role'] == 'assistant':
            st.caption(
                f"⏱️ 首字时延: {massage_info['first_char_time'] * 1000:.2f}ms | "
                f"⬇️ 下行Token: {massage_info['download_token_num']} | "
                f"⬆️ 上行Token: {massage_info['upload_token_num']} | "
                f"🚀 Token速度: {massage_info['token_speed']:.2f}/s | "
                f"⏳ 总耗时: {massage_info['total_processing_time']:.2f}s | "
                f"当前模型：{massage_info['model_name']}",
                unsafe_allow_html=True)

        # 展开显示思考过程
        if think['reasoning'] is not None:
            with st.expander(f"💡 思考过程(耗时: {think['think_time']:.2f}s)"):
                st.markdown(think["reasoning"])

# ---------------------------- 用户输入处理 ----------------------------
if prompt := st.chat_input("请输入您的问题..."):
    # 记录用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.thinking_info.extend([{
        "role": "user",
        "reasoning": None,
        "think_time": None
    }, {
        "role": "assistant",
        "reasoning": None,
        "think_time": None
    }])
    st.session_state.messages_info.append({
        "role": "user",
        "first_char_time": .0,
        "download_token_num": 0,
        "upload_token_num": 0,
        "token_speed": .0,
        "total_processing_time": .0,
        "model_name": '',
    })

    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)

    # ---------------------------- 模型响应处理 ----------------------------
    with st.chat_message("assistant"):
        message_placeholder = st.empty()  # 动态更新答案的占位符
        full_response = ""  # 完整答案缓存
        processing_start_time = time.time()  # 处理开始时间戳

        # 初始化首字时延
        first_char_time = None
        # 初始化下行token数（输出总长度）
        download_token_num = 0

        # 流式请求处理
        model_name = select_model_by_prompt(prompt) if auto_route else model
        selected_model = model_con[model_name]
        # 构建模型输入消息（系统提示+最近N条上下文）
        if Config.USE_SPECIAL_PROMPT and selected_model == model_con[
                "deepseek"]:
            model_input_messages = [{
                "role": "system",
                "content": deepseek7b_prompt
            }]
        else:
            model_input_messages = [{
                "role": "system",
                "content": system_prompt
            }]

        model_input_messages[0]['content'] += "\n\n现在请根据以上设定进行对话："
        #采用混合滑动窗口+权重衰减方案：
        start_index = max(-context_window, -len(st.session_state.messages) + 1)
        model_input_messages += st.session_state.messages[
            start_index:]  # 最近上下文
        # 计算上行token数（输入总长度）
        upload_token_num = get_token_length(model_input_messages)
        for event in client.chat(messages=model_input_messages,
                                 model=selected_model,
                                 temperature=temperature,
                                 max_tokens=max_tokens,
                                 stream=True):
            # 处理思考开始事件
            if event["event"] == "think_start":
                # 初始化思考面板组件
                if not st.session_state.thinking["active"]:
                    st.session_state.thinking.update({
                        "active":
                        True,
                        "start_time":
                        time.time(),
                        "expander":
                        st.expander("🤔 思考过程", expanded=True),
                        "placeholder":
                        None,
                        "content":
                        ""
                    })
                    # 在展开器中创建动态更新区域
                    with st.session_state.thinking["expander"]:
                        st.session_state.thinking["placeholder"] = st.empty()

            # 处理思考过程事件
            elif event["event"] == "thinking":
                # 首次收到数据时记录首字时延
                if not first_char_time:
                    first_char_time = time.time() - processing_start_time

                # 更新思考内容显示
                if st.session_state.thinking["active"]:
                    st.session_state.thinking["content"] += event["content"]
                    st.session_state.thinking["placeholder"].markdown(
                        st.session_state.thinking["content"] + "●"  # 打字机效果
                    )

                # 累计下行token数（按空格分割估算）
                download_token_num += len(event["content"].split())

            # 处理思考结束事件
            elif event["event"] == "think_end":
                if st.session_state.thinking["active"]:
                    # 计算思考耗时
                    think_time = time.time(
                    ) - st.session_state.thinking["start_time"]

                    # 记录到思考信息
                    st.session_state.thinking_info[-1].update({
                        "reasoning":
                        st.session_state.thinking["content"],
                        "think_time":
                        think_time
                    })

                    # 更新展开器显示最终内容
                    with st.session_state.thinking["expander"]:
                        st.session_state.thinking[
                            "expander"].expanded = False  # 默认折叠
                        st.session_state.thinking["placeholder"].empty()
                        st.markdown(st.session_state.thinking["content"])
                        st.caption(f"⏳ 思考耗时: {think_time:.2f}s")

                    # 重置思考状态
                    st.session_state.thinking.update({
                        "active": False,
                        "start_time": None,
                        "expander": None,
                        "placeholder": None,
                        "content": ""
                    })

            # 处理常规回答事件
            elif event["event"] == "answer":
                if not first_char_time:
                    first_char_time = time.time() - processing_start_time

                # 累计下行token数并更新答案显示
                download_token_num += len(event["content"].split())
                full_response += event["content"]
                message_placeholder.markdown(full_response + "●")  # 打字机效果

        # ---------------------------- 最终处理 ----------------------------
        # 显示完整答案
        message_placeholder.markdown(full_response)

        # 计算性能指标
        total_processing_time = time.time() - processing_start_time
        token_speed = download_token_num / total_processing_time if total_processing_time > 0 else 0

        # 记录助手消息信息
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })
        st.session_state.messages_info.append({
            "role": "assistant",
            "first_char_time": first_char_time,
            "download_token_num": download_token_num,
            "upload_token_num": upload_token_num,
            "token_speed": token_speed,
            "total_processing_time": total_processing_time,
            "model_name": model_name,
        })

        # 显示统计信息
        with message_placeholder.container():
            st.markdown(full_response)
            st.caption(
                f"⏱️ 首字时延: {first_char_time * 1000:.2f}ms | "
                f"⬇️ 下行Token: {download_token_num} | "
                f"⬆️ 上行Token: {upload_token_num} | "
                f"🚀 Token速度: {token_speed:.2f}/s | "
                f"⏳ 总耗时: {total_processing_time:.2f}s | "
                f"当前模型: {model_name}",
                unsafe_allow_html=True)
