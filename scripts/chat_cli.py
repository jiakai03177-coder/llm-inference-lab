"""
交互式终端聊天工具: 与本地部署的 vLLM Qwen2.5-0.5B-Instruct 模型实时对话
支持:
1. 像 ChatGPT 一样流式实时打字机输出
2. 保持多轮对话上下文记忆与异常回滚保护
3. 输入 clear 清空历史，输入 exit 退出
"""

import json
import requests

API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def main():
    print("=" * 60)
    print("🤖 欢迎来到本地大模型交互对话终端 (由 vLLM 提供推理加速)")
    print("📌 提示: 输入想问的问题按回车即可获得流式回答")
    print("📌 特殊指令: 输入 'clear' 清空上下文记忆，输入 'exit' 退出程序")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "你是由用户在本地 RTX 3060 显卡上部署的 AI 助手。请礼貌、专业、简洁地回答用户的问题。"}
    ]

    while True:
        try:
            user_input = input("\n🧑 你: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 对话结束，再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print("👋 对话结束，再见！")
            break
        if user_input.lower() == "clear":
            messages = [messages[0]]
            print("🧹 对话历史已清空！已开启全新话题。")
            continue

        # 保证添加到历史记录的都是标准字符串
        user_msg = {"role": "user", "content": str(user_input)}
        messages.append(user_msg)

        print("🤖 助手: ", end="", flush=True)

        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 512,
            "stream": True
        }

        full_answer = ""
        request_success = False

        try:
            resp = requests.post(API_URL, json=payload, stream=True, timeout=60)
            if resp.status_code == 200:
                for line in resp.iter_lines():
                    if line:
                        line_str = line.decode("utf-8", errors="ignore")
                        if line_str.startswith("data: "):
                            data_str = line_str[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    print(content, end="", flush=True)
                                    full_answer += content
                            except Exception:
                                pass
                print()  # 换行
                if full_answer.strip():
                    messages.append({"role": "assistant", "content": full_answer.strip()})
                    request_success = True
                else:
                    # 如果流式回答为空，回滚最后一条用户提问
                    messages.pop()
                    print("(未收到模型有效输出，已重置当前轮次)")
            else:
                print(f"\n❌ 服务端返回错误: {resp.text}")
                # 异常时回滚当前提问，防止历史记录结构损坏
                messages.pop()

        except requests.exceptions.ConnectionError:
            print("\n❌ 无法连接到 vLLM 服务端 (http://localhost:8000)！请检查服务终端是否在运行。")
            break
        except Exception as e:
            print(f"\n❌ 发生异常: {e}")
            if messages and messages[-1] == user_msg:
                messages.pop()

if __name__ == "__main__":
    main()
