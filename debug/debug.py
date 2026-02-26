"""调试交互式模式 - 修复版"""
import asyncio
from pathlib import Path
from nanobot.config.loader import load_config
from nanobot.bus.queue import MessageBus
from nanobot.agent.loop import AgentLoop
from nanobot.cron.service import CronService
from nanobot.providers.litellm_provider import LiteLLMProvider
from loguru import logger

logger.enable("nanobot")

def make_provider(config):
    """创建 LLM provider（从 cli/commands.py 复制的正确实现）"""
    from nanobot.providers.litellm_provider import LiteLLMProvider
    from nanobot.providers.openai_codex_provider import OpenAICodexProvider
    from nanobot.providers.custom_provider import CustomProvider

    model = config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    p = config.get_provider(model)

    # OpenAI Codex (OAuth)
    if provider_name == "openai_codex" or model.startswith("openai-codex/"):
        return OpenAICodexProvider(default_model=model)

    # Custom: direct OpenAI-compatible endpoint, bypasses LiteLLM
    if provider_name == "custom":
        return CustomProvider(
            api_key=p.api_key if p else "no-key",
            api_base=config.get_api_base(model) or "http://localhost:8000/v1",
            default_model=model,
        )

    from nanobot.providers.registry import find_by_name
    spec = find_by_name(provider_name)
    if not model.startswith("bedrock/") and not (p and p.api_key) and not (spec and spec.is_oauth):
        print("Error: No API key configured.")
        print("Set one in ~/.nanobot/config.json under providers section")
        raise ValueError("No API key configured")

    # 正确的 LiteLLMProvider 调用方式
    return LiteLLMProvider(
        api_key=p.api_key if p else None,
        api_base=config.get_api_base(model),
        default_model=model,
        extra_headers=p.extra_headers if p else None,
        provider_name=provider_name,
    )

async def main():
    # 加载配置
    config = load_config()

    # 创建 provider（使用正确的函数）
    provider = make_provider(config)  # 修复这里

    # 创建核心组件
    bus = MessageBus()

    # 创建 cron 服务
    from nanobot.config.loader import get_data_dir
    cron_store_path = get_data_dir() / "cron" / "jobs.json"
    cron = CronService(cron_store_path)

    # 创建 agent 循环
    agent_loop = AgentLoop(
        bus=bus,
        provider=provider,  # 传入正确的 provider 对象
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        temperature=config.agents.defaults.temperature,
        max_tokens=config.agents.defaults.max_tokens,
        max_iterations=config.agents.defaults.max_tool_iterations,
        memory_window=config.agents.defaults.memory_window,
        brave_api_key=config.tools.web.search.api_key or None,
        exec_config=config.tools.exec,
        cron_service=cron,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        mcp_servers=config.tools.mcp_servers,
        channels_config=config.channels,
    )

    session_key = "debug:session"
    print("进入交互模式 (输入 'exit' 或 'quit' 退出):")

    while True:
        try:
            # 1. 获取输入
            user_input = input("\nUser: ")

            if user_input.lower() in ['exit', 'quit', 'q']:
                break

            # 2. 发送给 Agent (保持 session_key 不变以维持记忆)
            response = await agent_loop.process_direct(
                user_input,
                session_key=session_key
            )

            # 3. 打印结果
            print(f"Agent: {response}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

    await agent_loop.close_mcp()

if __name__ == "__main__":
    # 在这行设置断点，然后使用 PyCharm Debug 运行
    asyncio.run(main())