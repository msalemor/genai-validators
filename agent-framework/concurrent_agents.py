# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os
from typing import Any

from agent_framework import ChatMessage, ConcurrentBuilder
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

load_dotenv()
endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
api_version=os.getenv("AZURE_OPENAI_API_VERSION")
deployment_name=os.getenv("AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME")
api_key=os.getenv("AZURE_OPENAI_API_KEY")


"""
Sample: Concurrent fan-out/fan-in (agent-only API) with default aggregator

Build a high-level concurrent workflow using ConcurrentBuilder and three domain agents.
The default dispatcher fans out the same user prompt to all agents in parallel.
The default aggregator fans in their results and yields output containing
a list[ChatMessage] representing the concatenated conversations from all agents.

Demonstrates:
- Minimal wiring with ConcurrentBuilder().participants([...]).build()
- Fan-out to multiple agents, fan-in aggregation of final ChatMessages
- Workflow completion when idle with no pending work

Prerequisites:
- Azure OpenAI access configured for AzureOpenAIChatClient (use az login + env vars)
- Familiarity with Workflow events (AgentRunEvent, WorkflowOutputEvent)
"""


async def main() -> None:
    # 1) Create three domain agents using AzureOpenAIChatClient
    chat_client = AzureOpenAIChatClient(endpoint=endpoint,
                                   deployment_name=deployment_name,
                                   api_version=api_version,
                                   api_key=api_key,)

    researcher = chat_client.create_agent(
        instructions=(
            "You're an expert market and product researcher. Given a prompt, provide concise, factual insights,"
            " opportunities, and risks."
        ),
        name="researcher",
    )

    marketer = chat_client.create_agent(
        instructions=(
            "You're a creative marketing strategist. Craft compelling value propositions and target messaging"
            " aligned to the prompt."
        ),
        name="marketer",
    )

    legal = chat_client.create_agent(
        instructions=(
            "You're a cautious legal/compliance reviewer. Highlight constraints, disclaimers, and policy concerns"
            " based on the prompt."
        ),
        name="legal",
    )

    # 2) Build a concurrent workflow
    # Participants are either Agents (type of AgentProtocol) or Executors
    workflow = ConcurrentBuilder().participants([researcher, marketer, legal]).build()

    # 3) Run with a single prompt and pretty-print the final combined messages
    events = await workflow.run("We are launching a new budget-friendly electric bike for urban commuters.")
    outputs = events.get_outputs()

    if outputs:
        print("===== Final Aggregated Conversation (messages) =====")
        for output in outputs:
            messages: list[ChatMessage] | Any = output
            for i, msg in enumerate(messages, start=1):
                name = msg.author_name if msg.author_name else "user"
                print(f"{'-' * 60}\n\n{i:02d} [{name}]:\n{msg.text}")

if __name__ == "__main__":
    asyncio.run(main())