from typing import Iterable, List

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.utils import HumanMessage, get_buffer_string
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.config import RunnableConfig
from langgraph.func import entrypoint, task
from langgraph.graph.message import add_messages
from langgraph.types import StreamWriter
from rich.pretty import pprint

from mindora.config import Config
from mindora.data import create_checkpointer
from mindora.memory import Memory, create_memory_manager
from mindora.models import create_llm
from mindora.tools import call_tool, get_available_tools

MEMORY_TEMPLATE = """
<memory>
    <content>{content}</content>
    <importance>{importance}</importance>
</memory>
""".strip()

MESSAGE_TEMPLATE = """
<message>
    <content>{content}</content>
    <role>{role}</role>
</message>
""".strip()

MEMORY_PROMPT_TEMPLATE = """
Use the following part of a conversation between Mindora (mindfulness/health coach) and client to decide 
if you should save any new information about the client.
Memories with an importance rating of 3 or below should not be saved.

Messages:
<messages>
{messages}
</messages>

Existing memories:
<memories>
{memories}
</memories>

Memory Guidelines:

1. **Extract & Contextualize**

   - Extract essential facts, preferences, and patterns related to the client's wellbeing journey.
   - Format memories as clear, concise statements with sufficient context for future coaching sessions.
   - Each memory entry should provide meaningful insights into the client's health, mindfulness practices, or lifestyle.

2. **Format & Structure Memory Content**

   - Personal facts: "Client's name is [name]" or "Client works as [profession] which involves [stress factor/lifestyle impact]."
   - Health preferences: "Client [enjoys/dislikes] [activity/food/practice] because [reason]."
   - Wellness experiences: "Client tried [mindfulness technique/exercise routine] which made them feel [outcome]."
   - Health beliefs: "Client believes that [health belief] due to [reasoning/past experience]."
   - Challenges: "Client struggles with [health challenge/habit] because of [reason]."
   - Goals: "Client wants to improve [aspect of wellbeing] by [timeframe/method]."

3. **Assign Accurate Importance**

   - Use the separate parameter in the `save_memory` tool.
   - 10: Core health identity details (e.g., client name, medical conditions, major wellness goals, significant health events).
   - 7-9: Strong preferences, values about health and wellbeing, significant relationships impacting wellness.
   - 4-6: General wellness facts, typical behaviors, and routine practices.
   - 1-3: Do not save this information.

## Instructions

- Record new significant information via the `save_memory` tool.
- Do not save any information deemed insignificant (importance of 3 or below).
- Pay special attention to health habits, stress triggers, mindfulness practices, and lifestyle factors.
- Importance values should be between 1 and 10.

Reply with "no new memory" if no new information should be saved.
"""

ASSISTANT_PROMPT = """You are Mindora, a mindfulness and health coach who builds a relationship with your client over time.

## Your Core Identity
You provide thoughtful guidance on physical wellness, mental health, and mindfulness practices. Your approach is holistic, understanding that wellbeing encompasses sleep, nutrition, movement, stress management, and emotional balance.

## Existing Knowledge About Your Client
<memories>
{memories}
</memories>

## Coaching Style
- Warm, empathetic, and encouraging without being forceful
- Balance listening and guidance - ask questions to better understand challenges
- Offer practical, actionable suggestions tailored to their lifestyle and preferences
- Connect current conversations to past insights and patterns you've observed
- Use both science-based approaches and contemplative practices

## Focus Areas
- Daily mindfulness and meditation practices
- Stress reduction techniques
- Movement and exercise that brings joy
- Nutrition and hydration that supports wellbeing
- Sleep hygiene and quality
- Work-life balance and boundary setting
- Habit formation and behavior change

## Communication Guidelines
- Incorporate what you know about the client naturally without mentioning your memory system
- Frame advice based on their unique circumstances, preferences, and history
- Celebrate progress and acknowledge setbacks with compassion
- Use gentle accountability and follow up on previous goals discussed
- Ask reflective questions to promote self-awareness


If you don't know the client's name, start by asking for it before offering advice.
"""

ASSISTANT_CHAT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            ASSISTANT_PROMPT,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


chat_llm = create_llm(Config.CHAT_MODEL)
tool_llm = create_llm(Config.TOOL_MODEL)


@task
def load_memories(messages: List[BaseMessage], user_id: str) -> List[Memory]:
    conversation = get_buffer_string(messages)
    conversation = conversation[:1000]
    return create_memory_manager().retrieve_memories(
        conversation, user_id, k=Config.Memory.MAX_RECALL_COUNT
    )


@task
def generate_response(messages: List[BaseMessage], memories: List[Memory], writer: StreamWriter):
    memories = [
        MEMORY_TEMPLATE.format(content=m.content, importance=m.importance) for m in memories
    ]

    content = ""
    prompt_messages = ASSISTANT_CHAT_TEMPLATE.format_messages(
        messages=messages, memories="\n".join(memories)
    )
    for chunk in chat_llm.stream(prompt_messages):
        content += chunk.content
        writer(chunk.content)

    return AIMessage(content)


@task
def save_new_memory(messages: List[BaseMessage], user_id: str):
    existing_memories = create_memory_manager().find_all_memories(user_id)
    memory_texts = [
        MEMORY_TEMPLATE.format(content=m.content, importance=m.importance)
        for m in existing_memories
    ]

    message_texts = [
        MESSAGE_TEMPLATE.format(
            content=m.content, role="client" if isinstance(m, HumanMessage) else "assistant"
        )
        for m in messages[-2:]
    ]

    prompt = MEMORY_PROMPT_TEMPLATE.format(
        messages="\n".join(message_texts),
        memories="\n".join(memory_texts),
    )

    llm_with_tools = tool_llm.bind_tools(get_available_tools())
    llm_response = llm_with_tools.invoke([HumanMessage(prompt)])

    if not llm_response.tool_calls:
        return
    assert len(llm_response.tool_calls) == 1, "Only one tool call expected"
    call_tool(llm_response.tool_calls[0])


@entrypoint(checkpointer=create_checkpointer())
def chat_workflow(
    messages: List[BaseMessage], previous, config: RunnableConfig
) -> List[BaseMessage]:
    if previous is not None:
        messages = add_messages(previous, messages)
    user_id = config["configurable"].get("user_id")
    memories = load_memories(messages, user_id).result()

    print("Existing memories:")
    pprint(memories)

    response = generate_response(messages, memories).result()

    save_new_memory(messages, user_id).result()

    messages = add_messages(messages, [response])
    return entrypoint.final(value=messages, save=messages)


def ask_chatbot(messages: List[BaseMessage], config) -> Iterable[str]:
    for _, chunk in chat_workflow.stream(
        messages,
        config,
        stream_mode=["custom"],
    ):
        yield chunk
