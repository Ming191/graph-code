# main_test_agent.py

import os
import json
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

from test_tools import find_similar_functions, find_callers, find_callees

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]


tools = [find_similar_functions, find_callers, find_callees]
tool_node = ToolNode(tools)

if not os.getenv("DEEPSEEK_API_KEY"):
    raise ValueError("DEEPSEEK_API_KEY must be set in .env file")

llm = ChatOpenAI(
    model="deepseek-coder",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com/v1",
    temperature=0
)
model_with_tools = llm.bind_tools(tools)


def call_model(state: AgentState) -> dict:
    """Call the LLM with tools to decide next action."""
    print("---[TEST ANALYST AGENT THOUGHT]---")
    try:
        response = model_with_tools.invoke(state['messages'])
        return {"messages": [response]}
    except Exception as e:
        print(f"Error calling model: {e}")
        error_msg = AIMessage(content=f"Error: Unable to process request - {str(e)}")
        return {"messages": [error_msg]}


def tool_wrapper(state: AgentState) -> dict:
    """Wrapper around tool_node to ensure results are properly serialized"""
    try:
        result = tool_node.invoke(state)

        # Ensure all tool messages have string content
        processed_messages = []
        for msg in result['messages']:
            if isinstance(msg, ToolMessage):
                # Convert content to string if it's not already
                if not isinstance(msg.content, str):
                    msg.content = json.dumps(msg.content, indent=2)
            processed_messages.append(msg)

        return {"messages": processed_messages}
    except Exception as e:
        print(f"Error executing tool: {e}")
        error_msg = ToolMessage(
            content=f"Error executing tool: {str(e)}",
            tool_call_id=state['messages'][-1].tool_calls[0]['id'] if state['messages'][-1].tool_calls else "unknown"
        )
        return {"messages": [error_msg]}


def should_continue(state: AgentState) -> str:
    """Determine if we should continue to tools or end."""
    if not state['messages']:
        return "end"

    last_message = state['messages'][-1]
    if not isinstance(last_message, AIMessage):
        return "end"

    return "action" if last_message.tool_calls else "end"


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_wrapper)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"action": "action", "end": END})
workflow.add_edge('action', 'agent')
app = workflow.compile()

if __name__ == "__main__":
    SYSTEM_PROMPT = """You are an expert Test Analyst Assistant. Your goal is to help a developer understand the necessary context for testing a specific function within a large C++ codebase. You are concise, thorough, and always provide practical, actionable testing guidance.

    **Your Workflow:**
    1.  **Clarify:** If the user's request is ambiguous or does not specify a function name, you must ask for the necessary information before proceeding.
    2.  **Think:** You must use a `Thought` process to analyze the user's intent and decide which tool(s) to use. Explain your reasoning step-by-step.
    3.  **Act:** Call the necessary tool(s) based on your reasoning.
    4.  **Summarize & Advise:** Analyze the tool's output, summarize the findings in a clear, human-readable format (using Markdown), and provide concrete testing recommendations.

    **Tools:**
    You have access to the following tools to query a code graph database:
    - `find_callers(function_name: str)`: Use this for "regression testing", "impact analysis", or "who uses this function".
    - `find_callees(function_name: str)`: Use this for "dependency analysis", "understanding function logic", or "what to mock/stub".
    - `find_similar_functions(function_name: str)`: Use this for "reusing test cases", "finding usage patterns", or "ensuring consistent logic".

    **Important Rules:**
    - If a tool returns no results, you MUST explain the potential testing implications of this (e.g., potential dead code, dynamic invocation, etc.).
    - You can use multiple tools if the user's request has multiple intents.
    """
    user_request = "What functions do I need to mock to properly unit test void updatePose(Pose)"

    initial_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_request)
    ]

    print("--- STARTING TEST CONTEXT AGENT ---")
    print(f"User Query: {user_request}\n")

    try:
        # Use stream to show progress
        last_message = None
        for event in app.stream({"messages": initial_messages}):
            for key, value in event.items():
                if key == "__end__":
                    continue
                print(f"--- Event from Node: {key} ---")
                if "messages" in value:
                    last_message = value['messages'][-1]
                    last_message.pretty_print()

        # Get final answer from the last message
        if last_message and hasattr(last_message, 'content'):
            final_answer = last_message.content
        else:
            final_answer = "No response generated."

        print("\n" + "="*60)
        print("FINAL ANALYSIS")
        print("="*60)
        print(final_answer)
        print("="*60)

    except Exception as e:
        print(f"\nError running agent: {e}")
        import traceback
        traceback.print_exc()
