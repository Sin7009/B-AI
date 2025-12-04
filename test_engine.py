import os
import asyncio
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda

# Set Mock Env BEFORE importing engine which initializes LLM
os.environ["OPENROUTER_API_KEY"] = "sk-mock-key"
os.environ["OPENAI_API_KEY"] = "sk-mock-key"

from engine import get_graph, AgentState
import engine

async def test_engine_flow():
    print("--- STARTING HEADLESS ENGINE TEST ---")

    # 1. Initialize MemorySaver (Mocking Postgres)
    memory = MemorySaver()

    # 2. Mocking LLM and Tools

    # Mock call_llm_async for SOLVER/ORCHESTRATOR nodes
    async def mock_llm_call(role, context, user_query=""):
        print(f"[MockLLM] Calling {role}...")
        if role == "ORCHESTRATOR": return "SOLVER"
        if role == "TRIZ": return "Inversion: Charge for NOT using the bot."
        if role == "SYSTEM": return "Bottleneck: Payment processing speed."
        if role == "CRITIC": return "RISK: Users hate paying."
        # Post mortem
        if role == "POST_MORTEM": return "Be clearer."
        return "Mock Response"

    engine.call_llm_async = mock_llm_call

    # Mock search
    engine.search.invoke = lambda q: "Mock Search Results"

    # Mock global LLM for Synthesizer which uses it directly in a chain
    # We use RunnableLambda to make it compatible with the pipe | operator
    engine.llm = RunnableLambda(lambda x: AIMessage(content="**VERDICT**: This is a mock final verdict. The system works."))

    # 3. Get Graph
    graph = get_graph(checkpointer=memory)

    # 4. Simulate User Input
    user_id = "test_user_123"
    config = {"configurable": {"thread_id": user_id}}

    query = "Как монетизировать телеграм бота?"
    input_state = {
        "messages": [HumanMessage(content=query)],
        "user_query": query
    }

    print(f"User Query: {query}")

    print("--- Running Graph ---")
    async for event in graph.astream(input_state, config, stream_mode="values"):
        # Print keys to verify flow
        present_keys = [k for k, v in event.items() if v]
        print(f"State Update: {present_keys}")

        if "final_verdict" in event and event["final_verdict"]:
            print(f"\nFINAL VERDICT: {event['final_verdict']}")

    print("\n--- TEST COMPLETE ---")

if __name__ == "__main__":
    try:
        asyncio.run(test_engine_flow())
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
