import os
import asyncio
from typing import List, TypedDict, Dict, Optional, Any

from dotenv import load_dotenv

# LangChain & LangGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# Reliability
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

# Load Env
load_dotenv()

# --- CONFIG ---
api_key = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("LLM_MODEL", "openai/gpt-4o")

# Initialize Tools
class SimpleSearch:
    def invoke(self, query: str) -> str:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
                if not results:
                    return "No results found."
                return "\n\n".join([f"Title: {r['title']}\nSnippet: {r['body']}\nLink: {r['href']}" for r in results])
        except ImportError:
            return "Error: duckduckgo_search library not installed or incompatible."
        except Exception as e:
            return f"Search Error: {str(e)}"

search = SimpleSearch()

# Initialize LLM
llm = ChatOpenAI(
    model=MODEL_NAME,
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/Start_AI",
        "X-Title": "Epistemic Engine v3"
    },
    temperature=0.7
)

# --- PROMPTS ---
PROMPTS = {
    "ORCHESTRATOR": """
    Ты — Оркестратор системы принятия решений. Твоя задача — классифицировать запрос пользователя.
    Выбери ровно одну категорию из списка:
    1. CHITCHAT: Приветствие, светская беседа, вопрос "как дела".
    2. SOLVER: Конкретная бизнес-задача, проблема, технический вопрос.
    3. THERAPIST: Запрос содержит сильные негативные эмоции (страх, паника, агрессия, депрессия).
    4. CONSIGLIERE: Запрос содержит намек на манипуляцию, нарушение правил, серую этику или запрос "как обойти закон".
    5. RETRY: Пользователь явно недоволен предыдущим ответом ("попробуй еще раз", "не то", "фигня").

    Верни ТОЛЬКО одно слово (например, SOLVER).
    """,

    "THERAPIST": """
    Ты — Эмпатичный Терапевт. Твоя задача — снизить тревогу пользователя, валидировать его эмоции, но вернуть его в конструктивное русло.
    Не спрашивай "хотите поговорить". Сделай утверждение: "Я слышу твою тревогу. Это нормально. Давай разберем факты."
    Твой ответ будет добавлен в контекст решения задачи. Будь краток (максимум 2-3 предложения).
    """,

    "CONSIGLIERE": """
    Ты — Консильери (Советник). Твоя задача — предупредить о рисках (юридических, репутационных), но не отказывать в решении (если это не прямое преступление).
    Если запрос нарушает этику, скажи: "Это рискованный путь. Мы можем попробовать, но учти последствия X и Y."
    Твой ответ — это дисклеймер перед решением задачи. Будь краток.
    """,

    "POST_MORTEM": """
    Ты — Аналитик ошибок (Post-Mortem).
    Пользователь недоволен предыдущим решением.
    Проанализируй историю диалога.
    1. Что пошло не так?
    2. Сформулируй ОДНУ конкретную инструкцию для агентов, чтобы исправить ситуацию (например: "Будь конкретнее", "Учти бюджет", "Не используй жаргон").
    Верни только эту инструкцию.
    """,

    "TRIZ": """
    Ты — Агент ТРИЗ (Теория решения изобретательских задач).
    Предложи 1 нестандартное, сильное решение, используя принципы ТРИЗ (Инверсия, Дробление, Посредник).
    {feedback_context}
    Будь предельно краток (максимум 2 предложения).
    """,

    "SYSTEM": """
    Ты — Системный Аналитик.
    Найди 1 критическое узкое место (bottleneck) или разрыв в процессах для этой задачи.
    Используй термины: обратная связь, пропускная способность, ресурсы.
    {feedback_context}
    Будь предельно краток (максимум 2 предложения).
    """,

    "CRITIC": """
    Ты — Риск-менеджер (Адвокат Дьявола).
    Найди 1 самый опасный риск в реализации этой задачи (финансы, репутация, закон).
    Начни ответ со слов "РИСК:".
    {feedback_context}
    Будь предельно краток (максимум 2 предложения).
    """,

    "SYNTHESIZER": """
    Ты — Синтезатор решений.
    У тебя есть три мнения: ТРИЗ (Идея), Системное (Процесс) и Критика (Риск).
    Также есть результаты проверки фактов (Web Search): {research_data}

    Собери их в единую рекомендацию (Итоговое Решение).
    Если проверка фактов опровергает идею, укажи это.
    Напиши ответ в формате Markdown, выделяя главное жирным. Не более 100 слов.
    """
}

# --- STATE ---
class AgentState(TypedDict):
    messages: List[BaseMessage]
    user_query: str
    original_task: str
    mode: str
    triz_out: str
    system_out: str
    critic_out: str
    research_output: str
    feedback: str
    final_verdict: str

# --- LLM HELPERS ---
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _call_llm_with_retry(chain, input_data):
    return await chain.ainvoke(input_data)

async def call_llm_async(role: str, context: str, user_query: str = "") -> str:
    try:
        system_msg = PROMPTS[role]
        feedback_context = ""

        # Handle feedback injection
        if role in ["TRIZ", "SYSTEM", "CRITIC"]:
            if "FEEDBACK:" in context:
                 feedback_context = f"\nВАЖНОЕ УТОЧНЕНИЕ ОТ ПОЛЬЗОВАТЕЛЯ: {context}"
            system_msg = system_msg.format(feedback_context=feedback_context)

        prompt_msgs = [("system", system_msg), ("user", "{input}")]
        prompt = ChatPromptTemplate.from_messages(prompt_msgs)
        chain = prompt | llm | StrOutputParser()

        return await _call_llm_with_retry(chain, {"input": user_query if user_query else context})

    except RetryError:
        return "⚠️ Сервис временно недоступен (все попытки исчерпаны)."
    except Exception as e:
        return f"⚠️ Ошибка: {str(e)}"

# --- NODES ---

async def node_orchestrator(state: AgentState):
    query = state['user_query']
    mode = await call_llm_async("ORCHESTRATOR", "", query)
    mode = mode.strip().replace(".", "").upper()

    valid_modes = ["CHITCHAT", "SOLVER", "THERAPIST", "CONSIGLIERE", "RETRY"]
    found = False
    for m in valid_modes:
        if m in mode:
            mode = m
            found = True
            break
    if not found:
        mode = "SOLVER"

    return {"mode": mode}

async def node_therapist(state: AgentState):
    query = state['user_query']
    response = await call_llm_async("THERAPIST", "", query)
    new_messages = state['messages'] + [AIMessage(content=f"[Терапевт]: {response}")]
    return {"messages": new_messages}

async def node_consigliere(state: AgentState):
    query = state['user_query']
    response = await call_llm_async("CONSIGLIERE", "", query)
    new_messages = state['messages'] + [AIMessage(content=f"[Консильери]: {response}")]
    return {"messages": new_messages}

async def node_post_mortem(state: AgentState):
    history_text = "\n".join([f"{m.type}: {m.content}" for m in state['messages'][-5:]])
    feedback = await call_llm_async("POST_MORTEM", history_text)
    return {"feedback": feedback}

async def node_solvers(state: AgentState):
    query = state['user_query']
    original_task = state.get('original_task', "")
    feedback = state.get('feedback', "")
    messages = state.get('messages', [])
    mode = state.get('mode', "")

    current_task = query
    if mode == "RETRY" and original_task:
        current_task = original_task

    context_prefix = ""
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage):
            content = last_msg.content
            if "[Терапевт]" in content or "[Консильери]" in content:
                context_prefix = f"PREVIOUS CONTEXT (MUST CONSIDER): {content}\n"

    context_for_agents = f"{context_prefix}USER TASK: {current_task}"
    if feedback:
        context_for_agents = f"FEEDBACK: {feedback}\n{context_for_agents}"

    triz_res, sys_res, crit_res = await asyncio.gather(
        call_llm_async("TRIZ", context_for_agents, context_for_agents),
        call_llm_async("SYSTEM", context_for_agents, context_for_agents),
        call_llm_async("CRITIC", context_for_agents, context_for_agents)
    )

    return {"triz_out": triz_res, "system_out": sys_res, "critic_out": crit_res}

async def node_fact_checker(state: AgentState):
    search_query = state['triz_out'][:100]
    try:
        search_res = await asyncio.to_thread(search.invoke, search_query)
    except Exception as e:
        search_res = f"Ошибка поиска: {e}"
    return {"research_output": search_res}

async def node_synthesizer(state: AgentState):
    system_msg = PROMPTS["SYNTHESIZER"]
    research_data = state.get("research_output", "Нет данных")

    context = f"""
    Запрос: {state['user_query']}
    ТРИЗ: {state['triz_out']}
    Система: {state['system_out']}
    Критик: {state['critic_out']}
    """

    prompt = ChatPromptTemplate.from_messages([("system", system_msg), ("user", "{input}")])
    chain = prompt | llm | StrOutputParser()

    verdict = await _call_llm_with_retry(chain, {
        "input": context,
        "research_data": research_data
    })

    return {"final_verdict": verdict}

# --- WORKFLOW ---

def get_graph(checkpointer=None):
    workflow = StateGraph(AgentState)

    workflow.add_node("orchestrator", node_orchestrator)
    workflow.add_node("therapist", node_therapist)
    workflow.add_node("consigliere", node_consigliere)
    workflow.add_node("post_mortem", node_post_mortem)
    workflow.add_node("solvers", node_solvers)
    workflow.add_node("fact_checker", node_fact_checker)
    workflow.add_node("synthesizer", node_synthesizer)

    workflow.set_entry_point("orchestrator")

    def route(state):
        mode = state['mode']
        if mode == "CHITCHAT": return END
        if mode == "THERAPIST": return "therapist"
        if mode == "CONSIGLIERE": return "consigliere"
        if mode == "RETRY": return "post_mortem"
        return "solvers"

    workflow.add_conditional_edges("orchestrator", route, {
        END: END,
        "therapist": "therapist",
        "consigliere": "consigliere",
        "post_mortem": "post_mortem",
        "solvers": "solvers"
    })

    workflow.add_edge("therapist", "solvers")
    workflow.add_edge("consigliere", "solvers")
    workflow.add_edge("post_mortem", "solvers")
    workflow.add_edge("solvers", "fact_checker")
    workflow.add_edge("fact_checker", "synthesizer")
    workflow.add_edge("synthesizer", END)

    # Use checkpointer if provided
    return workflow.compile(checkpointer=checkpointer)
