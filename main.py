# !pip install langchain-openai langgraph langchain-core python-dotenv rich nest_asyncio langchain-community duckduckgo-search

import nest_asyncio
nest_asyncio.apply()

import os
import asyncio
import sys
from typing import List, TypedDict, Dict, Optional, Any

# Загрузка переменных окружения
from dotenv import load_dotenv

# LangChain & LangGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_community.tools import DuckDuckGoSearchRun

# Надежность (Retries)
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

# Визуализация (Rich)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.layout import Layout
from rich.markdown import Markdown

# --- НАСТРОЙКА (SETUP) ---
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    print("ОШИБКА: Не найден OPENROUTER_API_KEY в файле .env")
    sys.exit(1)

# Модель по умолчанию (можно переопределить через env)
MODEL_NAME = os.getenv("LLM_MODEL", "openai/gpt-4o")

# Инициализация модели через OpenRouter
llm = ChatOpenAI(
    model=MODEL_NAME,
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/Start_AI", # Для статистики OpenRouter
        "X-Title": "Epistemic Engine v3"
    },
    temperature=0.7
)

search = DuckDuckGoSearchRun()
console = Console()

# --- 1. ПРОМПТЫ (SYSTEM PROMPTS) ---
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

# --- 2. ЛОГИКА LLM (ASYNC & RELIABILITY) ---

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _call_llm_with_retry(chain, input_data):
    """Внутренняя функция для вызова LLM с механизмом повторов."""
    return await chain.ainvoke(input_data)

async def call_llm_async(role: str, context: str, user_query: str = "") -> str:
    """
    Асинхронный вызов LLM с обработкой ошибок и ретраями.
    """
    try:
        system_msg = PROMPTS[role]
        
        # Инъекция feedback в промпт для Solvers
        feedback_context = ""
        if role in ["TRIZ", "SYSTEM", "CRITIC"] and "FEEDBACK:" in context:
             feedback_context = f"\nВАЖНОЕ УТОЧНЕНИЕ ОТ ПОЛЬЗОВАТЕЛЯ: {context}"
             system_msg = system_msg.format(feedback_context=feedback_context)
        elif role in ["TRIZ", "SYSTEM", "CRITIC"]:
             # If no feedback, we still need to format the placeholder if it exists in the template
             system_msg = system_msg.format(feedback_context="")

        # Для синтезатора (безопасная замена)
        # Мы НЕ будем делать replace здесь, чтобы избежать ошибок парсинга фигурных скобок в данных.
        # Вместо этого мы будем передавать research_data как input variable в chain.
        # Но call_llm_async спроектирован под простой input string.
        # Если role == SYNTHESIZER, мы ожидаем что user_query - это словарь или JSON?
        # Нет, в node_synthesizer мы переделаем вызов.
        # Здесь оставляем стандартную логику, но если вдруг вызовут - ничего не сломается, просто {research_data} останется текстом.
        pass

        prompt_msgs = [("system", system_msg), ("user", "{input}")]
        prompt = ChatPromptTemplate.from_messages(prompt_msgs)
        chain = prompt | llm | StrOutputParser()
        
        return await _call_llm_with_retry(chain, {"input": user_query if user_query else context})

    except RetryError:
        return "⚠️ Сервис временно недоступен (все попытки исчерпаны)."
    except Exception as e:
        return f"⚠️ Ошибка: {str(e)}"

# --- 3. ГРАФ (STATE) ---

class AgentState(TypedDict):
    """Состояние агента, передаваемое между узлами графа."""
    messages: List[BaseMessage]   # История сообщений
    user_query: str               # Текущий запрос (последний)
    original_task: str            # Оригинальная задача (для Retry)
    mode: str                     # Режим (SOLVER, CHITCHAT, etc)
    triz_out: str
    system_out: str
    critic_out: str
    research_output: str          # Результаты поиска
    feedback: str                 # Инструкции от PostMortem
    final_verdict: str

# --- 4. УЗЛЫ (NODES) ---

async def node_orchestrator(state: AgentState):
    query = state['user_query']
    
    with Progress(SpinnerColumn(), TextColumn("[cyan]Оркестратор: Классификация запроса..."), console=console, transient=True) as progress:
        progress.add_task("think", total=None)
        mode = await call_llm_async("ORCHESTRATOR", "", query)
        mode = mode.strip().replace(".", "").upper()
    
    # Fallback / Cleaning
    valid_modes = ["CHITCHAT", "SOLVER", "THERAPIST", "CONSIGLIERE", "RETRY"]
    found = False
    for m in valid_modes:
        if m in mode:
            mode = m
            found = True
            break
    if not found:
        mode = "SOLVER" # Default

    color_map = {
        "CHITCHAT": "green", "SOLVER": "blue", "THERAPIST": "magenta",
        "CONSIGLIERE": "red", "RETRY": "yellow"
    }
    console.print(Panel(f"Режим: [bold {color_map.get(mode, 'white')}]{mode}[/]", title="🧠 ОРКЕСТРАТОР", border_style="cyan"))
    
    return {"mode": mode}

async def node_therapist(state: AgentState):
    """Снимает эмоциональное напряжение, добавляет контекст."""
    query = state['user_query']
    with Progress(SpinnerColumn(), TextColumn("[magenta]Терапевт: Валидация эмоций..."), console=console, transient=True) as progress:
        progress.add_task("therapy", total=None)
        response = await call_llm_async("THERAPIST", "", query)

    console.print(Panel(response, title="❤️ Терапевт", border_style="magenta"))

    # Добавляем ответ терапевта в историю как AIMessage, чтобы Solvers его видели как контекст
    new_messages = state['messages'] + [AIMessage(content=f"[Терапевт]: {response}")]

    return {"messages": new_messages}

async def node_consigliere(state: AgentState):
    """Предупреждает о рисках."""
    query = state['user_query']
    with Progress(SpinnerColumn(), TextColumn("[red]Консильери: Оценка рисков..."), console=console, transient=True) as progress:
        progress.add_task("risk", total=None)
        response = await call_llm_async("CONSIGLIERE", "", query)

    console.print(Panel(response, title="🕶️ Консильери", border_style="red"))

    new_messages = state['messages'] + [AIMessage(content=f"[Консильери]: {response}")]
    return {"messages": new_messages}

async def node_post_mortem(state: AgentState):
    """Анализ провала и генерация инструкций."""
    # Собираем историю в строку для анализа
    history_text = "\n".join([f"{m.type}: {m.content}" for m in state['messages'][-5:]]) # Последние 5

    with Progress(SpinnerColumn(), TextColumn("[yellow]Post-Mortem: Анализ ошибок..."), console=console, transient=True) as progress:
        progress.add_task("analyze", total=None)
        feedback = await call_llm_async("POST_MORTEM", history_text)

    console.print(Panel(feedback, title="🔄 Работа над ошибками", border_style="yellow"))
    return {"feedback": feedback}

async def node_solvers(state: AgentState):
    query = state['user_query']
    original_task = state.get('original_task', "")
    feedback = state.get('feedback', "")
    messages = state.get('messages', [])
    mode = state.get('mode', "")

    # Определяем реальную задачу. Если это RETRY, то берем original_task
    current_task = query
    if mode == "RETRY" and original_task:
        current_task = original_task
        console.print(f"[italic grey50]Использую оригинальную задачу для повтора: {current_task}[/]")

    # Формируем контекст с учетом последнего сообщения от Терапевта/Консильери
    context_prefix = ""
    if messages:
        last_msg = messages[-1]
        # Проверяем, является ли последнее сообщение "нашим" (AIMessage) и содержит ли маркеры
        if isinstance(last_msg, AIMessage):
            content = last_msg.content
            if "[Терапевт]" in content or "[Консильери]" in content:
                context_prefix = f"PREVIOUS CONTEXT (MUST CONSIDER): {content}\n"
    
    context_for_agents = f"{context_prefix}USER TASK: {current_task}"

    # Если есть feedback, добавляем его к запросу
    if feedback:
        context_for_agents = f"FEEDBACK: {feedback}\n{context_for_agents}"
        console.print(f"[italic yellow]Применяю обратную связь: {feedback}[/]")

    console.print("[bold]Запуск параллельных агентов...[/]")
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
        progress.add_task("[green]ТРИЗ генерирует идею...", total=None)
        progress.add_task("[blue]Системный анализ...", total=None)
        progress.add_task("[red]Поиск рисков...", total=None)
        
        triz_res, sys_res, crit_res = await asyncio.gather(
            call_llm_async("TRIZ", context_for_agents, context_for_agents),
            call_llm_async("SYSTEM", context_for_agents, context_for_agents),
            call_llm_async("CRITIC", context_for_agents, context_for_agents)
        )

    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    grid.add_row(
        Panel(triz_res, title="💡 ТРИЗ", border_style="green"),
        Panel(sys_res, title="⚙️ Системный", border_style="blue")
    )
    console.print(grid)
    console.print(Panel(crit_res, title="🛡️ Критик", border_style="red"))
    
    return {"triz_out": triz_res, "system_out": sys_res, "critic_out": crit_res}

async def node_fact_checker(state: AgentState):
    """Извлекает факты и проверяет их через Web Search."""
    # 1. Извлечь ключевые утверждения (упрощенно: просто берем идею ТРИЗ)
    search_query = state['triz_out'][:100] # Первые 100 символов идеи

    with Progress(SpinnerColumn(), TextColumn("[cyan]Fact Checker: Проверка фактов в Web..."), console=console, transient=True) as progress:
        progress.add_task("search", total=None)
        try:
            # Запускаем синхронный тул в тредпуле
            search_res = await asyncio.to_thread(search.invoke, search_query)
        except Exception as e:
            search_res = f"Ошибка поиска: {e}"

    # Ограничим вывод
    snippet = search_res[:300] + "..." if len(search_res) > 300 else search_res
    console.print(Panel(snippet, title="🌐 Web Search (DuckDuckGo)", border_style="cyan"))

    return {"research_output": search_res}

async def node_synthesizer(state: AgentState):
    # Используем безопасное форматирование переменных LangChain
    system_msg = PROMPTS["SYNTHESIZER"]
    research_data = state.get("research_output", "Нет данных")
    
    context = f"""
    Запрос: {state['user_query']}
    ТРИЗ: {state['triz_out']}
    Система: {state['system_out']}
    Критик: {state['critic_out']}
    """
    
    with Progress(SpinnerColumn(), TextColumn("[magenta]Синтез финального решения..."), console=console, transient=True) as progress:
         progress.add_task("synth", total=None)

         # Создаем шаблон, где {research_data} - это input variable
         prompt = ChatPromptTemplate.from_messages([("system", system_msg), ("user", "{input}")])
         chain = prompt | llm | StrOutputParser()

         # Передаем research_data как переменную
         verdict = await _call_llm_with_retry(chain, {
             "input": context,
             "research_data": research_data
         })

    return {"final_verdict": verdict}

# --- 5. СБОРКА ГРАФА (WORKFLOW) ---

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
    return "solvers" # Default SOLVER

workflow.add_conditional_edges("orchestrator", route, {
    END: END,
    "therapist": "therapist",
    "consigliere": "consigliere",
    "post_mortem": "post_mortem",
    "solvers": "solvers"
})

# Pass-through edges
workflow.add_edge("therapist", "solvers")
workflow.add_edge("consigliere", "solvers")
workflow.add_edge("post_mortem", "solvers")

# Core Logic
workflow.add_edge("solvers", "fact_checker")
workflow.add_edge("fact_checker", "synthesizer")
workflow.add_edge("synthesizer", END)

app = workflow.compile()

# --- 6. ЗАПУСК (MAIN) ---

async def main():
    console.clear()
    console.print(Panel.fit("[bold white]EPISTEMIC ENGINE v3.0 (OpenRouter Edition)[/]\n[grey50]Powered by LangGraph & GPT-4o[/]", border_style="green"))
    console.print("[italic grey50]Введите 'exit' для выхода.[/]\n")

    # Persistent memory session
    chat_history = []

    # Храним последнюю значимую задачу для логики Retry
    last_valid_task = ""

    while True:
        try:
            q = await asyncio.get_event_loop().run_in_executor(None, input, ">> Вы: ")

            if q.lower() in ['exit', 'quit', 'выход']: break
            if not q.strip(): continue
            
            console.rule("[bold cyan]Обработка[/]")
            
            # Добавляем вопрос пользователя в историю
            chat_history.append(HumanMessage(content=q))

            initial_state = {
                "messages": chat_history,
                "user_query": q,
                "original_task": last_valid_task,
                "mode": "", "triz_out": "", "system_out": "", "critic_out": "",
                "research_output": "", "feedback": "", "final_verdict": ""
            }
            
            final_state = await app.ainvoke(initial_state)
            
            # Обновляем историю сообщений из состояния (там могли добавиться сообщения Терапевта/Консильери)
            chat_history = final_state['messages']

            # Если это был рабочий режим (не болтовня и не повтор), запоминаем задачу как "оригинал"
            # Это эвристика: считаем, что если мы дошли до вердикта в режимах SOLVER/THERAPIST/CONSIGLIERE, то это была задача.
            if final_state['mode'] in ["SOLVER", "THERAPIST", "CONSIGLIERE"]:
                last_valid_task = q

            if final_state['mode'] == "CHITCHAT":
                response = "Привет! Я готов решать сложные задачи. Введи свой бизнес-запрос."
                console.print(Panel(response, title="🤖 Ассистент", border_style="green"))
                chat_history.append(AIMessage(content=response))
            else:
                verdict = final_state['final_verdict']
                console.rule("[bold green]ИТОГОВОЕ РЕШЕНИЕ[/]")
                console.print(Panel(Markdown(verdict), border_style="bold green"))
                chat_history.append(AIMessage(content=verdict))
            
            print("\n")

        except KeyboardInterrupt:
            console.print("\n[bold red]Завершение работы...[/]")
            break
        except EOFError:
             break

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
