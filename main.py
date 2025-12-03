# !pip install langchain-openai langgraph langchain-core python-dotenv rich nest_asyncio

import nest_asyncio
nest_asyncio.apply()


import os
import asyncio
import sys
from typing import List, TypedDict, Dict

# Загрузка переменных окружения
from dotenv import load_dotenv

# LangChain & LangGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# Визуализация (Rich)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.layout import Layout

# --- НАСТРОЙКА ---
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    print("ОШИБКА: Не найден OPENROUTER_API_KEY в файле .env")
    sys.exit(1)

# Инициализация модели через OpenRouter
# Мы используем класс ChatOpenAI, но меняем base_url
llm = ChatOpenAI(
    model="openai/gpt-4o",  # Можно поменять на "anthropic/claude-3.5-sonnet"
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/Start_AI", # Для статистики OpenRouter
        "X-Title": "Epistemic Engine v3"
    },
    temperature=0.7
)

console = Console()

# --- 1. ПРОМПТЫ (МОЗГИ АГЕНТОВ) ---
PROMPTS = {
    "ORCHESTRATOR": """
    Ты — Оркестратор системы принятия решений. Классифицируй запрос.
    1. Если это приветствие или болтовня -> верни "CHITCHAT".
    2. Если это конкретная задача/проблема -> верни "SOLVER".
    Верни ТОЛЬКО одно слово.
    """,

    "TRIZ": """
    Ты — Агент ТРИЗ (Теория решения изобретательских задач).
    Предложи 1 нестандартное, сильное решение, используя принципы ТРИЗ (Инверсия, Дробление, Посредник).
    Будь предельно краток (максимум 2 предложения).
    """,

    "SYSTEM": """
    Ты — Системный Аналитик.
    Найди 1 критическое узкое место (bottleneck) или разрыв в процессах для этой задачи.
    Используй термины: обратная связь, пропускная способность, ресурсы.
    Будь предельно краток (максимум 2 предложения).
    """,

    "CRITIC": """
    Ты — Риск-менеджер (Адвокат Дьявола).
    Найди 1 самый опасный риск в реализации этой задачи (финансы, репутация, закон).
    Начни ответ со слов "РИСК:".
    Будь предельно краток (максимум 2 предложения).
    """,
    
    "SYNTHESIZER": """
    Ты — Синтезатор решений.
    У тебя есть три мнения: ТРИЗ (Идея), Системное (Процесс) и Критика (Риск).
    Собери их в единую рекомендацию (Final Verdict).
    Напиши ответ в формате Markdown, выделяя главное жирным. Не более 50 слов.
    """
}

# --- 2. ЛОГИКА LLM (ASYNC) ---

async def call_llm_async(role: str, context: str, user_query: str = "") -> str:
    """Асинхронный вызов OpenRouter"""
    try:
        system_msg = PROMPTS[role]
        # Для синтезатора контекст - это ответы других агентов, для остальных - вопрос юзера
        content = context if role == "SYNTHESIZER" else user_query
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("user", "{input}")
        ])
        chain = prompt | llm | StrOutputParser()
        
        # Реальный вызов
        return await chain.ainvoke({"input": content})
    except Exception as e:
        return f"Error: {str(e)}"

# --- 3. ГРАФ (STATE) ---

class AgentState(TypedDict):
    user_query: str
    mode: str
    triz_out: str
    system_out: str
    critic_out: str
    final_verdict: str

# --- 4. УЗЛЫ (NODES) ---

async def node_orchestrator(state: AgentState):
    query = state['user_query']
    
    # Визуализация мыслительного процесса
    with Progress(SpinnerColumn(), TextColumn("[cyan]Оркестратор: Классификация запроса..."), console=console, transient=True) as progress:
        progress.add_task("think", total=None)
        mode = await call_llm_async("ORCHESTRATOR", "", query)
        mode = mode.strip().replace(".", "").upper()
    
    # Фоллбек, если LLM вернет мусор
    if "CHITCHAT" in mode: mode = "CHITCHAT"
    else: mode = "SOLVER"

    color = "green" if mode == "CHITCHAT" else "yellow"
    console.print(Panel(f"Режим: [bold {color}]{mode}[/]", title="🧠 ORCHESTRATOR", border_style="cyan"))
    
    return {"mode": mode}

async def node_solvers(state: AgentState):
    query = state['user_query']
    
    console.print("[bold]Запуск параллельных агентов...[/]")
    
    # ПАРАЛЛЕЛЬНОЕ ВЫПОЛНЕНИЕ (Real Async)
    # Мы запускаем 3 запроса к OpenRouter одновременно
    
    results = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        # Создаем задачи
        task_triz = progress.add_task("[green]ТРИЗ генерирует идею...", total=None)
        task_sys = progress.add_task("[blue]Системный анализ...", total=None)
        task_crit = progress.add_task("[red]Поиск рисков...", total=None)
        
        # Await gather - ждем всех сразу
        # Это сокращает время ожидания в 3 раза
        triz_res, sys_res, crit_res = await asyncio.gather(
            call_llm_async("TRIZ", "", query),
            call_llm_async("SYSTEM", "", query),
            call_llm_async("CRITIC", "", query)
        )
        
    # Вывод результатов в красивой таблице
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    
    grid.add_row(
        Panel(triz_res, title="💡 TRIZ Agent", border_style="green"),
        Panel(sys_res, title="⚙️ System Agent", border_style="blue")
    )
    console.print(grid)
    console.print(Panel(crit_res, title="🛡️ Critic Agent", border_style="red"))
    
    return {"triz_out": triz_res, "system_out": sys_res, "critic_out": crit_res}

async def node_synthesizer(state: AgentState):
    # Собираем контекст для синтезатора
    context = f"""
    Запрос пользователя: {state['user_query']}
    
    Мнение ТРИЗ: {state['triz_out']}
    Мнение Системщика: {state['system_out']}
    Мнение Критика: {state['critic_out']}
    """
    
    with Progress(SpinnerColumn(), TextColumn("[magenta]Синтез финального решения..."), console=console, transient=True) as progress:
        progress.add_task("synth", total=None)
        verdict = await call_llm_async("SYNTHESIZER", context)
        
    return {"final_verdict": verdict}

# --- 5. СБОРКА ГРАФА ---

workflow = StateGraph(AgentState)

workflow.add_node("orchestrator", node_orchestrator)
workflow.add_node("solvers", node_solvers)
workflow.add_node("synthesizer", node_synthesizer)

workflow.set_entry_point("orchestrator")

def route(state):
    if state['mode'] == "CHITCHAT": return END
    return "solvers"

workflow.add_conditional_edges("orchestrator", route, {END: END, "solvers": "solvers"})
workflow.add_edge("solvers", "synthesizer")
workflow.add_edge("synthesizer", END)

app = workflow.compile()

# --- 6. ЗАПУСК ---

async def main():
    console.clear()
    console.print(Panel.fit("[bold white]EPISTEMIC ENGINE v3.0 (OpenRouter Edition)[/]\n[grey50]Powered by LangGraph & GPT-4o[/]", border_style="green"))
    console.print("[italic grey50]Type 'exit' to quit.[/]\n")

    while True:
        try:
            q = await asyncio.get_event_loop().run_in_executor(None, input, ">> User: ")
            if q.lower() in ['exit', 'quit']: break
            if not q.strip(): continue
            
            console.rule("[bold cyan]Processing[/]")
            
            initial_state = {
                "user_query": q,
                "mode": "", "triz_out": "", "system_out": "", "critic_out": "", "final_verdict": ""
            }
            
            # Запуск асинхронного графа
            final_state = await app.ainvoke(initial_state)
            
            # Если был чат-бот, просто напишем приветствие (для демо экономим токены на синтезе)
            if final_state['mode'] == "CHITCHAT":
                console.print(Panel("Привет! Я готов решать сложные задачи. Введи свой бизнес-запрос.", title="🤖 Assistant", border_style="green"))
            else:
                console.rule("[bold green]FINAL VERDICT[/]")
                console.print(Panel(final_state['final_verdict'], border_style="bold green"))
            
            print("\n")

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    asyncio.run(main())

