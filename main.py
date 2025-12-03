# !pip install langchain-openai langgraph langchain-core python-dotenv rich nest_asyncio

import nest_asyncio
nest_asyncio.apply()

import os
import asyncio
import sys
from typing import List, TypedDict, Dict, Optional

# Загрузка переменных окружения
from dotenv import load_dotenv

# LangChain & LangGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# Надежность (Retries)
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

# Визуализация (Rich)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.layout import Layout

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

console = Console()

# --- 1. ПРОМПТЫ (SYSTEM PROMPTS) ---
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
    Собери их в единую рекомендацию (Итоговое Решение).
    Напиши ответ в формате Markdown, выделяя главное жирным. Не более 50 слов.
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
    Возвращает текст ответа или сообщение об ошибке, если все попытки исчерпаны.
    """
    try:
        system_msg = PROMPTS[role]
        # Для синтезатора контекст - это ответы других агентов, для остальных - вопрос юзера
        content = context if role == "SYNTHESIZER" else user_query
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("user", "{input}")
        ])
        chain = prompt | llm | StrOutputParser()
        
        # Вызов с ретраем
        return await _call_llm_with_retry(chain, {"input": content})

    except RetryError:
        return "⚠️ Сервис временно недоступен (все попытки исчерпаны)."
    except Exception as e:
        # Ловим любые другие неожиданные ошибки
        return f"⚠️ Ошибка: {str(e)}"

# --- 3. ГРАФ (STATE) ---

class AgentState(TypedDict):
    """Состояние агента, передаваемое между узлами графа."""
    user_query: str
    mode: str
    triz_out: str
    system_out: str
    critic_out: str
    final_verdict: str

# --- 4. УЗЛЫ (NODES) ---

async def node_orchestrator(state: AgentState):
    """
    Узел Оркестратора: Определяет тип запроса (Болтовня или Задача).
    """
    query = state['user_query']
    
    # Визуализация мыслительного процесса
    with Progress(SpinnerColumn(), TextColumn("[cyan]Оркестратор: Классификация запроса..."), console=console, transient=True) as progress:
        progress.add_task("think", total=None)
        mode = await call_llm_async("ORCHESTRATOR", "", query)

        # Очистка вывода от лишних символов
        mode = mode.strip().replace(".", "").upper()
    
    # Фоллбек логика
    if "CHITCHAT" in mode:
        mode = "CHITCHAT"
    elif "⚠️" in mode:
        # Если произошла ошибка в LLM, лучше по умолчанию попробовать решить задачу,
        # но в реальном проде стоит сообщить об ошибке.
        mode = "SOLVER"
    else:
        mode = "SOLVER"

    color = "green" if mode == "CHITCHAT" else "yellow"
    console.print(Panel(f"Режим: [bold {color}]{mode}[/]", title="🧠 ОРКЕСТРАТОР", border_style="cyan"))
    
    return {"mode": mode}

async def node_solvers(state: AgentState):
    """
    Узел Решателей: Запускает 3 агентов параллельно (ТРИЗ, Системный, Критик).
    """
    query = state['user_query']
    
    console.print("[bold]Запуск параллельных агентов...[/]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        # Создаем задачи для индикатора прогресса
        progress.add_task("[green]ТРИЗ генерирует идею...", total=None)
        progress.add_task("[blue]Системный анализ...", total=None)
        progress.add_task("[red]Поиск рисков...", total=None)
        
        # Await gather - ждем всех сразу (Параллельное выполнение)
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
        Panel(triz_res, title="💡 Агент ТРИЗ", border_style="green"),
        Panel(sys_res, title="⚙️ Системный Аналитик", border_style="blue")
    )
    console.print(grid)
    console.print(Panel(crit_res, title="🛡️ Критик", border_style="red"))
    
    return {"triz_out": triz_res, "system_out": sys_res, "critic_out": crit_res}

async def node_synthesizer(state: AgentState):
    """
    Узел Синтезатора: Объединяет все мнения в итоговый ответ.
    """
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

# --- 5. СБОРКА ГРАФА (WORKFLOW) ---

workflow = StateGraph(AgentState)

workflow.add_node("orchestrator", node_orchestrator)
workflow.add_node("solvers", node_solvers)
workflow.add_node("synthesizer", node_synthesizer)

workflow.set_entry_point("orchestrator")

def route(state):
    """Маршрутизация на основе решения Оркестратора"""
    if state['mode'] == "CHITCHAT": return END
    return "solvers"

workflow.add_conditional_edges("orchestrator", route, {END: END, "solvers": "solvers"})
workflow.add_edge("solvers", "synthesizer")
workflow.add_edge("synthesizer", END)

app = workflow.compile()

# --- 6. ЗАПУСК (MAIN) ---

async def main():
    console.clear()
    console.print(Panel.fit("[bold white]EPISTEMIC ENGINE v3.0 (OpenRouter Edition)[/]\n[grey50]Powered by LangGraph & GPT-4o[/]", border_style="green"))
    console.print("[italic grey50]Введите 'exit' для выхода.[/]\n")

    while True:
        try:
            # Асинхронный ввод, чтобы не блокировать event loop
            q = await asyncio.get_event_loop().run_in_executor(None, input, ">> Вы: ")

            if q.lower() in ['exit', 'quit', 'выход']: break
            if not q.strip(): continue
            
            console.rule("[bold cyan]Обработка[/]")
            
            initial_state = {
                "user_query": q,
                "mode": "", "triz_out": "", "system_out": "", "critic_out": "", "final_verdict": ""
            }
            
            # Запуск асинхронного графа
            final_state = await app.ainvoke(initial_state)
            
            # Если был чат-бот
            if final_state['mode'] == "CHITCHAT":
                console.print(Panel("Привет! Я готов решать сложные задачи. Введи свой бизнес-запрос.", title="🤖 Ассистент", border_style="green"))
            else:
                console.rule("[bold green]ИТОГОВОЕ РЕШЕНИЕ[/]")
                console.print(Panel(final_state['final_verdict'], border_style="bold green"))
            
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
