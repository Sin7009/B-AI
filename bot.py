import os
import asyncio
import logging
import sys
import time

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# Import local modules
from engine import get_graph, AgentState
from database import db, DATABASE_URL

# Setup Logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Load Env
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN not found in .env")
    sys.exit(1)

# Initialize Bot and Dispatcher
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# Global checkpointer variable
checkpointer = None

# --- UTILS ---
def format_progress_message(state_update: dict, current_text: str) -> str:
    """Updates the status message based on what nodes just finished."""
    new_text = current_text

    # Check for outputs and append checkboxes
    if "triz_out" in state_update:
        new_text += "\n✅ ТРИЗ сгенерировал идею"
    if "system_out" in state_update:
        new_text += "\n✅ Системный анализ завершен"
    if "critic_out" in state_update:
        new_text += "\n✅ Риски оценены"
    if "research_output" in state_update:
        new_text += "\n✅ Факты проверены"

    return new_text

# --- HANDLERS ---

@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    user = message.from_user
    # Register in DB
    try:
        db_user = await db.register_or_update_user(
            telegram_id=user.id,
            username=user.username,
            full_name=user.full_name
        )
        logger.info(f"User {user.id} registered/updated.")
    except Exception as e:
        logger.error(f"DB Error: {e}")
        await message.answer("⚠️ Ошибка базы данных.")
        return

    welcome_text = (
        f"Привет, <b>{user.first_name}</b>! 👋\n\n"
        "Я — Epistemic Engine v3.0. Я помогу тебе решить сложную бизнес-задачу, "
        "используя ТРИЗ, Системный анализ и Критическое мышление.\n\n"
        "Просто опиши свою проблему, и я запущу команду агентов."
    )
    await message.answer(welcome_text)

@dp.message()
async def handle_message(message: types.Message):
    user_id = message.from_user.id
    query = message.text

    # 1. Update User Activity
    await db.register_or_update_user(user_id, message.from_user.username, message.from_user.full_name)

    # 2. Prepare Graph
    # Use the persistent connection pool from global checkpointer
    graph = get_graph(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": str(user_id)}}

    # Check existing state to see if we have context
    # (Optional: Logic to clear history could go here)

    # 3. Send "Thinking" message
    status_msg = await message.answer("🧠 <b>Анализирую задачу...</b>")

    # 4. Stream Graph Execution
    final_verdict = ""
    last_valid_task = "" # Logic to track task similar to CLI

    # We need to construct the input state.
    # If it's a new conversation, we send messages. If continuing, LangGraph handles history via thread_id.

    # However, to pass the *new* message, we must provide it.
    input_state = {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
    }

    last_update_time = 0
    UPDATE_INTERVAL = 1.0  # Seconds

    try:
        async for event in graph.astream(input_state, config, stream_mode="values"):
            # 'event' is the full state at that point in time

            # Detect Node Completion by checking if fields are non-empty and differ from "last seen"?
            # Simpler approach: Just reconstruct the status message based on what's present.

            progress_text = "🧠 <b>Анализирую задачу...</b>"

            if event.get("mode"):
                mode = event["mode"]
                if mode == "CHITCHAT":
                    # Chitchat ends immediately usually
                    pass
                elif mode != "SOLVER":
                    progress_text += f"\n👉 Режим: {mode}"

            if event.get("triz_out"): progress_text += "\n✅ ТРИЗ сгенерировал идею"
            if event.get("system_out"): progress_text += "\n✅ Системный анализ завершен"
            if event.get("critic_out"): progress_text += "\n✅ Риски оценены"
            if event.get("research_output"): progress_text += "\n🔍 Факты проверены"

            # Throttling updates to avoid Telegram 429 Errors
            current_time = time.time()
            if (current_time - last_update_time >= UPDATE_INTERVAL):
                try:
                    # Only edit if text actually changed (aiogram handles this optimization often, but explicit is better)
                    if progress_text != status_msg.html_text:
                        await status_msg.edit_text(progress_text)
                        last_update_time = current_time
                except Exception as e:
                    logger.warning(f"Failed to update status message: {e}")

            # Capture verdict
            if event.get("final_verdict"):
                final_verdict = event["final_verdict"]
                # Force final update of status
                try:
                    if progress_text != status_msg.html_text:
                        await status_msg.edit_text(progress_text)
                except:
                    pass

            # Capture mode for immediate response
            if event.get("mode") == "CHITCHAT" and not final_verdict:
                pass

        # 5. Final Output
        # Delete progress message or keep it? Usually keep as log.

        currentState = await graph.get_state(config)
        state_values = currentState.values
        mode = state_values.get("mode")

        if mode == "CHITCHAT":
            await message.answer("🤖 Привет! Я готов решать сложные задачи. Введи свой бизнес-запрос.")

        elif final_verdict:
            # HTML Formatting
            # Replace markdown bold **text** with <b>text</b> if needed, or rely on aiogram's Markdown parser?
            # User asked for HTML. LLM generates Markdown.
            # Simple heuristic: Let's use aiogram's Markdown parser for the verdict, it's safer than converting.
            # But we promised HTML structure for the "thinking" parts.

            # Let's send the verdict as Markdown
            await message.answer(final_verdict, parse_mode=ParseMode.MARKDOWN)

            # Optional: Send specific agent outputs in expandable blocks if requested
            # Telegram doesn't support "expandable" blocks in standard messages yet (only spoilers).
            # We can use spoilers || hidden text ||.

            details = (
                f"<b>Подробности:</b>\n\n"
                f"💡 <b>ТРИЗ:</b> <tg-spoiler>{state_values.get('triz_out', '')}</tg-spoiler>\n\n"
                f"🛡️ <b>Критик:</b> <tg-spoiler>{state_values.get('critic_out', '')}</tg-spoiler>"
            )
            await message.answer(details)

    except Exception as e:
        logger.error(f"Graph Error: {e}")
        await message.answer(f"⚠️ Произошла ошибка при обработке: {e}")


# --- STARTUP ---
async def on_startup():
    global checkpointer

    # Init DB
    await db.init_db()

    # Init Checkpointer
    # We use the same connection string logic
    conn_string = DATABASE_URL.replace("+asyncpg", "") # AsyncPostgresSaver uses psycopg logic?
    # Wait, langgraph-checkpoint-postgres documentation says:
    # AsyncPostgresSaver.from_conn_string("postgresql://user:pass@host:5432/db")
    # Our URL has +asyncpg which might confuse it if it uses psycopg3 directly or asyncpg.
    # Documentation says it uses `psycopg-pool`.
    # Let's stick to standard postgres:// scheme for it if needed, but asyncpg scheme usually works for async libs.
    # Let's try passing the pool or connection string.

    # Actually, let's create the pool manually to be safe or use from_conn_string
    # Fix: AsyncPostgresSaver requires 'postgresql://' scheme, not 'postgresql+asyncpg://'
    checkpointer = AsyncPostgresSaver.from_conn_string(conn_string)

    # Need to call setup to create checkpoint tables
    # Note: from_conn_string is a context manager or returns an object?
    # In async it's usually `async with ...`.
    # But we need it to persist across requests.

    # Correct pattern:
    # async with AsyncPostgresSaver.from_conn_string(...) as checkpointer:
    #    ...
    # But we are in a long-running app.

    # We will instantiate it here.
    # NOTE: AsyncPostgresSaver needs to be entered (aentered) to set up the connection pool.
    await checkpointer.__aenter__()
    await checkpointer.setup()

async def on_shutdown():
    if checkpointer:
        await checkpointer.__aexit__(None, None, None)

# --- MAIN ---
async def main():
    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
