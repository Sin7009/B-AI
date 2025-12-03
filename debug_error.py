import subprocess
import asyncio
import os
import sys

async def run_test(input_text, description, env_overrides=None):
    print(f"\n--- Running Test: {description} ---")

    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    process = subprocess.Popen(
        [sys.executable, 'main.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )

    full_input = input_text + "\nexit\n"
    stdout, stderr = process.communicate(input=full_input)

    print("STDOUT LAST 10 LINES:")
    lines = stdout.split('\n')
    for line in lines[-20:]:
        print(line)

async def main():
    await run_test("Тест ошибки", "Bad API Key", {"OPENROUTER_API_KEY": "sk-bad-key"})

if __name__ == "__main__":
    asyncio.run(main())
