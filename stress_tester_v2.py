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

    # We append 'exit' to ensure the loop terminates
    full_input = input_text + "\nexit\n"
    stdout, stderr = process.communicate(input=full_input)

    if process.returncode != 0:
        print(f"FAILED with return code {process.returncode}")
        print("STDERR:", stderr)
    else:
        print("SUCCESS (Return Code 0)")
        if "ИТОГОВОЕ РЕШЕНИЕ" in stdout:
            print("Verdict generated.")
        elif "Режим: CHITCHAT" in stdout:
             print("Detected CHITCHAT mode.")
        else:
            print("No verdict found.")
            # print("STDOUT snippet:", stdout[-1000:])

    return stdout, stderr

async def main():
    # 2. Long input
    long_text = "слово " * 1000 # 6000 chars
    await run_test(long_text, "Long Input (6000 chars)")

    # 5. Bad API Key
    await run_test("Почему небо голубое?", "Bad API Key", {"OPENROUTER_API_KEY": "sk-bad-key"})

if __name__ == "__main__":
    asyncio.run(main())
