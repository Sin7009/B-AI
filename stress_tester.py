import subprocess
import asyncio
import os

async def run_test(input_text, description):
    print(f"--- Running Test: {description} ---")
    process = subprocess.Popen(
        ['python', 'main.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ
    )

    stdout, stderr = process.communicate(input=input_text + "\nexit\n")

    if process.returncode != 0:
        print(f"FAILED with return code {process.returncode}")
        print("STDERR:", stderr)
    else:
        print("SUCCESS")
        # Check if output contains expected parts (rudimentary check)
        if "FINAL VERDICT" in stdout:
            print("Verdict generated.")
        else:
            print("No verdict found (might be intended for chitchat or error).")
            # print("STDOUT snippet:", stdout[-500:])

    return stdout, stderr

async def main():
    # 1. Empty input
    await run_test("\n", "Empty Input")

    # 2. Long input
    long_text = "слово " * 500
    await run_test(long_text, "Long Input (3000 chars)")

    # 3. Special chars
    await run_test("System error! 🤖 /drop database", "Special Chars / Injection attempt")

    # 4. English Input
    await run_test("How to bake a cake?", "English Input")

if __name__ == "__main__":
    asyncio.run(main())
