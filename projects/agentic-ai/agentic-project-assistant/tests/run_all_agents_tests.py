import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import subprocess


def main() -> None:
    """Run all agent test scripts in sequence."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(base_dir, "./")
    output_file = os.path.join(base_dir, "test_results.txt")

    test_scripts = [
        "action_planning_agent.py",
        "augmented_prompt_agent.py",
        "direct_prompt_agent.py",
        "evaluation_agent.py",
        "knowledge_augmented_prompt_agent.py",
        "rag_knowledge_prompt_agent.py",
        "routing_agent.py",
    ]

    with open(output_file, "w", encoding="utf-8") as out_file:
        for script in test_scripts:
            script_path = os.path.join(test_dir, script)
            if not os.path.exists(script_path):
                print(f"❌ Script not found: {script_path}")
                continue

            header = f"\n{'='*40}\nRunning: {script}\n{'='*40}\n"
            print(header)
            out_file.write(header)

            try:
                result = subprocess.run(
                    ["python", script_path],
                    capture_output=True,
                    text=True,
                    cwd=test_dir,
                    check=False,
                )

                output = result.stdout + result.stderr
                print(output)
                out_file.write(output)

            except Exception as e:
                error_msg = f"\n[ERROR] Could not run {script}: {e}\n"
                print(error_msg)
                out_file.write(error_msg)

    print(f"\n✅ All tests completed. Results written to {output_file}")


if __name__ == "__main__":
    main()
