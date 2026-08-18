import os
import zipfile
import argparse


def package_agent(agent_name="project_dqn", output_zip="final-project-agent-code.zip"):
    agent_dir = os.path.join("agent_code", agent_name)
    if not os.path.exists(agent_dir):
        raise FileNotFoundError(f"Agent directory {agent_dir} does not exist.")

    callbacks_path = os.path.join(agent_dir, "callbacks.py")
    if not os.path.exists(callbacks_path):
        raise FileNotFoundError(f"Required callbacks.py missing in {agent_dir}")

    print(f"Packaging best agent '{agent_name}' into '{output_zip}'...")

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(agent_dir):
            for file in files:
                # Exclude logs or pycache
                if '__pycache__' in root or file.endswith('.pyc') or 'logs' in root:
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start="agent_code")
                zipf.write(file_path, arcname)
                print(f"  + Added: {arcname}")

    print(f"\nSUCCESS: Submitted agent ZIP created at '{os.path.abspath(output_zip)}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Package submitted agent directory into final ZIP")
    parser.add_argument("--agent", type=str, default="project_dqn", help="Agent folder name")
    parser.add_argument("--out", type=str, default="final-project-agent-code.zip", help="Output ZIP file path")
    args = parser.parse_args()

    package_agent(args.agent, args.out)
