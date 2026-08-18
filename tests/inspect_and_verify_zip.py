import os
import shutil
import sys
import tempfile
import zipfile
import subprocess

# Ensure root is in python path
sys.path.insert(0, os.path.abspath('.'))
from create_submission_zip import package_agent


def inspect_and_verify_zip():
    zip_path = "final-project-agent-code.zip"
    package_agent("project_dqn", zip_path)

    print("\n=======================================================")
    print("INSPECTING SUBMISSION ZIP FILE TREE")
    print("=======================================================\n")

    if not os.path.exists(zip_path):
        print("FAIL: ZIP file does not exist!")
        return False, []

    file_list = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        file_list = zf.namelist()

    print(f"Zip File: {os.path.abspath(zip_path)} ({os.path.getsize(zip_path)} bytes)")
    print("Files inside ZIP:")
    for f in sorted(file_list):
        print(f"  - {f}")

    # Validation 1: Check root prefix
    for f in file_list:
        if not f.startswith("project_dqn/"):
            print(f"FAIL: Found file outside project_dqn directory: {f}")
            return False, file_list

    # Validation 2: Check required files
    required_files = [
        "project_dqn/callbacks.py",
        "project_dqn/train.py",
        "project_dqn/features.py",
        "project_dqn/model.py",
        "project_dqn/replay_buffer.py",
        "project_dqn/utils.py",
        "project_dqn/model_data/model.pt"
    ]
    for rf in required_files:
        if rf not in file_list:
            print(f"FAIL: Missing required file in ZIP: {rf}")
            return False, file_list

    # Validation 3: Check forbidden files
    forbidden_keywords = ['main.py', 'environment.py', 'experiments', 'tests', 'report_data', 'README.md']
    for f in file_list:
        for fk in forbidden_keywords:
            if fk in f:
                print(f"FAIL: Found forbidden framework/experiment file in ZIP: {f}")
                return False, file_list

    print("\nPASS: ZIP file structure contains ONLY the best trained agent directory!\n")

    # Clean Extraction Test
    temp_dir = os.path.join(tempfile.gettempdir(), 'bomberman_zip_audit_clean')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    print(f"Executing Clean Extraction Test into: {temp_dir}")
    
    agent_code_dir = os.path.join(temp_dir, 'agent_code')
    os.makedirs(agent_code_dir, exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'logs'), exist_ok=True)

    # A. Framework files copied to temp host environment
    root_dir = os.path.abspath('.')
    framework_files = ['main.py', 'environment.py', 'settings.py', 'events.py', 'items.py', 'fallbacks.py', 'agents.py', 'replay.py']
    for ff in framework_files:
        shutil.copy(os.path.join(root_dir, ff), os.path.join(temp_dir, ff))

    shutil.copytree(os.path.join(root_dir, 'agent_code', 'random_agent'), os.path.join(temp_dir, 'agent_code', 'random_agent'))

    # B. Submitted ZIP unzipped into temp_dir/agent_code/
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(agent_code_dir)

    # Run game with submitted agent from unzipped path
    cmd = [
        sys.executable, 'main.py', 'play',
        '--agents', 'project_dqn', 'random_agent', 'random_agent', 'random_agent',
        '--no-gui', '--n-rounds', '3'
    ]

    print(f"Running tournament simulation with unzipped agent...")
    proc = subprocess.run(cmd, cwd=temp_dir, capture_output=True, text=True)

    if proc.returncode == 0:
        print("PASS: Unzipped submitted agent executed cleanly with 0 errors!")
        return True, file_list
    else:
        print(f"FAIL: Execution returned exit code {proc.returncode}")
        print("Stderr:", proc.stderr)
        return False, file_list


if __name__ == '__main__':
    ok, files = inspect_and_verify_zip()
    if not ok:
        sys.exit(1)
