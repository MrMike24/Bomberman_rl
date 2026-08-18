import os
import shutil
import sys
import tempfile
import zipfile
import subprocess

sys.path.insert(0, os.path.abspath('.'))
from create_submission_zip import package_agent


def test_unzip_and_execute():
    print("=======================================================")
    print("AUDIT ITEM 13 & 14: CLEAN TEMP UNZIP EXECUTION TEST")
    print("=======================================================")

    zip_file = "final-project-agent-code.zip"
    package_agent("project_dqn", zip_file)

    if not os.path.exists(zip_file):
        print("FAIL: ZIP file creation failed.")
        return False

    temp_dir = os.path.join(tempfile.gettempdir(), 'bomberman_clean_audit_test')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    print(f"Extracting '{zip_file}' into clean temporary directory '{temp_dir}'...")

    agent_code_dir = os.path.join(temp_dir, 'agent_code')
    os.makedirs(agent_code_dir, exist_ok=True)

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(agent_code_dir)

    print("Unzipped contents:")
    for root, dirs, files in os.walk(agent_code_dir):
        for f in files:
            print("  -", os.path.relpath(os.path.join(root, f), agent_code_dir))

    # Copy framework files to clean temp directory
    root_dir = os.path.abspath('.')
    os.makedirs(os.path.join(temp_dir, 'logs'), exist_ok=True)
    framework_files = ['main.py', 'environment.py', 'settings.py', 'events.py', 'items.py', 'fallbacks.py', 'agents.py', 'replay.py']
    for ff in framework_files:
        shutil.copy(os.path.join(root_dir, ff), os.path.join(temp_dir, ff))

    shutil.copytree(os.path.join(root_dir, 'agent_code', 'random_agent'), os.path.join(temp_dir, 'agent_code', 'random_agent'))
    shutil.copytree(os.path.join(root_dir, 'agent_code', 'rule_based_agent'), os.path.join(temp_dir, 'agent_code', 'rule_based_agent'))

    # Run game from temp directory
    cmd = [
        sys.executable, 'main.py', 'play',
        '--agents', 'project_dqn', 'random_agent', 'random_agent', 'random_agent',
        '--no-gui', '--n-rounds', '5'
    ]

    print(f"Running command inside temp directory: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=temp_dir, capture_output=True, text=True)

    print("Output:")
    print(proc.stdout)
    if proc.stderr:
        print("Stderr:", proc.stderr)

    if proc.returncode == 0:
        print("PASS: Unzipped submitted agent executed successfully in clean environment!")
        return True
    else:
        print(f"FAIL: Execution returned exit code {proc.returncode}")
        return False


if __name__ == '__main__':
    ok = test_unzip_and_execute()
    if not ok:
        sys.exit(1)
