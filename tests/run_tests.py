import sys
import os
import traceback

# Add current directory to path
sys.path.insert(0, os.path.abspath('.'))

from tests.test_features import (
    test_bomb_danger_map,
    test_bfs_pathfinding,
    test_can_escape_bomb,
    test_qlearning_feature_extraction,
    test_dqn_feature_extraction
)

def main():
    tests = [
        ("test_bomb_danger_map", test_bomb_danger_map),
        ("test_bfs_pathfinding", test_bfs_pathfinding),
        ("test_can_escape_bomb", test_can_escape_bomb),
        ("test_qlearning_feature_extraction", test_qlearning_feature_extraction),
        ("test_dqn_feature_extraction", test_dqn_feature_extraction)
    ]
    
    passed = 0
    failed = 0
    print("Running feature engineering unit tests...")
    print("==========================================")
    for name, test_func in tests:
        try:
            test_func()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as err:
            print(f"[FAIL] {name}: {err}")
            traceback.print_exc()
            failed += 1

    print("==========================================")
    print(f"Results: {passed} passed, {failed} failed.")
    if failed > 0:
        sys.exit(1)

if __name__ == '__main__':
    main()
