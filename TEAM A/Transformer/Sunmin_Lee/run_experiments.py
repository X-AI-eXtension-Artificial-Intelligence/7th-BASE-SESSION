# =============================================================================
# run_experiments.py
# 여러 실험 설정을 순차적으로 실행하는 스크립트
# 실험: base / small / large / custom(single-head) 비교
# =============================================================================

import os
import subprocess
import json
from datetime import datetime


# =============================================================================
# 실험 목록 정의
# conf.py의 experiment 변수를 바꿔가며 실행
# =============================================================================
EXPERIMENTS = ["base", "small", "custom"]   # large는 메모리 부족 시 제외

# 결과 저장 경로
RESULTS_DIR = "saved/experiment_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_single_experiment(exp_name):
    """
    단일 실험을 실행합니다.
    conf.py의 experiment 값을 변경 후 train.py 실행.
    """
    print(f"\n{'='*60}")
    print(f"  실험 시작: [{exp_name}]")
    print(f"  시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # conf.py의 experiment 값 변경
    _update_experiment_in_conf(exp_name)

    # train.py 실행
    result = subprocess.run(
        ["python", "train.py"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"  [오류] {exp_name} 실험 실패")
        print(result.stderr)
        return None

    # 결과 저장
    save_path = os.path.join(RESULTS_DIR, f"{exp_name}_log.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(result.stdout)

    print(f"  완료! 로그 저장: {save_path}")
    return result.stdout


def _update_experiment_in_conf(exp_name):
    """conf.py의 experiment 변수를 변경합니다."""
    with open("conf.py", "r", encoding="utf-8") as f:
        content = f.read()

    # experiment = "base" → experiment = "small" 등으로 교체
    import re
    content = re.sub(
        r'^experiment\s*=\s*["\'].*["\']',
        f'experiment = "{exp_name}"',
        content,
        flags=re.MULTILINE
    )

    with open("conf.py", "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  conf.py 업데이트: experiment = '{exp_name}'")


if __name__ == "__main__":
    print("\n[다중 실험 실행기]")
    print(f"실험 목록: {EXPERIMENTS}")
    print("각 실험이 순차적으로 실행됩니다.\n")

    for exp in EXPERIMENTS:
        run_single_experiment(exp)

    print("\n모든 실험 완료!")
    print(f"결과 저장 위치: {RESULTS_DIR}/")
