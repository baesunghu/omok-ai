"""Small-net experiment status (11x11_v2, 2026-10-21, user). Usage: python small_net_status.py [-w]
Prints the capacity results so far (round 1, round 2), distillation data progress, which experiment
processes are running and the GPU. -w refreshes every 5 s (Ctrl+C to quit).
"""
import os
import subprocess
import sys
import time

D = os.path.dirname(os.path.abspath(__file__))
if os.name == "nt":
    os.system("chcp 65001 >nul")
sys.stdout.reconfigure(encoding="utf-8")


def lines(name):
    try:
        with open(os.path.join(D, name), encoding="utf-8", errors="ignore") as f:
            return [l.rstrip() for l in f if l.strip() and "warn" not in l.lower()]
    except OSError:
        return []


def running():
    try:
        out = subprocess.run(["wmic", "process", "where", "name='python.exe'", "get", "CommandLine"],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return "?"
    names = {"small_net_capacity": "1차 용량실험", "distill_data_gen": "데이터 생성", "distill_train": "2차 증류학습",
             "_restart_train": "본 학습"}
    return "  ".join(f"{v} {'실행중' if k in out else '-'}" for k, v in names.items())


def gpu():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,memory.used",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=3).stdout.strip()
    except Exception:
        return "?"


def show():
    out = [f"=== 작은 모델 실험  {time.strftime('%m-%d %H:%M:%S')}  GPU(%, °C, MiB) {gpu()} ===", running(), ""]
    out.append("[1차 용량 실험] 선생 = best(ep960), 버퍼 62,440국면, 8에포크")
    r1 = [l for f in ("small_net_capacity.log", "small_net_capacity_part2.log") for l in lines(f) if "params |" in l]
    out += ["  " + l.strip() for l in r1] or ["  (결과 없음)"]
    out.append("")
    g = lines("distill_data_gen.log")
    out.append("[증류 데이터 생성] " + (g[-1] if g else "-"))
    out.append("")
    out.append("[2차 증류 학습] 100만 국면, 대국 단위 평가 분리")
    r2 = lines("distill_train.log")
    out += ["  " + l.strip() for l in r2] or ["  (아직 시작 전)"]
    out.append("")
    out.append("지표: top1 = 선생과 최선수 일치, KL 낮을수록 좋음, v_sign = 유불리 판단 일치,")
    out.append("      v_corr = 승률 예측 상관, v_rel = 평균값 대비 오차(낮을수록 좋음), rows/s = 초당 평가 국면")
    print("\n".join(out), flush=True)


if "-w" in sys.argv:
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        show()
        time.sleep(5)
else:
    show()
