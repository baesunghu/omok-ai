"""총괄 현황 한 번에 보기 (2026-10-10). 사용: python status_all.py   (반복: python status_all.py watch)"""
import json, os, re, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
SP = r"C:\Users\MYPC~1\AppData\Local\Temp\claude\C--Users-My-PC-Desktop-final\cdf58682-cb60-4179-92b0-f3acf6965694\scratchpad"


def tail(path, n):
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.readlines()[-n:]
    except OSError:
        return []


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20, encoding="utf-8", errors="ignore").stdout
    except Exception as e:
        return f"({e})"


def show():
    print("=" * 70); print("총괄 현황", time.strftime("%Y-%m-%d %H:%M:%S")); print("=" * 70)

    print("\n[학습] 최근 에폭")
    rep = [l for l in tail("model/training_report_11x11.txt", 400) if "] Epoch " in l]
    for l in rep[-4:]:
        m = re.search(r"\[(.*?)\] Epoch (\d+)/\d+ \| Policy Loss: ([\d.]+) \| Value Loss: ([\d.]+)", l)
        g = re.search(r"Gate vs best[^|]*", l)
        if m:
            print(f"  ep{m[2]}  {m[1][11:16]}  policy {m[3]}  value {m[4]}" + (f"  | {g[0].strip()}" if g else ""))
    gates = [l for l in rep if "PROMOTED" in l]
    if gates:
        print("  최근 승격:", re.search(r"Epoch (\d+)/", gates[-1])[1])

    print("\n[학습 설정 배포: q 0.2 + 반감기]")
    import glob
    newest = max(glob.glob("train_stdout_v*.log"), key=os.path.getmtime)
    v23 = tail(newest, 400)
    q = [l.strip() for l in v23 if "[value target]" in l]
    print("  " + (q[-1] if q else "아직 배포 전 (ep905 게이트 대기) 또는 로그 없음"))
    err = [l for l in tail(newest.replace("stdout", "stderr"), 20) if l.strip()]
    if err:
        print("  stderr 마지막:", err[-1].strip()[:120])

    print("\n[Rapfi 사다리 진행]")
    print("  " + "\n  ".join(run([sys.executable, "rapfi_progress.py"]).strip().splitlines()[-8:]))
    for l in tail("rapfi_ladder_watch.log", 3):
        print("  log:", l.strip()[:120])

    print("\n[문제 채점]")
    # 2026-10-12: newest finished grading + any grading still running (one-off tactical_*.out)
    last = None
    try:
        with open(os.path.join("model", "tactical_problem_log.jsonl"), encoding="utf-8") as f:
            last = json.loads(f.readlines()[-1])
        print(f"  최근 완료: {last['time']}  {last['checkpoint']}  대표 {last['passed']}/{last['total']}"
              f"  졸업 {last.get('grad_passed', '?')}/{last.get('grad_total', '?')}")
        for x in last.get("results", []):
            print(f"    {'통과' if x.get('passed') else '실패'}  {str(x.get('name', x.get('id', '')))[:40]}")
    except Exception as e:
        print("  (채점 기록 없음)", e)
    import glob
    outs = sorted(glob.glob("tactical_*.out"), key=os.path.getmtime)
    if outs and (not last or os.path.getmtime(outs[-1]) > time.mktime(time.strptime(last["time"], "%Y-%m-%d %H:%M:%S"))):
        lines = [l.strip() for l in tail(outs[-1], 20) if l.strip()]
        print(f"  진행 중: {outs[-1]} ({len([l for l in lines if l.startswith(('PASS', 'FAIL'))])}문제 끝)")
        for l in lines[-6:]:
            print("    " + l[:110])

    print("\n[800번대 다시 재기]")
    try:
        d = json.load(open(os.path.join(SP, "remeasure_800s.json")))
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"  {k}: " + ", ".join(f"{n}노드 {x['score']}%" for n, x in v.items()))
            else:
                print(f"  {k}: {v}")
    except Exception:
        print("  (결과 파일 없음/진행 중)")

    print("\n[문제 난이도]")
    try:
        d = json.load(open("model/problem_difficulty.json"))
        print(f"  {len(d)}/26 문제 완료")
    except Exception:
        print("  (없음)")

    print("\n[GPU]")
    print("  " + run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used", "--format=csv,noheader"]).strip())
    print("\n[주요 프로세스]")
    ps = run(["powershell", "-NoProfile", "-Command",
              "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | ForEach-Object { $_.CommandLine }"])
    for key, name in (("_restart_train.py", "학습"), ("watchdog_.py", "워치독"), ("rapfi_ladder_watch.py", "사다리 감시"),
                      ("remeasure_800s", "800번대 재측정"), ("spectate.py", "관전 서버"), ("server.py", "웹 대국 서버")):
        print(f"  {name:<10} {'실행 중' if key in ps else '-'}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "watch":
        while True:
            os.system("cls"); show(); time.sleep(60)
    else:
        show()
