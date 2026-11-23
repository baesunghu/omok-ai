# shared/ — 모든 실행의 코드 (한 벌)

세 실행 폴더(`11x11_naps/`, `11x11_v2/`, `15x15/`)에 복사되어 있던 코드를 한 벌로 합친 곳이다(2026-11 정리).

- **실행마다 다른 값은 각 실행 폴더의 `run_config.py`에 있다.** 판 크기, 게이트 패배 가중치, 학습률 범위, 배치 크기, 엔진 종류(VCF), 감시 설정이 여기에 모여 있다.
- 코드는 `run_config`에서 값을 읽으므로, 같은 파일이 세 실행 모두에서 그 실행의 설정대로 동작한다.
- 실행 폴더에는 `run_config.py`, 학습 실행 파일(`_restart_train.py`, 15x15는 `_restart_train15.py`), 그리고 그 실행에서만 쓴 스크립트(그래프, 감시 화면, 15x15 지도학습 등)만 남는다.

## 쓰는 법

학습은 실행 폴더에서 시작한다(감시 프로그램도 같은 명령을 쓴다).

```bash
cd 11x11_naps
python _restart_train.py
```

도구는 다룰 실행 폴더 안에서 실행한다.

```bash
cd 11x11_v2
python ../shared/ladder_ci.py
```

실행 폴더 밖에서 돌리려면 `OMOK_RUN`으로 폴더를 지정한다.

```bash
OMOK_RUN=11x11_v2 python shared/ladder_ci.py
```

## 학습

| 파일 | 내용 |
|---|---|
| `train_run.py` | 학습 시작: 중복 실행 잠금(`train.lock`)과 공통 하이퍼파라미터. 실행 폴더의 `_restart_train.py`가 부른다 |
| `naps_trainer.py` | NAPS 학습 루프 `train_mcts_alpha_zero`: 대국 생성 → 버퍼 → 학습 → 게이트 → 측정. 아래 `naps_*` 모듈을 부른다 |
| `naps_common.py` | 학습기 전체가 함께 쓰는 설정과 상태: `model/` 파일 경로, 보고 로거, 시뮬레이션 수·LR 설정, 가치 목표, 리플레이 버퍼, 에포크 수 |
| `naps_openings.py` | 대국 시작 국면: 대칭으로 줄인 첫 수, 전술·강제 시작 국면 |
| `naps_selfplay.py` | 자가대국 한 판(`_play_one_mcts_game`)과 여러 판 |
| `naps_gate.py` | 게이트: 후보가 현재 best와 둔다. 점수 = (승 + 0.5 × 무 − 벌점 × 패) / 판 수 |
| `naps_vsdet.py` | 결정론적 미니맥스 상대와의 대국: 평가와 학습 데이터 |
| `naps_league.py` | 리그: 지난 체크포인트와 best 모델을 상대로 두는 대국 |
| `naps_tactics.py` | 합성 전술 국면: 닫힌 3, 포크, 쌍사, 열린 3, 진단 국면 |
| `naps_hard_examples.py` | 어려운 국면: 끝난 대국에서 신경망이 틀리거나 망설인 국면을 깊은 탐색으로 다시 라벨해 풀에 모으고 학습에 섞는다 |
| `naps_threats.py` | 판 위협 검사: 4목의 빈칸, 포크 자리, 바로 이기는 자리, 상대가 다음 수에 포크를 만들 수 있는지 |
| `naps_tactical_refresh.py` | 전술 보강 학습: 합성·채굴 전술 국면으로 몇 에폭 지도학습, 에폭마다 열린 3 국면 주입 |
| `mcts_cpp.py` | MCTS: 신경망 평가를 C++ 엔진에 연결 (v2 엔진이면 VCF·MEGA 인자도 넘김) |
| `selfplay_pool.py` | 대국 풀 (주 경로): 한 에폭의 자가대국·미니맥스·리그 대국과 보강·게이트 작업을 프로세스 10개가 한 대기열에서 나눠 둔다 (v2는 C++ 스레드 자가대국 `11x11_v2/selfplay_cpp.py`) |
| `selfplay_pool_tasks.py` | 대국 풀의 작업자가 작업 하나를 처리하는 법, 끝난 대국을 학습 표본으로 바꾸는 법 (출처별 가중치) |
| `selfplay_pool_evaluator.py` | 대국 풀의 GPU 평가: 작업자들의 평가 요청을 모델별로 모아 한 번에 계산 |
| `selfplay_pool_fallback.py` | 예비 대국 풀: 주 풀이 꺼졌거나(`MP_UNIFIED_OFF`) 실패했을 때 자가대국만 여러 프로세스로 둔다. 이것도 실패하면 스레드로 둔다 |
| `mp_eval.py` | 미니맥스 상대 평가를 여러 프로세스로 (실패하면 스레드로) |
| `fast_board.py` | 판 크기(`run_config`에서), 승리 판정, 파일 이름 접미사 `FILE_SUFFIX` (`_11x11`) |
| `board_encoding.py` | 판을 신경망 입력 평면 5장으로 바꾸기 (내 돌, 상대 돌, 빈칸, 내·상대 포크 칸), 학습 표본의 8가지 대칭 (`augment_sample`) |
| `nets.py` | 신경망 `GomokuNetV1`, `GomokuNetV2` (SE 블록 + GroupNorm), `GomokuNetV3` (V2 + 생존 헤드, NAPS가 쓰는 구조). v3 실행의 `NetV3`도 이것을 바탕으로 한다 |
| `deterministic_play.py` | 결정론적 미니맥스 상대 (C++ `deterministic_play_core`로 위임) |

## 측정과 감시

| 파일 | 내용 |
|---|---|
| `rapfi_match.py` | 우리 모델 대 Rapfi 대국 (Gomocup 프로토콜) |
| `rapfi_match_pooled.py` | GPU 평가를 모아서 하는 Rapfi 대국 (실험용) |
| `rapfi_ladder.py` | Rapfi 래더 측정과 대등점 계산 |
| `rapfi_ladder_watch.py` | 학습 중 정해진 에폭마다 래더·채점·래더 승격을 자동 실행 (단계는 `run_config.LADDER_LEVELS`) |
| `ladder_promo.py` | 래더 승격 판정 |
| `tactical_problem_check.py` | 졸업·정오 문제 채점 (문제 세트는 `11x11_naps/tactical_problems/`, 11x11 전용) |
| `problem_difficulty.py` | 문제별 Rapfi 난이도 |
| `watchdog_.py` | 학습 프로세스 감시와 자동 재시작 (감시 대상은 `run_config.TRAIN_SCRIPT`) |
| `promotion_sms.py` | 승격 문자 알림 |
| `ladder_ci.py` | 래더 대등점의 부트스트랩 신뢰구간 |
| `rapfi_progress.py`, `status_all.py`, `hourly_status.py` | 진행 상황 보기, 총괄 현황, 한 시간마다 상태 기록 |
| `fork_status_check.py`, `fork_status_check_solver.py`, `attack_ladder.py` | 포크(쌍삼·쌍사) 공격·방어 단계 검사 |
| `double_three_test.py` | 삼삼 측정 문제 |
| `elo_defense.py`, `elo_roundrobin.py`, `elo_trend.py` | 체크포인트끼리 Elo 비교 |
| `draw_trend.py`, `loss_trend_annotated.py`, `matchup_chart.py` | 그래프 (`matchup_chart.py`는 NAPS 기록용, `11x11_naps`에서 실행) |
| `human_rapfi_rating.py` | 웹 대국판 방문자의 수를 Rapfi 노드 단위로 평가 (`11x11_naps/hlt/web/games.jsonl`) |

옛 학습기(`legacy_trainer_basic.py`, `legacy_trainer_mcts.py`), 파이썬 MCTS(`mcts_python.py`), 그리고 이것들에 기대던 도구
`_vs_human_turn.py`, `eval_league_greedy.py`는 `archive/shared/`로 옮겼다. 두 도구는 V1 신경망만 읽어서 Gen12 이후의 체크포인트(V3)를
읽지 못한다. 사람 대 AI 대국은 웹 대국판(`11x11_naps/hlt/web/`)이 맡는다.
