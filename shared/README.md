# shared/ — 모든 실행에 공통인 코드

세 실행 폴더(`10x10/`, `11x11_v2/`, `15x15/`)에 **내용이 완전히 같게** 복사되어 있던 파일을 한 벌만 둔 곳이다(2026-11 정리).
실행 폴더에는 그 실행에만 있는 파일, 또는 실행마다 내용이 다른 파일만 남는다. 판 크기(`fast_board.py`), 학습 본체(`alphazero_mcts_make_lrfix.py`), 학습 실행(`_restart_train.py`), 감시·채점 스크립트가 그렇다.

## 쓰는 법

**import해서 쓰는 모듈**은 실행 폴더의 스크립트가 불러오면 그대로 쓰인다. 판 크기처럼 실행마다 다른 값은 그 실행 폴더의 `fast_board.py`를 따른다.
실행 폴더의 스크립트는 맨 앞쪽에서 이 폴더를 `sys.path`에 넣는다.

**직접 실행하는 도구**는 다룰 실행 폴더 안에서 실행한다.

```bash
cd 11x11_v2
python ../shared/ladder_ci.py
```

실행 폴더 밖에서 돌리려면 `OMOK_RUN`으로 폴더를 지정한다.

```bash
OMOK_RUN=11x11_v2 python shared/ladder_ci.py
```

## 목록

| 파일 | 종류 | 내용 |
|---|---|---|
| `alphago_lee.py` | 모듈 | 판을 신경망 입력으로 바꾸기, V1 신경망 |
| `alphago_lee_v2_seblock.py` | 모듈 | V2 신경망 (SE 블록 + GroupNorm) |
| `alphago_lee_v3_survival.py` | 모듈 | V3 신경망 (V2 + 생존 헤드). NAPS와 v3 모두 이 구조를 바탕으로 한다 |
| `alphazero_mcts.py` | 모듈 | 파이썬 MCTS와 후보 수 가지치기 |
| `selfplay_mp.py` | 모듈 | 멀티프로세스 자가대국 (멈춤 스위치 파일은 실행 폴더의 `MP_SELFPLAY_OFF`) |
| `mp_eval.py` | 모듈 | 멀티프로세스 공식 평가 |
| `determinisic_play.py` | 모듈 | 결정론적 미니맥스 상대 (C++ `deterministic_play_core`로 위임) |
| `attack_ladder.py` | 모듈 | 공격 쪽 단계 검사 (`fork_status_check.py`의 짝) |
| `alphago_lee_make.py` | 모듈 (옛 학습기) | 작은 판 시절 학습기. 데이터 증강과 로거를 다른 모듈이 가져다 쓴다 |
| `alphazero_mcts_make.py` | 모듈 (옛 학습기) | lrfix 이전 학습기. `mp_eval.py` 등이 가져다 쓴다 |
| `rapfi_ladder.py` | 도구 | Rapfi 래더 측정 (각 실행의 `rapfi_ladder_watch.py`, v3의 `ladder_watch_v3.py`가 실행) |
| `rapfi_progress.py` | 도구 | 진행 중인 래더 측정 보기 |
| `ladder_ci.py` | 도구 | 래더 대등점의 부트스트랩 신뢰구간 |
| `problem_difficulty.py` | 도구 | 문제별 Rapfi 난이도 |
| `double_three_test.py` | 도구 | 삼삼 측정 문제 (측정 전용) |
| `elo_defense.py` | 도구 | 선공 이점을 뺀 실력 비교 |
| `elo_roundrobin.py` | 도구 | 체크포인트 라운드로빈 Elo |
| `elo_trend.py` | 도구 | best 계보의 Elo 추이 (게이트 기록에서 계산) |
| `eval_league_greedy.py` | 도구 | 과거 체크포인트와 공정 평가 (잡음 없음, 탐욕적) |
| `draw_trend.py` | 도구 | 에폭별 승부·무승부 비율 |
| `loss_trend_annotated.py` | 도구 | 손실 추이 그래프 (파이프라인 변경 시점 표시) |
| `matchup_chart.py` | 도구 | 직접 대결 요약 그래프 (NAPS 기록용, `10x10`에서 실행) |
| `status_all.py` | 도구 | 총괄 현황 한 화면 |
| `hourly_status.py` | 도구 | 한 시간마다 상태 기록 |
| `_vs_human_turn.py` | 도구 | 채팅으로 사람 대 AI 대국을 한 수씩 진행 |

실행마다 내용이 달라 실행 폴더에 남은 파일 가운데 이름이 같은 것(`fast_board.py`, `rapfi_match.py`, `tactical_problem_check.py`, `ladder_promo.py` 등)은 각 실행의 설정이나 수정이 들어 있는 별개의 파일이다.
