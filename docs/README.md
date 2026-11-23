# 문서 안내

이 폴더가 프로젝트 문서의 전부다. 문서끼리 수치나 판단이 다르면 **현행 문서가 우선**한다.

## 결론

**메기 학습법(CATFISH)은 현재 테스트한 규모에서는 효과 없음.**

- 범위: 11×11 오목, v3 신경망(128채널 × 8블록), 60 에폭 학습, 시드 3쌍.
- 근거: 메기만 다르고 나머지는 같은 사전 등록 통제 실험. 60 에폭 모델끼리 직접 대결 550판에서 메기 쪽 점수 49.3%(95% 구간 약 46~53%). 사전 등록 기준(95% 하한 > 50%)에 미치지 못함.
- 사후 확인(판정에는 쓰지 않음)도 같은 방향: 더 센 메기 42.5%, 본 학습 후반 분기(P) 48.0%, 새 시작 위치 재검증(약 460판, 중단) 49.3%.
- 이유: 이 규모에서는 자가대국이 가치망의 착각을 이미 스스로 고쳤다. 일부러 심은 착각도 1~10 에폭 만에 회복됐다.
- 시험하지 않은 것: 더 큰 판(15×15), 더 큰 신경망, 더 긴 학습, 더 많은 시드, 다른 게임.

v3의 성과(대등점, 문제 통과, 정오)는 측정된 사실이다. 다만 v3는 검벨 탐색, 보조 목표, 작은 신경망, 게이트 제거, 탐험 장치, 메기를 한꺼번에 바꿨고, 그 가운데 메기의 몫은 확인되지 않았다.

## 문서 목록

| 문서 | 내용 | 쓴 날 | 상태 |
|---|---|---|---|
| [NAPS_lineage.md](NAPS_lineage.md) | 모델 계보 전체: 3x3 → NAPS → 여명 → v2 → v3 → 검증 | 10/09 ~ 11월 | **현행** |
| [CATFISH_report.md](CATFISH_report.md) | 메기 리그: 방법, 기원, 통제 실험 전 사례(1~8장), 통제 실험과 결론(9~10장) | 10/23 초안, 11월 결론 갱신 | **현행** |
| [records/value_ablation_ep960.md](records/value_ablation_ep960.md) | 가치망 제거 실험 (메기의 출발점) | 10/12 | 기록 |
| [records/snapthrough_evidence_ep900_d13.md](records/snapthrough_evidence_ep900_d13.md) | NAPS-900의 13수 해법 (3,7) | 10/11 ~ 10/12 | 기록 |
| [code_history.md](code_history.md) | 코드 주석에서 옮긴 날짜별 기록 (파일·함수별) | 8월 ~ 11월 | 기록 |
| [archive/15x15_migration_plan.md](archive/15x15_migration_plan.md) | 15×15 전환 준비 목록 | 10/13 | 보관 (계획) |
| [archive/design_balanced_openings_20260926.md](archive/design_balanced_openings_20260926.md) | 균형 시작 국면 설계안 | 9/26 | 보관 (구현 안 함) |
| [archive/paper_handover.md](archive/paper_handover.md) | 논문 작업 인수인계 메모 | 10/12 | 보관 |
| [archive/paper_pack_20261015/](archive/paper_pack_20261015/README.md) | 논문 초본 재료와 그림 | 10/15 | 보관 |
| [archive/make_paper_v8.py](archive/make_paper_v8.py) | 논문 v8(.docx) 생성 스크립트 | 10/26 | 보관 |

- **현행**: 지금의 결론과 수치를 담는다.
- **기록**: 한 실험의 기록. 결과는 그대로 유효하고, 해석은 현행 문서를 따른다.
- **보관**: 쓴 날 시점의 계획·메모. 이후 바뀐 내용은 고치지 않았다.

## 메기에 관한 서술을 읽는 법

- 10/18 ~ 10/28에 쓴 문장(보고서 1~8장, 계보의 4막 후반과 5막, `festival/`의 축제 영상)은 통제 실험 **전**의 가설과 관찰이다. "메기 덕분에"라는 해석은 이후 통제 실험으로 지지되지 않았다.
- 결론은 보고서 9~10장과 계보 6막이다.

## 용어

모든 문서에서 같은 말을 쓴다. 예전 문서의 다른 표기는 2026-11 정리에서 바꿨다.

| 용어 | 뜻 | 예전 표기 |
|---|---|---|
| 대등점 | 우리 모델과 50 대 50이 되는 Rapfi의 탐색 노드 수. 클수록 강하다 | 대등국면, 대등 노드, 동등 노드 |
| 래더 | 노드 수가 다른 여러 단계의 Rapfi와 두어 대등점을 맞추는 측정 | 사다리 |
| 에폭 | 자가대국 한 묶음을 두고 학습하는 한 주기 | 에포크 |
| 메기 (CATFISH) | 현재 신경망 복사본의 가치망이나 정책망 출력을 일부러 끄거나 줄인 연습 상대 | 메기 리그, 메기 학습법 |
| 직접 대결 | 두 체크포인트를 같은 시작 위치에서 색을 바꿔 두는 비교 (`11x11_v2/v3/head2head.py`) | 관전, h2h |
| 사전 등록 | 결과를 보기 전에 예측과 판정 기준을 파일로 적고 SHA-256 지문을 남기는 것 | |
| 졸업 | 1·2단계 문제 10개 전부 통과 (10/23 정의. 9/29 여명 때는 졸업 문제 3개) | |
| 정오 (NOON) | 졸업 후 3단계 18문제 중 80%(15문제) 이상. 10/24부터 문제 기준만 쓴다. v3 ep530이 처음 도달 | |
| 실행 폴더 | `11x11_naps/`(11×11 NAPS, 예전 이름 `10x10/`), `11x11_v2/`(v2·v3), `15x15/`. 설정(`run_config.py`)과 그 실행만의 스크립트가 있고, 코드는 `shared/` | |

## 코드에서 경로와 설정을 찾는 곳

| 무엇 | 어디 |
|---|---|
| 저장소 루트, Rapfi·scratch 위치 (환경 변수로 바꿀 수 있음) | `omok_paths.py` |
| 실행마다 다른 값 (판 크기, 게이트 패배 가중치, 학습률, 배치, 엔진, 감시 설정) | 각 실행 폴더의 `run_config.py` |
| 파일 이름 접미사 (`_11x11`) | `shared/fast_board.py`의 `FILE_SUFFIX` (판 크기에서 만듦) |
| 코드 (신경망, 학습 본체, MCTS, 측정·감시) | `shared/` 한 벌 (도구는 실행 폴더 안에서 `python ../shared/<도구>.py`) |
| 졸업·정오 문제 세트 (28문제, 11x11) | `11x11_naps/tactical_problems/web_game58_problems.json` (모든 11x11 실행이 이 한 벌을 씀) |
| 한 번 쓰고 끝난 스크립트 | `archive/<실행 폴더>/` |
| 메기 통제 실험의 팔 설정 | `11x11_v2/v3/arms.py` (실행: `python 11x11_v2/v3/train_v3.py <팔 이름>`) |

## 이름이 바뀐 것 (2026-11 정리)

기록 문서(계보, 실험 기록, 보관 문서, `manual/`, `festival/`)는 그때의 이름을 그대로 쓴다. 지금 저장소의 이름은 다음과 같다.

| 예전 이름 | 지금 이름 |
|---|---|
| `10x10/` (11×11 NAPS 실행 폴더) | `11x11_naps/` |
| `alphago_lee.py` | `shared/board_encoding.py` (판 → 입력 평면) + `shared/nets.py` (V1 신경망) |
| `alphago_lee_v2_seblock.py`, `alphago_lee_v3_survival.py` | `shared/nets.py` |
| `AlphaZeroTicTacToe`, `AlphaZeroTicTacToeV2`, `AlphaZeroTicTacToeV3` (신경망 클래스) | `GomokuNetV1`, `GomokuNetV2`, `GomokuNetV3` |
| `alphazero_mcts_make_lrfix.py` (NAPS 학습 본체, 4,080행 한 파일) | `shared/naps_trainer.py` (학습 루프) + `shared/naps_*.py` 8개 (게이트, 리그, 어려운 국면 등) |
| `alphazero_mcts_lrfix.py` (C++ 엔진을 쓰는 MCTS) | `shared/mcts_cpp.py` |
| `alphazero_mcts.py` (파이썬 MCTS) | `archive/shared/mcts_python.py` (더 쓰지 않음) |
| `alphago_lee_make.py`, `alphazero_mcts_make.py` (옛 학습기) | `archive/shared/legacy_trainer_basic.py`, `archive/shared/legacy_trainer_mcts.py` (더 쓰지 않음) |
| `determinisic_play.py` | `shared/deterministic_play.py` |
| `_restart_train.py`의 본문 | `shared/train_run.py` (실행 폴더의 `_restart_train.py`는 이것을 부른다) |
| 실행 폴더마다 한 벌씩 있던 코드 | `shared/` 한 벌 + 실행 폴더의 `run_config.py` |

코드 주석에 붙어 있던 날짜와 `(user)` 표시는 지우고 설명만 남겼다. 원래 주석은 [code_history.md](code_history.md)에 파일·함수별로 옮겨 두었다.

체크포인트는 가중치(`state_dict`)로 저장되고 버퍼는 numpy 배열만 담으므로, 이름이 바뀌어도 예전 파일을 그대로 읽는다.

## 확인하지 못한 것

- ep960 래더 승격 확률: `archive/paper_handover.md` 안에서 87%와 86%로 다르다. 원 기록(학습 로그)이 저장소에 없다.
- Kaggle 계획(`11x11_v2/kaggle/kaggle_plan.py`)의 시드 4·5, E 팔(데이터 비용 통제), G 팔(적응형 메기): 결과 기록이 저장소에 없다.
- `CATFISH_report.md` 참고문헌 11~13번의 서지 표기.
