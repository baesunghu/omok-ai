# archive/ — 한 번 쓰고 끝난 스크립트

기록으로만 남긴 일회용·시험용 스크립트다(2026-11 정리). 원래 있던 실행 폴더 이름의 하위 폴더에 두었다.
실행 폴더를 복사할 때 함께 복사된 사본은 지우고 원본 하나만 남겼다.

다시 돌리면 원래 실행 폴더(예: `11x11_naps/`)를 대상으로 동작한다. 다만 대부분 그날의 체크포인트나 파일을 전제로 해서, 지금 다시 돌릴 일은 거의 없다.

| 파일 | 날짜 | 무엇이었나 |
|---|---|---|
| `11x11_naps/value_head_reset.py` | 9/27 | NAPS ep408에서 가치망 머리만 초기화한 1회 작업 (Hard-7 라벨 오염) |
| `11x11_naps/_rapfi_oneoff_840.py` | 10/09 | 래더 감시 프로그램이 멈춘 사이 승격된 ep840을 따로 잰 1회 측정 |
| `11x11_naps/snapthrough_eval_d13.py` | 10/11 | 스냅스루 (3,7) 검증. 결과: [`docs/records/snapthrough_evidence_ep900_d13.md`](../docs/records/snapthrough_evidence_ep900_d13.md) |
| `11x11_naps/value_head_reset_trial.py` | 10/12 | ep960에서 가치망 초기화를 시험한 사본 (학습 중인 모델은 건드리지 않음) |
| `11x11_naps/alphazero_mcts_make_lrtest.py` | 9월 | 학습률 시험용 학습기 변형. 다른 코드가 쓰지 않는다 |
| `15x15/_setup15_patch.py` | 10/14 | 10x10(지금 11x11_naps)에서 복사한 도구를 15x15 파일 이름으로 고친 1회 패치. 지금은 `run_config.py`의 판 크기와 `shared/fast_board.FILE_SUFFIX`가 대신한다 |
| `11x11_v2/switch_to_ep960.py` | 10/21 | v2 런을 Gen12 대신 NAPS ep960에서 다시 시작한 1회 작업 |
