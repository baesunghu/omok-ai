# 코드 속 날짜별 기록

2026-11 정리에서 코드 주석의 날짜와 `(user)` 표시를 지우고, 원래 주석을 여기에 그대로 옮겼다.
코드에는 설명만 남아 있다. 줄 번호는 정리 직전 기준이라 지금 파일과 조금 어긋날 수 있다. 함수 이름으로 찾는 것이 정확하다.
옛 학습기(`legacy_trainer_*.py`, 지금은 `archive/shared/`)와 `archive/`의 주석은 원래대로 두었다.

## `11x11_naps/gomokuwithai_match.py`

**(module level)**

- 41행

  ```text
  # 2026-10-10: GWA_CKPT selects another checkpoint (default Dawn); its games go to a log named after it.
  ```

- 48행

  ```text
  # 2026-09-30: time-matched mode (SIMS = 0): Dawn searches until it has used as much
  # wall time as the site AI's average move time so far in this match (min 1100 sims).
  ```

- 54행

  ```text
  # 2026-10-26 (user): GWA_ATTACK=1 -> attack-leaning final pick (hlt/web/attack_pick.py) when our AI is white;
  # GWA_SIDE=X/O pins our colour (default: alternate). Both change the log name.
  ```

- 63행

  ```text
  # 2026-10-26 (user wants to watch): current board after every ply, read by hlt/gwa_live/index.html
  ```

- 79행

  ```text
  # 2026-10-26 (user): v3 NetV3 checkpoints (e.g. NOON ep530) load too; they play with the 11x11_v2 VCF engine, as on the web board
  ```

**dawn_move**

- 177행

  ```text
  # 2026-10-26 (user): analysis for the live viewer (admin /gwa): top candidates, the most-visited move vs the
  # move played, root win rate. q_value is from the opponent's side, so the mover's win rate is (1 - q) / 2.
  ```

## `11x11_naps/hlt/web/live_dashboard.py`

**(module level)**

- 20행

  ```text
  # 2026-10-16: spectator view proxied through this (password-protected) server
      (on: WATCH_BASE = "/watch#watch=")
  ```

- 25행

  ```text
  # 2026-10-16 (user): sessions hidden from the dashboard
      (on: IGN = os.path.join(HERE, "dashboard_ignored.json"))
  ```

**do_GET**

- 226행

  ```text
  # 2026-10-16 (user): admin-only
      (on: if self.headers.get("Authorization") != AUTH:)
  ```

- 238행

  ```text
  # 2026-10-26 (user): watch the ep530 vs GWA4 match here too
      (on: elif self.path.split("?")[0] in ("/gwa", "/gwa/"):)
  ```

## `11x11_naps/hlt/web/server.py`

**(module level)**

- 26행

  ```text
  # 2026-10-26 (user): festival board uses the 11x11_v2 engine with VCF on, same as the NOON grading and the Rapfi ladder
  # (was mcts_core_solver: vlfix + mate distance + solver, no VCF; same search_batched API)
  ```

- 89행

  ```text
  # 2026-09-30 (user): visitors choose the model. "dawn" = graduated NAPS-625 (frozen),
  # "latest" = the newest gate-promoted best checkpoint (reloaded when a game starts).
  ```

- 97행

  ```text
  # 2026-10-11 (user): festival candidates frozen in model/festival_candidates/ can be chosen too
  ```

- 104행

  ```text
  # 2026-10-26 (user): NOON model v3 ep530 (정오 충족 10/25 14:14:29), the festival pick
  ```

- 266행

  ```text
  # 2026-10-13 (user): in-progress games are written to disk after every move, so a server restart no longer
  # loses them (a won game vs the best was lost to a restart). Restored on start; an AI turn resumes thinking.
  ```

- 519행

  ```text
  # 2026-10-13 (user): "--lan" also serves other devices on the same network (festival QR code access)
  ```

**_load_net**

- 53행

  ```text
  # 2026-10-26: v3 NetV3 checkpoint (same loader as spectate.py)
      (on: if any(k.startswith("opp_policy_head.") for k in sd):)
  ```

**last_promoted_epoch**

- 68행

  ```text
  # 2026-10-11: the training report holds every gate line across restarts (stdout logs are per run)
  ```

## `11x11_naps/hlt/web/spectate.py`

**(module level)**

- 30행

  ```text
  # 2026-10-26 (user): VCF engine from 11x11_v2, same as the NOON grading (was mcts_core_solver, no VCF)
  ```

- 44행

  ```text
  # 2026-10-23 (user): "기록 초기화" hides older games from the table; the log itself is kept
  ```

- 50행

  ```text
  # 2026-10-11 (user: "same positions over and over"): 0.3 -> 0.15, wider pool
      (on: OPENING_MIN_SHARE = 0.15)
  ```

- 51행

  ```text
  # 2026-10-11: black's first stone is a random cell within 2 of the centre (25 cells)
      (on: FIRST_STONE_RADIUS = 2)
  ```

**list_players**

- 65행

  ```text
  # 2026-10-11 (user): frozen festival candidates (e.g. ep905, a best with no 10-epoch snapshot),
  # listed right after dawn/best so they are easy to find; newest first
  ```

- 72행

  ```text
  # 2026-10-23 (user): v2 strongest and v3 (Gumbel, NetV3 128x8) checkpoints for head-to-head viewing
  ```

**get_net**

- 100행

  ```text
  # 2026-10-23: v3 NetV3 checkpoint
      (on: if any(k.startswith("opp_policy_head.") for k in sd):)
  ```

## `11x11_v2/gomokuwithai_match.py`

**(module level)**

- 39행

  ```text
  # 2026-10-10: GWA_CKPT selects another checkpoint (default Dawn); its games go to a log named after it.
  ```

- 46행

  ```text
  # 2026-09-30: time-matched mode (SIMS = 0): Dawn searches until it has used as much
  # wall time as the site AI's average move time so far in this match (min 1100 sims).
  ```

## `11x11_v2/make_catfish.py`

**(module level)**

- 27행

  ```text
  # 2026-10-21 (user): a "half" value catfish -- last value Linear x0.5, so tanh(0.5 z): the same judgement,
  # just less sure of it (a softened, not a dead, value head).
  ```

- 36행

  ```text
  # 2026-10-22 (user): Dawn (ep625, graduated 9/29) as an intact league anchor -- the cleanest value head in the
  # lineage (before the Hard-7 contamination) and it passed the graduation problems 3/3.
  ```

## `11x11_v2/run_config.py`

**(module level)**

- 13행

  ```text
  # 2026-10-21 15:15 (user): MEGA off (Rapfi 3000: VCF 60% vs VCF+MEGA 31%); was (30, 15)
      (on: MEGA_VISITS, MEGA_MIN_SIMS = (0, 0), 1000)
  ```

- 20행

  ```text
  # 2026-10-21 (user: raise GPU use): leaves per MCTS evaluation request (was 16)
      (on: EVAL_BATCH = 32)
  ```

## `11x11_v2/v3/head2head.py`

**(module level)**

- 27행

  ```text
  # 2026-10-26 (user: evaluate ep400 more): 36 openings = 72 games, the 10 above first
      (on: if os.environ.get("H2H_MORE"):)
  ```

- 30행

  ```text
  # 2026-10-31 (user: catfish A/B at ep30/60): 100 distinct openings = 200 games, the 10 above first
      (on: if os.environ.get("H2H_ALL"):)
  ```

- 62행

  ```text
  # 2026-10-31 (user: 관전): sharded runs write the game in progress for ab_scratch/ab_live.py
      (on: LIVE = None)
  ```

**load**

- 44행

  ```text
  # 2026-10-30 (user): 15x15 V4 net played on the 11x11 board
      (on: if "policy_head.out.weight" in sd:)
  ```

## `11x11_v2/v3/ladder_watch_v3.py`

**(module level)**

- 73행

  ```text
  # 2026-10-22 (user): graduation / tier problems right after each ladder
  ```

- 75행

  ```text
  # 2026-10-22 23:55 (user): stream to the log as it runs -- tier 1 prints first, so it can be read before the rest
  ```

## `11x11_v2/v3/monitor_v3.py`

**procs**

- 53행

  ```text
  # 2026-10-23: cp949 decode crash on Korean paths
      (on: encoding="utf-8", errors="replace").stdout or "")
  ```

## `11x11_v2/v3/rapfi_selfcheck.py`

**(module level)**

- 31행

  ```text
  # 2026-10-24: force the defender's first move (snap-through check)
      (on: if first is None and FIRST:)
  ```

## `11x11_v2/v3/train_v3.py`

**(module level)**

- 60행

  ```text
  # 2026-10-22 21:00 (user): CATFISH league -- crippled copies of the CURRENT net, rebuilt every epoch:
  # value0 (last value Linear zeroed), value_half (x0.5), policy0 (uniform prior). The catfish side
  # searches with plain PUCT and its moves are not recorded; only the main net's full-search moves are.
  ```

- 63행

  ```text
  # 2026-10-23 02:10 (user): policy_half added (policy logits x0.5 = flatter prior) to poke policy blind spots;
  # tier-3 failures at ep120 were policy blind spots, not value overconfidence.
  # Same time (user): policy0 dropped -- the main net beat it 95-97% (ep79-128), too weak to teach anything.
  # 3 x 16 = 48 league games, 144 self-play (same split as before).
  # {kind: scale}: "value*" scales the last value Linear, "policy*" the last policy Linear, 0 = switched off.
  ```

- 76행

  ```text
  # 2026-10-23 03:xx (user): 16 -> 8 per catfish (league 25% -> 12.5%); ep120 lost to ep100 40% (4/8/8), suspected catfish over-fit
      (on: LEAGUE_GAMES = 8)
  ```

- 81행

  ```text
  # 2026-10-23 (user): 16/8 -> 32/16, low-prior moves reach the candidate set more often
      (on: GUMBEL_M_FULL, GUMBEL_M_FAST = 32, 16)
  ```

- 82행

  ```text
  # 2026-10-23 (user): exploration for self-play ("창발적 수"): random opening stones, early-move temperature
  # (sampled from the Gumbel improved policy), Dirichlet noise on the Gumbel candidate sampling.
  ```

- 94행

  ```text
  # 2026-10-23 07:55 (user): 1e-3 -> 3e-4; policy loss minus target entropy flat at ~0.71 for 20 epochs
      (on: LR, WD = 3e-4, 1e-4)
  ```

## `15x15/ladder_chart_ci.py`

**(module level)**

- 12행

  ```text
  # 2026-10-14 (user): Elo is the main metric, parity nodes secondary. Colab calibration, Rapfi 100 nodes = 0.
  ```

## `15x15/run_config.py`

**(module level)**

- 16행

  ```text
  # 2026-10-16 (user, ep93): 0.75 -> 0.4 (no draws on 15x15; 11-9 passes, 10-10 does not)
      (on: GATE_LOSS_PENALTY = 0.4)
  ```

- 17행

  ```text
  # 2026-10-13: canonical openings include far corners on 15x15; keep the gate near the centre
      (on: GATE_OPENING_RADIUS = 3)
  ```

- 18행

  ```text
  # 2026-10-15 (user): the 2e-5 polish band of the mature 11x11 net kept the young 15x15 net flat
      (on: LR_PEAK, LR_MIN = 1e-4, 2e-5)
  ```

- 24행

  ```text
  # 2026-10-14: a name distinct from the 11x11 trainer on the same laptop
      (on: TRAIN_SCRIPT = "_restart_train15.py")
  ```

- 25행

  ```text
  # 2026-10-14 (user): 20 -> 40 per level for a tighter Elo estimate
      (on: LADDER_GAMES = 40)
  ```

- 26행

  ```text
  # 2026-10-16: run 3 starts from a Rapfi-10 epoch 0
      (on: LADDER_LEVELS = "30,100,300,1000,3000")
  ```

## `15x15/sl_rapfi_data.py`

**(module level)**

- 20행

  ```text
  # 2026-10-16 (user): 10 for a deliberately weak epoch 0
      (on: NODES = int(os.environ.get("SL_NODES", "100")))
  ```

## `shared/board_encoding.py`

**(module level)**

- 27행

  ```text
  # 2026-09-16: optional native C++/pybind11 acceleration for _fork_mask
  # (cpp_mcts/fork_mask_cpp.cpp, same build pattern as mcts_core/
  # deterministic_play_core -- see cpp_mcts/build_fork_mask_cpp.bat). Explored
  # via a workflow that tried 4 independent optimization strategies (numba
  # JIT, fully-vectorized numpy, this C++ port, a bitboard/SWAR approach) and
  # benchmarked them against each other; this one and numba tied for fastest
  # (~9-11x over the pure-numpy version below), C++ was picked to match this
  # project's existing convention of a compiled engine + pure-Python fallback
  # (mcts_core, deterministic_play_core) rather than adding a new numba
  # runtime dependency to this hot path. Verified byte-for-byte identical to
  # the pure-Python version below on 374 random/edge-case boards (including a
  # duplicate-threat-pair dedup stress case) both by the workflow agent and
  # independently re-verified in this session. Falls back to the pure-Python
  # implementation if the compiled extension isn't present (e.g. a different
  # machine/Python ABI) -- same try/except ImportError pattern as
  # mcts_python.py's _mcts_core.
  ```

## `shared/fork_status_check_solver.py`

**(module level)**

- 51행

  ```text
  # 2026-09-27 solver-engine re-measure copy
      (on: import mcts_core_solver as mcts_core_verbose)
  ```

## `shared/human_rapfi_rating.py`

**(module level)**

- 22행

  ```text
  # 2026-10-10 (user): low end. Rapfi plays identically at 1/3/5 nodes (no search: only forced blocks, else (1,1)), so 5 stands for all three; 15/20 added for resolution; 3000/10000 dropped
      (on: LEVELS = [5, 10, 15, 20, 30, 100, 300, 1000])
  ```

**main**

- 92행

  ```text
  # 2026-10-23 (user): DATES=2026-10-17,2026-10-18 rates only those days (festival); TAG names the output json
  ```

## `shared/ladder_promo.py`

**(module level)**

- 25행

  ```text
  # 2026-10-12 (user): 80% -> 90% (fewer lucky promotions / inflated reference)
      (on: P_MIN = 0.90)
  ```

- 26행

  ```text
  # 2026-10-13 (user): gate candidates also need P(stronger than best) >= 60% (an equal model
      (on: P_CAND = 0.60)
  ```

**check**

- 89행

  ```text
  # 2026-10-17: grad may be None (15x15 has no problem set yet)
      (on: def check(ckpt_name, weights_path, dry_run=False, candidate=False):)
  ```

- 103행

  ```text
  # 2026-10-14 (15x15): no graduation problems yet
      (on: if g is None and ref.get("grad") is not None:)
  ```

## `shared/loss_trend_annotated.py`

**(module level)**

- 36행

  ```text
  # 2026-10-08 (user): show the whole NAPS run, not just ep300+
      (on: FROM = 1)
  ```

- 74행

  ```text
  # 2026-10-08 (user): reference line at Dawn's (graduated ep625) loss
  ```

- 85행

  ```text
  # 2026-10-11 (user): gate rule changed at ep911 -> draw a divider and use each era's score/threshold
  ```

## `shared/mcts_cpp.py`

**(module level)**

- 10행

  ```text
  # 2026-09-22 Gen12: V2 architecture (GroupNorm+SE), see nets.py
  ```

**__init__**

- 139행

  ```text
  # 2026-09-11 instrumentation: tracks real achieved batch sizes so a
  # target_batch change (e.g. 48->128, see forward_pass_benchmark.py's
  # finding that bs=128 gives ~17% more throughput in a synthetic
  # single-shot benchmark) can be verified against ACTUAL concurrency
  # in live self-play traffic instead of assumed. Printed periodically
  # from _flush, not exposed as a public API -- purely observational.
  ```

**_search_cpp**

- 416행

  ```text
  # 2026-09-28: KataGo forced playouts + policy target pruning on
  # noisy (training-data) searches only; evaluation, gate and
  # relabel searches run without noise and stay plain PUCT.
  ```

## `shared/mcts_python.py`

**__init__**

- 89행

  ```text
  # 2026-09-11 instrumentation: tracks real achieved batch sizes so a
  # target_batch change (e.g. 48->128, see forward_pass_benchmark.py's
  # finding that bs=128 gives ~17% more throughput in a synthetic
  # single-shot benchmark) can be verified against ACTUAL concurrency
  # in live self-play traffic instead of assumed. Printed periodically
  # from _flush, not exposed as a public API -- purely observational.
  ```

## `shared/naps_common.py`

**(module level)**

- 15행

  ```text
  # 2026-09-22 Gen12: swapped to the V2 architecture (GroupNorm+SE, see
  # nets.py) -- aliased as GomokuNet so every
  # existing instantiation site below (model/best_model/league_model/
  # gen_model) picks it up without individual edits. NOT weight-compatible
  # with the old nets.GomokuNetV1 checkpoints -- this run
  # starts from the freshly-trained gen12_full_FINAL.pth, not an old one.
  ```

- 21행

  ```text
  # 2026-09-25: V3 = V2 + auxiliary survival (remaining-plies) head, see
  # nets.py. Loads V2 checkpoints; forward() unchanged.
  ```

- 72행

  ```text
  # 2026-10-08 (user): 24,000 -> 196,608 (~40 epochs of data instead of ~5). Training
  # no longer sweeps the whole buffer: each epoch draws TRAIN_SAMPLES_PER_EPOCH at random.
  ```

- 204행

  ```text
  # 2026-10-13 (user): 1100 -> 1600 (deeper full searches = better policy targets)
      (on: SIM_RAMP_CEILING = 1600)
  ```

- 213행

  ```text
  # 2026-09-15: a blind calendar-based LR warm-restart (CosineAnnealingWarmRestarts
  # below) can land in the middle of a fresh promotion's destabilized window --
  # measured on this exact run: the epoch-273 promotion (first ever, after 252
  # straight 0%-loss gate checks) was followed by vs-Det formal win rate
  # collapsing from an oscillating 15-45% to a 0-10% floor, and the epoch-300
  # warm restart (LR 5e-5 -> 0.001) landed right inside that still-unrecovered
  # window, 27 epochs later, making it worse rather than helping the model
  # re-explore. LR_COOLDOWN_EPOCHS holds the scheduler frozen at its current
  # (already-decayed) LR for this many epochs right after any promotion, so a
  # calendar restart can't collide with a promotion-induced shift again.
  ```

## `shared/naps_gate.py`

**(module level)**

- 33행

  ```text
  # 2026-10-13 (user): 0.25 -> 0.35 (6-8-6, an even result, scored 0.275 and promoted ep1005). 2026-10-12: 0.50 -> 0.25. 2026-10-11: was 0.55 (AlphaGo Zero, draws 0.5)
      (on: GATE_PROMOTION_THRESHOLD = 0.35)
  ```

- 34행

  ```text
  # 2026-10-11 (user): "a model that does not lose" -- a loss now costs points:
  # score = (wins + 0.5*draws - GATE_LOSS_PENALTY*losses) / n, promoted only if score > 0.50 (strictly).
  # With 0.75 that is exactly "wins > 2.5 x losses"; draws alone can never promote (all draws = 0.50).
  # Per-run settings (run_config.py): gate loss penalty (11x11 0.75, 15x15 0.4), gate opening radius, LR band,
  # MCTS eval batch and pooled-evaluator batch.
  ```

- 39행

  ```text
  # 2026-10-13 (user): two-stage promotion. The gate (vs the current best only) can pick a model that is
  # weaker by the Rapfi ladder (ep1050 beat ep960 12-5-3 but measured 2,404 vs 4,432). A gate pass now only
  # makes a CANDIDATE: its weights go to model/gate_candidates/ and model/gate_candidate.json; the ladder
  # watcher measures it and promotes it (model/ladder_promote.json) only if it is not weaker than the best.
  ```

- 44행

  ```text
  # 2026-10-13 (user): see _play_one_mcts_vs_league_game
      (on: RECORD_LEAGUE_SIDE = False)
  ```

- 114행

  ```text
  # 2026-10-13 (15x15): canonical openings include far corner squares; there the gate keeps squares within
  # Chebyshev distance GATE_OPENING_RADIUS of the centre. The 11x11 runs used every canonical square (radius None).
  ```

## `shared/naps_hard_examples.py`

**_opponent_can_fork_next**

- 450행

  ```text
  # 2026-10-14 (15x15, now every run): same answer, window-based. The brute force was O(N^6) fast_check_win
  # calls per _fork_safe_cells and stalled the relabel step for 30+ minutes on 15x15.
  ```

**_immediate_win_cells**

- 487행

  ```text
  # 2026-10-14 (15x15, now every run): window scan instead of one fast_check_win per empty cell. A cell completes
  # 5+ in a row iff some 5-cell window through it holds 4 `color` stones and that one empty cell.
  ```

**_relabel_one**

- 659행

  ```text
  # 2026-09-24: same forcing as Hard-1/2, but the "known
  # correct cells" have to be computed here (not stored at
  # mining time) via _fork_safe_cells -- see that function's
  # docstring for why this was needed (the re-searched policy
  # target could otherwise inherit the same weak-prior
  # blind spot it's supposed to correct).
  ```

- 674행

  ```text
  # 2026-09-23: 1.0 -> 0.95 (label smoothing). value_overfit_check.py
  # found the live value head badly overconfident on FRESH,
  # never-trained-on self-play games (held-out MSE 1.36 --
  # worse than a constant-0 predictor; 61% of predictions
  # saturated past |0.9|; the model's own "very confident"
  # bucket only actually won ~67% of the time, not ~100%).
  # Confident artificial labels (this class and Hard-7) are
  # the most likely amplifier -- softening them off the
  # tanh's saturated tail should reduce that overconfidence
  # without giving up the corrective signal's direction.
  ```

- 686행

  ```text
  # 2026-09-23: unlike Hard-1/5, this is a confident NEGATIVE
  # label, not a policy-only class -- letting `other` reach an
  # unstoppable fork is a near-certain loss for `player`
  # regardless of what happens afterward (session evidence:
  # test_fork_prevention.py found 0/3 real HLT forks
  # preventable even with a fresh 10000-sim search, so this
  # needs a direct value correction, not just more search at
  # relabel time). -1.0 -> -0.95: same label-smoothing
  # rationale as Hard-1 above.
  ```

- 695행

  ```text
  # 2026-09-26: REMOVED -- policy-only now (value=None). The
  # board stored here is the position BEFORE the defender
  # moves, where fork-safe replies exist, so "near-certain
  # loss" was the value of the position AFTER a bad reply, not
  # this one. pessimism_probe (14 Hard-7 positions, safe reply
  # then best-vs-best playouts) measured raw value -0.78 vs an
  # empirical outcome of -0.04: this label was teaching the
  # value head that any fork threat means the game is lost.
  ```

**_get_open_three_injection_samples**

- 816행

  ```text
  # 2026-09-24: generate_fork_examples() (the open-3-3 / double-open-
  # three geometry -- see _fork_geometries) added to this SAME
  # continuous pool. It was already implemented and already used to
  # get delivered via tactical_refresh()'s burst fine-tune, but that
  # call was removed from the main loop this session (see the
  # tactical_refresh_interval guard's comment) and nothing replaced
  # it for this specific curriculum the way open-three's own got
  # migrated here on 2026-09-22 -- so it silently went from
  # "burst-delivered" to "never delivered". A held-out T-shape probe
  # this session (test_t_shape_33fork_realistic.py) found this
  # measurably regressing over the ~200 epochs since: raw policy
  # rank for the correct denial cell 3rd/121 at the Gen12 reset
  # point -> 47th/121 at the current live checkpoint, MCTS solving
  # it correctly at epoch 0 and failing every checkpoint since --
  # i.e. real forgetting of knowledge the supervised-pretrain
  # checkpoint already had, not just slow acquisition. No new
  # examples are being authored here -- this is reconnecting an
  # existing generator to an existing delivery mechanism.
  ```

**tactical_refresh**

- 1089행

  ```text
  # 2026-09-21 lr-fix restart: 5e-4 -> 5e-5. This was tuned (2026-09-10)
  # against the OLD main-loop lr=1e-3 (a 0.5x ratio); the main loop is now
  # 1e-4 (this restart's lr-fix), and tactical_refresh_interval dropped
  # 21->5 for denser gate-check signal, so this refresh now fires ~4.2x
  # more often. Left at 5e-4 it would repeatedly overpower the just-
  # lowered main LR with the same-ratio-violating nudge, ~4x more often --
  # likely why gate checks are still stuck at ~0% 25+ epochs into this
  # restart. Scaling down to preserve the original 0.5x-of-main-lr ratio.
  ```

## `shared/naps_league.py`

**(module level)**

- 34행

  ```text
  # 2026-09-30: epoch%3 -> Dawn / ep645 / normal draw
      (on: LEAGUE_ANCHOR_ROTATION = True)
  ```

- 148행

  ```text
  # 2026-10-08 (user): many opponents per epoch instead of one
      (on: LEAGUE_NUM_OPPONENTS = 12)
  ```

**_pick_league_opponent**

- 91행

  ```text
  # 2026-09-30 (user, option A): deterministic rotation over epochs --
  # epoch % 3 == 0 -> first anchor (Dawn ep625), == 1 -> second anchor (ep645),
  # == 2 -> the normal draw below (best / deep history / recency pool).
  ```

- 107행

  ```text
  # 2026-09-30 (user): fixed "anchor" opponents -- the graduated Dawn (ep625) and the
  # long-standing best ep645 -- so league play keeps facing the strongest known models
  # while the gate is stuck. Checked right after the best-checkpoint draw.
  ```

**_play_one_mcts_vs_league_game**

- 239행

  ```text
  # 2026-09-28: game-length cap (stones on the board); capped = draw.
  ```

- 289행

  ```text
  # 2026-09-28: PCR on the league side too (same full_search_prob
  # as the MCTS side): a fast search only picks the move and is not
  # recorded. At full_search_prob=1.0 this is the old behavior.
  ```

- 311행

  ```text
  # 2026-10-13 (user): league-side plies are no longer training targets -- the opponents are older,
  # weaker checkpoints (some ~700-1,000 equal nodes), so their visit counts taught weaker moves.
  ```

## `shared/naps_selfplay.py`

**_play_one_mcts_game**

- 80행

  ```text
  # 2026-09-28: max_plies caps the game length (plies counted from this
  # game's start position). A capped game is scored a draw, and its
  # survival labels are dropped (total_plies returned as None) since the
  # real remaining length is unknown.
  ```

## `shared/naps_tactics.py`

**generate_tactical_examples**

- 39행

  ```text
  # 2026-09-14: was (2,4) -- clustered-near-window clutter can afford more stones without spreading unrealistically thin
      (on: _tactical_clutter(board, set(cells), n_stones=random.randint(3, 36)))
  ```

**generate_closed_three_examples**

- 130행

  ```text
  # 2026-09-14: was (2,4) -- clustered-near-window clutter can afford more stones without spreading unrealistically thin
      (on: _tactical_clutter(board, set(cells), n_stones=random.randint(3, 36)))
  ```

**generate_fork_examples**

- 192행

  ```text
  # 2026-09-14: was (2,4) -- clustered-near-window clutter can afford more stones without spreading unrealistically thin
      (on: _tactical_clutter(board, exclude, n_stones=random.randint(3, 36)))
  ```

**generate_double_four_examples**

- 261행

  ```text
  # 2026-09-14: was (2,4) -- clustered-near-window clutter can afford more stones without spreading unrealistically thin
      (on: _tactical_clutter(board, exclude, n_stones=random.randint(3, 36)))
  ```

**(module level)**

- 274행

  ```text
  # 2026-09-15 (added after re-diagnosing the HLT-loss value/MCTS blind spot --
  # see scratch/multiproc_investigation/rediag_value_blindspot.py, run against
  # the live epoch443 checkpoint): a genuine OPEN three (3-in-a-row with BOTH
  # immediately-flanking cells empty) is a shape this curriculum never actually
  # covered. generate_tactical_examples' "4-of-5-gap" family is always one move
  # from a proven outcome (mover_completes or a single forced block);
  # generate_closed_three_examples' shape has only ONE open end. An open three
  # has TWO equally-correct response cells (either flank creates/denies an open
  # four), and re-testing 6 real/synthetic open-three positions against the
  # live checkpoint found the model still gets roughly half of them wrong --
  # MCTS converging on an unrelated move despite the raw policy prior usually
  # still ranking at least one true response cell near the top -- the same
  # "good-ish prior, MCTS/value diverges" pattern diagnosed earlier, just never
  # targeted by a dedicated curriculum shape until now.
  ```

- 358행

  ```text
  # 2026-09-15: Position A (epoch291 HLT loss) is a real open-three-defense
  # instance whose raw POLICY prior (not just the value head/MCTS search) has
  # drifted onto the historically-wrong move after repeated self-play/HLT
  # reinforcement -- confirmed by direct re-check against the live checkpoint
  # (rank 1/121, 12.6% prior on the wrong cell, vs the two correct blocks at
  # 3-5% each). A diffuse generic curriculum (generate_open_three_examples
  # above) teaches the general shape but has no special weight on this EXACT,
  # already-entrenched association -- unlearning it needs direct, repeated
  # counter-supervision on the specific pattern, not just broader coverage.
  ```

- 367행

  ```text
  # 2026-09-16: Position B (live test game, Claude vs the epoch462-promoted
  # checkpoint) -- a DIAGONAL open-three missed in a denser, real-game-like
  # position (5 X + 4 O stones already on the board, unlike Position A's
  # emptier board) despite the model correctly blocking a near-identical
  # horizontal open-three earlier in the SAME game. Confirms the open-three
  # fix from generate_open_three_examples works on the shape in isolation but
  # does not yet fully generalize to busier boards -- clutter density was
  # widened 3-36 (was 3-6, see _tactical_clutter calls throughout this file)
  # specifically to close this gap generically, and this exact position is
  # also hard-mined directly, same rationale as Position A.
  ```

- 388행

  ```text
  # 2026-09-16: Positions C/D/E found by the autonomous tactical_watch.py
  # loop's very first run -- randomly-generated open-three test boards
  # with dense (heavy-clutter) surrounding stones, all missed by the
  # live checkpoint (block_visit_share 2.0-6.5%) despite the widened
  # 3-36 clutter curriculum deployed the SAME epoch (expected: that
  # curriculum hasn't been trained on yet, only code-deployed -- these
  # are exactly the kind of busy-board misses it should start fixing
  # once the next tactical_refresh cycle actually trains on it).
  ```

- 408행

  ```text
  # 2026-09-16: 2nd tactical_watch.py manual run (~7h after the 1st,
  # loop wakeup chain appears to have stopped firing autonomously --
  # see chat) -- 3 more dense-board misses, block_visit_share
  # 0.0-1.2%. 2/5 tested passed this time (vs 0/3 the first run),
  # consistent with gradual, partial improvement rather than solved.
  ```

- 425행

  ```text
  # 2026-09-23 Position F: HLT (human) loss for Gen13/NAPS, epoch129,
  # cumulative_epoch 129 checkpoint. Root value swung to +0.977 (AI very
  # confident of winning) at move 24, then collapsed to -0.921 by move 36
  # as the human found the refutation -- see the session's HLT-analysis
  # discussion. Root cause traced by replaying row 8's stone history:
  # after X's move at (8,3) (ply 35), row 8 reads "...X.XXXOX." -- X at
  # cols 3,5,6,7 with a SINGLE empty gap at col 4 and O blocking col 8.
  # This is a plain 4-of-5-gap shape (the OLDEST tactical curriculum
  # family in this file, generate_tactical_examples) with exactly one
  # winning cell (8,4) -- yet the live model's actual move-36 search
  # never surfaced (8,4) in its top-5 candidates at any simulation
  # checkpoint (only (3,7)/(8,4-adjacent) cells the network's prior
  # never weighted), meaning this wasn't a search-depth miss but a
  # policy-prior blind spot on a pattern the model has trained on
  # extensively. Distinct from Positions A-E (all open-three shapes) --
  # this is a gap-four shape missed on a busy, asymmetric real board.
  ```

**generate_open_three_examples**

- 308행

  ```text
  # 2026-09-23: edge/corner oversampling (added earlier today, REMOVED
  # here) -- the organic self-play open-three test
  # (open3_organic_test.py) found the model missing threats located
  # centrally while repeatedly playing near row 0 instead, matching
  # exactly the kind of "edge habit" this oversampling could produce
  # if it overshot: skewing the network toward edge/corner moves as a
  # general prior rather than genuinely improving edge-tactic
  # recognition. Reverting to a flat, position-independent variant
  # count (matches the window-count distribution the board's own
  # geometry produces, not artificially reweighted) to isolate
  # whether the organic-test miss rate improves without it.
  ```

- 322행

  ```text
  # 2026-09-21: 3->8 variants (1 clean + more clutter diversity) --
  # open-three defense is the pattern the passive open3 tracker
  # has shown persistently flat/noisy (0.2-0.7 miss rate) despite
  # training, so this deliberately over-represents it relative to
  # the other four tactical families in the combined curriculum.
  ```

- 327행

  ```text
  # 2026-09-22: 8->14 -- tactical_refresh() (the periodic burst
  # supervised re-injection this curriculum used to feed) was
  # removed the same day: isolated testing
  # (scratch/.../tactical_refresh_destroy_test.py) measured ONE
  # refresh call immediately dropping vs-Det win rate 65%->25%,
  # and a separate retention check
  # (scratch/.../tactical_retention_check.py) found that gain
  # decayed from ~100% back to ~70% tactical accuracy within just
  # 2 epochs anyway -- net negative. Replaced by continuous
  # per-epoch injection into the ordinary replay buffer (see
  # _get_open_three_injection_samples), so this pool now needs
  # more inherent variant diversity since it's no longer refreshed
  # by a dedicated fine-tuning pass, just sampled from directly.
  ```

**generate_diagnosed_positions_examples**

- 496행

  ```text
  # 2026-09-16: widened 2-5 -> 3-36 for consistency with the other families (see generate_open_three_examples' header comment)
      (on: _tactical_clutter(board, placed_cells | set(correct_transformed), n_stones=random.randint(3, 36)))
  ```

## `shared/naps_trainer.py`

**train_mcts_alpha_zero**

- 105행

  ```text
  # 2026-09-26: PCR for vs-Det/vs-League games, kept separate so PCR can
  # be applied to self-play only. None = same as full_search_prob (the
  # previous behavior, where one value was passed to all three).
  ```

- 109행

  ```text
  # 2026-09-26: play self-play + vs-Det + vs-League games in ONE
  # multi-process pool (selfplay_mp_unified) instead of three sequential
  # phases. False = the previous sequential path (which is also the
  # automatic fallback whenever the unified pool fails).
  ```

- 114행

  ```text
  # 2026-09-27: value target = (1-w)*z + w*q for generated games (see
  # mixed_value_target). 0.0 = pure game result (previous behavior).
  ```

- 117행

  ```text
  # 2026-10-10 (user): half-life decay of the q weight. From epoch
  # value_target_q_decay_start on, w(epoch) = value_target_q_weight *
  # 0.5 ** ((epoch - start) / value_target_q_halflife); below
  # value_target_q_floor it becomes exactly 0 (pure z). halflife 0 = off.
  ```

- 124행

  ```text
  # 2026-09-28: cap self-play / league game length (plies); None = no cap.
  ```

- 126행

  ```text
  # 2026-09-28: vs-Deterministic games inside the refresh-epoch MINING pass
  # (the old hard-coded default was 16). 0 = mining uses self-play and
  # league games only.
  ```

- 140행

  ```text
  # 2026-09-21 lr-fix restart: base_lr 0.001->0.0001 and T_0 450->150.
  # Weight-drift measurement (see conversation/workflow findings) showed
  # lr=0.001 drifts the post-supervised-pretrain checkpoint 1.86x farther
  # (matched-epoch comparison) than lr=0.0001, and drift never reverses
  # even during low-instantaneous-LR windows of the T_0=450 schedule --
  # consistent with the gate-check having stayed stuck at ~0% for 564+
  # epochs after the Gen10 supervised-pretrain restart. The prior Gen7->
  # Gen8 supervised-pretrain restart broke through gate-check right around
  # the point its T_0=150 schedule decayed to ~1.2e-4 -- this restart
  # starts there directly instead of spending hundreds of epochs decaying
  # down to it.
  ```

- 151행

  ```text
  # 2026-09-30 (user): gate stuck 15 checks (650-720) while losses kept falling ->
  # halve the LR band (1e-4..5e-5 -> 5e-5..1e-5) to consolidate instead of drifting.
  ```

- 153행

  ```text
  # 2026-10-11 (user): ep937 cooldown ended -> LR jumped 1e-5 -> 4.4e-5 and ep940 fell to ~1,092 Rapfi nodes.
  # Cosine peak 5e-5 -> 2e-5 (band 2e-5..1e-5) to polish rather than shake before the festival.
  # LR band per run (run_config.py): 11x11 NAPS 2e-5..1e-5, 11x11_v2 5e-5..1e-5, 15x15 1e-4..2e-5
  # (2026-10-15, user: the 2e-5 polish band of the mature 11x11 net kept the young 15x15 net flat).
  ```

- 162행

  ```text
  # 2026-10-12 (user): restore Adam's moments across restarts. Every restart used to reset them
  # (loss spikes at 905/911/913/945). Loaded only if saved right after the epoch we resume at;
  # the LR stays whatever the schedule/cooldown says now (the saved param_groups LR is discarded).
  ```

- 166행

  ```text
  # 2026-10-13: a fresh Adam moves every weight by ~LR on its first steps (bias-corrected m/sqrt(v) ~ sign(g)),
  # which caused the loss spikes after every restart/rollback (ep1003 0.38, ep1047 0.41). If the moments
  # are not restored, ramp the LR over the first epochs instead.
  ```

- 296행

  ```text
  # 2026-10-12 (user): second promotion path -- Rapfi ladder. The gate (vs the current best only)
  # can be blocked by a matchup ("상성"); if the ladder watcher finds a measured checkpoint stronger
  # than the best with P >= 80% (bootstrap), it writes model/ladder_promote.json and the checkpoint
  # becomes the best AND the self-play generation model here, at the start of the next epoch.
  ```

- 470행

  ```text
  # 2026-09-22: continuous per-epoch open-three curriculum injection,
  # replacing tactical_refresh()'s periodic burst (see
  # _get_open_three_injection_samples docstring for why).
  ```

- 477행

  ```text
  # 2026-09-23: dedicated Hard-7 (general fork) quota, on top of
  # hard_samples' whole-pool draw above -- see
  # _get_fork_mistake_samples docstring (measured: without this,
  # Hard-7 got ~1-2 exposures/epoch by chance, essentially no real
  # training signal one full mining cycle after being added).
  ```

- 509행

  ```text
  # 2026-10-13 (user, clean rollback to ep960): with an emptied buffer, only generate data until
  # NAPS_MIN_BUFFER samples exist, so the first training steps do not overfit a tiny buffer.
  ```

- 543행

  ```text
  # 2026-10-14 (15x15): Adam moments restored on resume were loaded while the model sat on the CPU;
  # keep them on the training device or the first step fails (cuda:0 vs cpu).
  ```

- 585행

  ```text
  # 2026-09-26: value=None samples (policy-only hard classes) carry
  # NaN and are masked out here. Before this they were stored as
  # 0.0 and silently trained the value head toward "draw".
  ```

- 673행

  ```text
  # 2026-09-22: tactical_refresh() (the periodic burst supervised
  # re-injection that used to run here) was REMOVED -- see
  # _get_open_three_injection_samples docstring for the measured
  # before/after evidence (immediate vs-Det win-rate drop, decays back
  # within ~2 epochs anyway). The open-three/diagnosed-position
  # curriculum is now injected continuously every epoch instead (see
  # the open_three_samples call above, right after hard_samples).
  # refresh_mined_tactical_pool (mining fresh REAL tactical examples
  # from current self-play, unrelated to the burst fine-tune) and the
  # promotion gate check both still run on this same interval cadence
  # -- neither was implicated by the destroy/retention tests, and the
  # gate check in particular still needs a periodic "checkpoint" cadence
  # to compare against, unrelated to whether tactical_refresh runs.
  ```

- 711행

  ```text
  # 2026-10-11 (user): keep the weights of every gate where the OLD rule (W + D/2 >= 55%) would
  # have promoted but the new loss-penalty rule did not, so the new rule's judgement can be checked
  # later (e.g. Rapfi ladder). Only these disagreements are saved (~34 MB each).
  ```

- 766행

  ```text
  # 2026-09-11 bugfix: this used to be called at the TOP of the loop
  # (save_cumulative_epoch(epoch), before this epoch's own work ran) --
  # a lagging save that only confirmed epoch N complete once epoch N+1
  # had already STARTED. A manual stop anytime between epoch N's report
  # line printing and epoch N+1 beginning left the state file at N-1,
  # so a restart silently redid the whole already-completed epoch N
  # (confirmed live: two differently-valued "Epoch 240" report lines
  # from today's stop/restart, status_check's reset-heuristic then
  # misread the non-monotonic epoch number as a full history reset).
  # Saving epoch+1 here, right after this epoch's own checkpoint write,
  # closes that window: a kill after this line correctly resumes at
  # epoch+1; a kill before it correctly redoes only epoch (never more).
  ```

- 778행

  ```text
  # 2026-10-12: optimizer moments for the next restart (atomic write)
      (on: try:)
  ```

## `shared/nets.py`

**(module level)**

- 220행

  ```text
  # 2026-09-21: experimental fused-block architecture, built alongside (not
  # replacing) alphago_lee.py's V1 network (GomokuNetV1). Imports everything except
  # the model classes from board_encoding -- fork-mask helpers, board_to_input_array,
  # board_to_input_tensor, BOARD_SIZE/NUM_CELLS/NUM_INPUT_CHANNELS -- so there is
  # exactly one source of truth for the board/tensor representation; only the
  # network internals differ here.
  #
  # Combines three changes discussed in chat, each independently motivated:
  # 1. Depthwise-separable convolutions (MobileNetV3/EfficientNet-style) instead
  #    of full 3x3 convs -- ~8-9x fewer multiply-adds per conv layer at this
  #    channel count (256), directly targeting this project's recurring GPU
  #    bottleneck.
  # 2. GroupNorm instead of BatchNorm -- GroupNorm normalizes per-sample (no
  #    batch-wide running statistics), so it behaves identically regardless of
  #    batch size or train()/eval() mode. This eliminates an entire bug class
  #    this project has hit more than once (the 5x5 gate-check BatchNorm
  #    train-mode bug, 2026-08-29, and others referenced in this session) at
  #    the source, rather than continuing to rely on callers remembering to
  #    call .eval() before every MCTS leaf batch.
  # 3. Squeeze-and-Excitation channel attention (KataGo uses this in its own
  #    trunk) -- lets the network dynamically emphasize/suppress channels based
  #    on GLOBAL board context, which suits whole-board pattern reasoning
  #    (open-three / fork detection) better than treating every channel with
  #    fixed importance regardless of position.
  #
  # This combination (depthwise-separable conv + SE, "MBConv"-style) is not
  # novel -- it is the standard building block of MobileNetV3 and EfficientNet,
  # chosen here because it is well-validated in the literature, not because it
  # is untested.
  #
  # IMPORTANT: this is architecturally incompatible with existing checkpoints
  # (nets.GomokuNetV1's ResBlock uses full Conv2d + BatchNorm2d
  # with different parameter shapes) -- a model built from this file must be
  # trained from scratch, it cannot load model/best_mcts_11x11.pth or any
  # existing checkpoint. Nothing under 11x11_naps/model/ is touched by this file
  # merely existing; it only takes effect wherever something explicitly
  # imports GomokuNetV2 instead of the original.
  ```

**__init__**

- 355행

  ```text
  # 2026-09-21: back to 256/7 (matching the original ResBlock
  # architecture's sizing) -- the 320/16 bump was sized assuming
  # depthwise-separable's compute savings, which were dropped
  # (see DSConv's docstring: measured far slower wall-clock on
  # this GPU). Without that savings, 320/16 would have been a
  # 3.5x-bigger network (29,993,062 params) than the original,
  # not a lighter one -- reverting to the original size keeps
  # this an apples-to-apples GroupNorm+SE comparison.
  ```

## `shared/promotion_sms.py`

**(module level)**

- 26행

  ```text
  # second tier LADDER_ALERT2 = 20000 (정오 bar), same logic, added 2026-10-12 # 2026-10-12 (user): also text when a ladder run's equal point is >= 10,000 nodes
      (on: LADDER_ALERT = 10000)
  ```

**text_for**

- 52행

  ```text
  # 2026-10-12: ladder promotion path
      (on: if m.re is LADDER_PROMO:)
  ```

**ladder_hits**

- 86행

  ```text
  # 2026-10-12: a short ladder (e.g. 1500/3000/5000) can extrapolate far past its top level;
  # only claim a value the ladder actually reached, otherwise say "above the top level"
  ```

## `shared/rapfi_ladder.py`

**(module level)**

- 28행

  ```text
  # 2026-10-10 night (user): 20000 added (the 정오 bar sits around 20k)  # 2026-10-10 evening (user, option B): 300 dropped (85%+, no information), 10000 added (890 scored 62% at 5000)  # 2026-10-10 (user): 5000 added -- 860/880 still scored ~42% at 3000, so the top end was unmeasurable  # Rapfi search nodes per move (around the current equal-strength point)
      (on: LEVELS = [int(x) for x in os.environ.get("RAPFI_LEVELS", "1000,2000,3000,5000,10000,20000").split(",")])
  ```

**main**

- 72행

  ```text
  # 2026-09-29 (user decision): the official equal-strength point is a games-weighted
  # least-squares line of Elo vs log10(nodes) over ALL levels, solved for Elo = 0.
  # The single-crossing rule above swung 100..1250 on 20-game noise; kept for reference.
  ```

## `shared/rapfi_ladder_watch.py`

**(module level)**

- 33행

  ```text
  # 2026-10-09: the ep820 run died after the problem check -- on Windows, a child output that the
  # parent cannot decode with the locale codec leaves r.stdout as None. Force UTF-8 both ways.
  ```

**main**

- 69행

  ```text
  # 2026-10-14: resume a missed run, e.g. "python ../shared/rapfi_ladder_watch.py 20"
      (on: if len(sys.argv) > 1:)
  ```

- 73행

  ```text
  # 2026-10-13 (user): gate-passed candidates are measured as soon as they appear
  ```

- 98행

  ```text
  # 2026-10-13: ladder + grading are deterministic (node-limited Rapfi, no search noise), so a best
  # whose weights equal the last measured snapshot would replay identical games -- skip it.
  ```

- 104행

  ```text
  # 2026-10-12: skip re-measuring a best that the ladder reference already covers (same weights)
  ```

- 122행

  ```text
  # 2026-09-28: also measure the newest training checkpoint (not just the gated best),
  # so progress is visible even while the gate keeps rejecting candidates
  ```

- 141행

  ```text
  # 2026-10-12 (user): Rapfi-ladder promotion path (P >= 80% stronger than the best)
  ```

## `shared/rapfi_match.py`

**(module level)**

- 34행

  ```text
  # 2026-10-23 (user): VCF on wherever the engine has it (was off: engine default depth 0); VCF=0 turns it off
  ```

- 44행

  ```text
  # 2026-10-11: RAPFI_MORE_OPENINGS=1 -> all 25 cells within 2 of the centre (validation runs with >16 games,
  # so games do not repeat; the default 8 openings keep the ladder identical to its history)
  ```

- 49행

  ```text
  # 2026-10-10: games are deterministic (Rapfi single-threaded + node limit, our side greedy without noise), so a
  # level can be split across RAPFI_PAR worker processes (game g -> worker g % PAR) with identical per-game results.
  ```

**main**

- 134행

  ```text
  # 2026-10-22: 11x11_v3 checkpoint (NetV3, 128x8)
      (on: if any(k.startswith("opp_policy_head.") for k in sd):)
  ```

- 149행

  ```text
  # 2026-10-10: RAPFI_THREADS > 1 plays that many games at once in this process and evaluates their
  # positions in ONE shared GPU batch (cross-game batching). The search itself is unchanged (1100 sims,
  # batch 16 per game), but cuDNN picks different kernels for different batch sizes, so outputs differ by
  # ~1e-5 and a near-tied move can occasionally flip: results are statistically, not bit-for-bit, equal.
  ```

- 253행

  ```text
  # 2026-10-13 (15x15): Rapfi occasionally desyncs ("ERROR Unknown command") on 15x15;
  # restart the engine and replay the same game (deterministic) instead of killing the ladder.
  ```

## `shared/rapfi_match_pooled.py`

**worker**

- 76행

  ```text
  # 2026-10-23 (user): VCF on wherever the engine has it
      (on: if VCF and os.environ.get("VCF", "1") != "0":)
  ```

## `shared/rapfi_progress.py`

**(module level)**

- 17행

  ```text
  # 2026-10-10 evening: 300 dropped, 10000 added
      (on: LEVELS = [1000, 2000, 3000, 5000, 10000, 20000])
  ```

**show**

- 43행

  ```text
  # 2026-10-12: group by the run's checkpoint and Rapfi node count (not by position),
  # so one-off runs with other levels (e.g. 1500/3000/5000) and cancelled runs show correctly
  ```

## `shared/selfplay_mp.py`

**__init__**

- 546행

  ```text
  # 2026-09-26: with more tasks than workers (now the case on the
  # fallback path), a failed call can leave this process's feeder
  # thread blocked on unread tasks, which multiprocessing would join
  # forever at exit. Never join these feeders; shutdown() also drains.
  ```

**shutdown**

- 666행

  ```text
  # 2026-09-26: drop still-queued tasks first so workers reach their
  # sentinel after their current game (see __init__ note).
  ```

**generate_mcts_self_play_data_mp**

- 936행

  ```text
  # 2026-09-26: real ply indices (PCR skips recording fast-search
  # plies, so len(history) - k would undercount plies-to-end).
  ```

## `shared/selfplay_mp_unified.py`

**(module level)**

- 83행

  ```text
  # 2026-09-27: refresh/gate epochs -- the training model being mined with /
  # gated, and the tracked best checkpoint it is gated against.
  ```

**_run_task**

- 173행

  ```text
  # 2026-10-08: one tag per league opponent
      (on: ltag = spec[2] if len(spec) > 2 else TAG_LEAGUE)
  ```

**generate_epoch_games_unified**

- 775행

  ```text
  # 2026-10-08 (user): league_model may be a LIST of opponents; league game i
  # then plays opponent i % len(list), each under its own eval tag.
  ```

- 832행

  ```text
  # 2026-10-21 (user): self-play + vs-League games in C++ threads with large GPU
  # batches (selfplay_cpp); the worker pool stays as the fallback and for vs-Det.
  ```

## `shared/status_all.py`

**show**

- 55행

  ```text
  # 2026-10-12: newest finished grading + any grading still running (one-off tactical_*.out)
  ```

## `shared/tactical_problem_check.py`

**(module level)**

- 33행

  ```text
  # 2026-10-23 (user): VCF on wherever the engine has it (was off: engine default depth 0); VCF=0 turns it off
  ```

- 47행

  ```text
  # 2026-10-11: node limit replaces the time limit (see main)
      (on: RAPFI_NODES = 1000000)
  ```

**main**

- 53행

  ```text
  # 2026-10-23: FORCE_CPU=1 keeps the training GPU free
      (on: dev = torch.device("cuda" if torch.cuda.is_available() and not os.environ.get("FORCE_CPU") else "cpu"))
  ```

- 55행

  ```text
  # 2026-10-22: 11x11_v3 checkpoint (NetV3, 128x8)
      (on: if any(k.startswith("opp_policy_head.") for k in sd):)
  ```

- 70행

  ```text
  # 2026-10-14 (15x15): no problem set yet (researched once eq >= 1000)
      (on: if PROBLEMS is None:)
  ```

- 73행

  ```text
  # 2026-10-22 (user): PROBLEM_IDS=id1,id2 grades only those problems (all as tier 1, nothing logged)
  ```

- 77행

  ```text
  # 2026-10-11 (user): node-limited opponent so a grading run gives the same result every time (the 3 s
  # time limit made results depend on PC load; ep920 got 3/3 in one run and 1/3 in another with the same moves).
  # 1,000,000 nodes ~ what 3 s searched on an idle PC (~417k nodes/s), so difficulty stays comparable.
  ```

- 83행

  ```text
  # 2026-10-08 (user): 3-tier check. Problems without "tier" always run (tier 1 includes the
  # stage-1 graduation set). Tier 2 (5 representative loss-backtrack problems) runs only if every
  # stage-1 problem passed; tier 3 (the remaining ones) only if every tier-2 problem passed.
  ```

- 93행

  ```text
  # 2026-09-28 (user, option A): 3 trials at slightly different search budgets,
  # pass if >= 2 of 3 pass -- one deterministic game per problem was too noisy
  ```

- 150행

  ```text
  # 2026-09-28: stage 1 (graduation) problems vs stage 2 (stretch goal, together with Rapfi 2000)
  ```

## `shared/train_run.py`

**(module level)**

- 10행

  ```text
  # 2026-10-13 (user, clean rollback to ep960): train only once the emptied buffer holds 40k samples of the
  # new best's self-play (harmless afterwards: the buffer stays far above this).
  ```

**main**

- 62행

  ```text
  # 2026-09-10 intervention. Measured root cause: at 250 sims over ~45
  # radius-2 candidates the search gets ~5 visits/child, below what a 2-ply
  # refutation needs, so the visit distribution recorded as the policy
  # target was FLATTER and less accurate than the network's own prior
  # (block-position top-1: prior 60%, MCTS@250 10%, MCTS@900 67%). Training
  # was descending, not plateauing. Everything below either raises
  # visits-per-child or stops diluting the prior before the search can use it.
  ```

- 69행

  ```text
  # 2026-09-21 lr-fix restart: tactical_refresh_interval 21->5 so the gate
  # check (which rides this same cadence) samples ~4x more often over the
  # ~80-epoch observation window, instead of only 3-4 checks total.
  ```

- 72행

  ```text
  # 2026-10-12 17:3x (user): temperature_moves reverted 6 -> 4 (ep980/1000 ladder fell). Earlier: 4 -> 6 (more varied midgames, quiet-threat exposure); workers 10 -> 8 (GPU 92-96C)
  ```

- 73행

  ```text
  # 2026-10-13 (user): fast_sims 150 -> 300 (less noisy game outcomes = cleaner value targets)
  ```

- 74행

  ```text
  # 2026-10-13 (user): q weight fixed at 0.2 (value target = 0.8 z + 0.2 q); half-life decay off (halflife=0)
  ```

- 80행

  ```text
  # 2026-10-10 ~20:50 (user): q weight 0.5 -> 0.2 (value target = 0.8 z + 0.2 q)
      (on: value_target_q_weight=0.2,)
  ```

- 81행

  ```text
  # 2026-10-10 (user): q weight halves every 20 epochs (0.2 @905, 0.1 @925, 0.05 @945, ~0 after ep990)
      (on: value_target_q_halflife=0, value_target_q_decay_start=905,)
  ```

## `shared/watchdog_.py`

**restart_training**

- 55행

  ```text
  # 2026-09-23: switched OFF the Task Scheduler path. Confirmed via
  # Task Scheduler that "AZ11x11_Training" also carries a LogonTrigger
  # (fires on every Windows logon, independent of whether a trainer is
  # already running) -- after today's reboot this produced two
  # independently-advancing _restart_train.py processes at once (an
  # interleaved, offset-by-~10-epochs duplicate report sequence lasting
  # hours), and this watchdog has no admin rights to remove that trigger
  # itself. Launching directly here (same detached Start-Process pattern
  # used everywhere else in this project) means the ONLY way a trainer
  # ever starts is: (a) manually, or (b) this watchdog when it's
  # genuinely missing -- one code path, no hidden scheduler trigger to
  # race against.
  # No stdout/stderr redirect here -- training_report_11x11.txt (written
  # directly by report_logger, not via stdout) is the authoritative log
  # regardless, and train_stdout.log has an unexplained persistent lock
  # (unrelated process, survives full process kills) not worth fighting
  # inside an unattended restart path.
  ```

## 설명 속 날짜 (2차 정리)

설명 문장 중간에 있던 날짜와 `(user)` 표시를 지운 주석의 원문이다. 날짜 자체가 정보인 것
(사건 시각, 데이터 이름, "until 2026-11 this file was ...")은 코드에 그대로 두었다.

**`11x11_naps/hlt/web/attack_pick.py`**

- 30행: `# (2026-10-26 fix: any proven flag used to disable the pick, and far-off losing moves are almost always`

**`11x11_naps/hlt/web/live_dashboard.py`**

- 161행: `# ---- 2026-10-16 (user): admin settings for the public board, written into index.html's CFG block ----`

**`11x11_naps/hlt/web/server.py`**

- 339행: `# opening variety (2026-09-28): pick among near-best candidates in proportion to visits,`

**`11x11_naps/hlt/web/spectate.py`**

- 49행: `# sample near-best moves early so repeated pairings differ (2026-10-11: 6 -> 8)`
- 77행: `# 10/26: 530 = NOON (정오), festival pick`

**`11x11_v2/elo_timeline_chart.py`**

- 1행: `# Equal points as Elo over wall-clock time: NAPS (ladder_ci.json), v2, v3 (2026-10-22, user).`

**`11x11_v2/kaggle/kaggle_plan.py`**

- 33행: `# session 4 (11-02): adaptive catfish (G) on seeds 2 and 3, each against the same seed's A; attach session 1's output`

**`11x11_v2/ladder_all_chart.py`**

- 1행: `# 11x11_naps-folder 11x11 run (NAPS, ep540~) + 11x11_v2 equal points with 68/95% bands, one log y-axis (2026-10-22).`

**`11x11_v2/ladder_all_v3_chart.py`**

- 1행: `# ladder_all_chart.py + a third panel for 11x11_v3 (Gumbel cold start), same bootstrap as ladder_v2_chart (2026-10-22).`

**`11x11_v2/ladder_v2_chart.py`**

- 1행: `# 11x11_v2 equal-point chart with bootstrap 68/95% bands (2026-10-22). Data: rapfi_ladder_watch.log / ladder_ep40_rest.log.`

**`11x11_v2/v3/arms.py`**

- 29행: `# 11-01 why does catfish help/hurt? (seed 2, against A_nocatfish_s2)`
- 32행: `# 11-02 G arm (user: find the conditions where catfish works): B's catfish lost 21/24 to the main net (too weak),`

**`11x11_v2/v3/catfish_chart.py`**

- 1행: `# v3 CATFISH figure for the report (2026-10-23): value loss before/after catfish, and per-epoch losses to each catfish.`
- 11행: `# 10/24: policy0 -> policy_half at ep129`
- 21행: `# 10/24: games per catfish dropped 16 -> 8 at ep139, so plot the loss RATE (10-epoch rolling mean)`

**`11x11_v2/v3/cpp/patch_gumbel.py`**

- 1행: `# one-off patch: adds the Gumbel root to the v3 copy of mcts_core_solver.cpp (2026-10-22)`

**`11x11_v2/v3/head2head.py`**

- 69행: `# best effort: a viewer reading the file can make os.replace fail on Windows (11-01: crashed 2 shards)`

**`11x11_v2/v3/ladder_watch_v3.py`**

- 19행: `# 10/25 (user): measure every 25 epochs`
- 39행: `# hardest first (2026-10-22 user: decide sooner); 2026-10-22: ep70 quick ladder already scored 50% at 3000`
- 54행: `# 10/25 (user, option 1): a flat ladder gives no usable crossing (ep500: ~0 nodes); keep that result as it is, but`

**`11x11_v2/v3/policy_loss_chart.py`**

- 1행: `# v3 policy / value loss with key events (2026-10-23, user).`

**`11x11_v2/v3/train_v3.py`**

- 65행: `# Same time (user): policy0 dropped -- the main net beat it 95-97% (ep79-128), too weak to teach anything.`
- 70행: `# G arms (arms.py, 2026-11-02): after every epoch the main net's score vs each catfish (EMA) is pushed toward ~65%:`
- 78행: `# Gumbel full search (recorded). 2026-10-23 03h (user): 200 -> 400, stronger policy targets to win back the policy`
- 84행: `# share of self-play games that start from a random opening (04:55 user: 0.5 -> 0.25, black-win drift)`
- 150행: `# 04:55 (user): even only, equal stones for both sides`
- 323행: `# 10/25 (user): ladder every 25 epochs needs ep575 etc.`

**`shared/mcts_cpp.py`**

- 48행: `# ---- 11x11_v2 engine (2026-10-21, user); settings in run_config.py ----`
- 289행: `# Persistent C++ search-tree handle (2026-09-11, tree reuse): one`

**`shared/mcts_python.py`**

- 239행: `# Persistent C++ search-tree handle (2026-09-11, tree reuse): one`

**`shared/naps_common.py`**

- 78행: `# the 5x5 sibling file). 2026-08-29: raised 8_000->12_000, mirroring the`
- 304행: `# file -- 2026-09-09 this exact failure (empty train_state file ->`

**`shared/naps_gate.py`**

- 33행: `# 0.25 -> 0.35 (6-8-6, an even result, scored 0.275 and promoted ep1005). 2026-10-12: 0.50 -> 0.25. 2026-10-11: was 0.55 (AlphaGo Zero, draws 0.5)`
- 104행: `# Gate-check speed optimization (2026-09-10): the periodic tactical-refresh`
- 168행: `# only (2026-08-29: same bug confirmed in the 5x5 sibling's gate check).`

**`shared/naps_hard_examples.py`**

- 365행: `# Outcome-based hard-example mining (2026-09-11)`
- 376행: `#   Hard-6 (2026-09-11 addition, not in the original spec): the search's own`
- 391행: `# 800 -> 1800 (2026-09-12): production's own sim budget auto-ramped past this`
- 554행: `# Hard-7 (2026-09-23): general, template-free fork check -- does`
- 823행: `# migrated here on 2026-09-22 -- so it silently went from`
- 1089행: `# lr-fix restart: 5e-4 -> 5e-5. This was tuned (2026-09-10)`

**`shared/naps_league.py`**

- 33행: `# random anchor draw (2026-09-30 first try) -- superseded by the rotation below`
- 229행: `# full_search_prob=1.0 (2026-09-26 PCR fix).`
- 368행: `# Full +-1.0 scale (was 0.5 until 2026-09-10). The halved scale`

**`shared/naps_openings.py`**

- 45행: `# Gen3 eval-diversification (2026-09-09): row-major order meant every vs-Det`

**`shared/naps_selfplay.py`**

- 220행: `# Full +-1.0 scale (was 0.5 until 2026-09-10). The halved scale`

**`shared/naps_trainer.py`**

- 156행: `# (2026-10-15, user: the 2e-5 polish band of the mature 11x11 net kept the young 15x15 net flat).`
- 239행: `# target_batch=24, max_wait=0.05 (2026-09-15, was 48/0.005): live stats`
- 265행: `# Noise-anneal safety valve (2026-09-10): noise_anneal_epochs was just`
- 458행: `# Continuous sampling from the hard-mined pool (2026-09-11) -- the`
- 810행: `# Per-stage wall-clock (2026-09-26): games = all generation games,`

**`shared/naps_vsdet.py`**

- 190행: `# Small depth-0 floor (2026-09-09 diversity pass): depth>=1 minimax`
- 249행: `# equals len(history) at full_search_prob=1.0 (2026-09-26 PCR fix, see`
- 359행: `# Per-depth tally (2026-09-10). The aggregate "1-3 wins per 10 games" line`
- 387행: `# Full +-1.0 scale (was 0.5 until 2026-09-10). The halved scale`

**`shared/nets.py`**

- 78행: `# Gen3 capacity bump (2026-09-08): 192/5 plateaued (Policy Loss`
- 236행: `#    train-mode bug, 2026-08-29, and others referenced in this session) at`

**`shared/problem_difficulty.py`**

- 23행: `# s; 2026-10-10: a Rapfi pipe once stalled for an hour (both engines idle)`

**`shared/rapfi_ladder_watch.py`**

- 23행: `# per ladder level (2026-09-28: 10 -> 20; 11x11_v2 and 15x15: 40)`

**`shared/rapfi_match.py`**

- 32행: `# v2 engine: MEGA_EVAL=30,15 also applies MEGA stage 1 to these noiseless match searches (2026-10-21, user)`
- 51행: `# 3 workers: 20 games at 5000 nodes in ~12 min (2026-10-10 benchmark)`

**`shared/selfplay_mp_unified.py`**

- 73행: `# 11x11_v2 (2026-10-21, user): 64 with 32-leaf requests; other runs 24`
- 619행: `# Separate budget for the refresh/gate task runner (2026-09-27).`
- 892행: `# Refresh / gate epochs (2026-09-27)`

**`shared/train_run.py`**

- 18행: `# Single-instance lock (2026-09-24): the AZ11x11_Training scheduled task's`

**`shared/watchdog_.py`**

- 26행: `# 15x15: "_restart_train15.py", distinct from the 11x11 trainer (2026-10-14)`

## 줄인 설명의 원문 (2026-11 3차 정리)

핵심 학습 모듈의 긴 docstring과 주석 42개를 짧게 다시 쓰면서 지운 원문이다. 줄 번호는 다시 쓰기 직전 기준이다.
일지 투 표현("this session", "today" 등)과 머리말의 `(날짜, user: ...)` 인용을 지운 작은 수정은 git 기록에 있다.

**`shared/board_encoding.py`** _fork_mask_python (docstring)

```text
    """Returns a (BOARD_SIZE, BOARD_SIZE) float32 array with 1.0 at every
    empty cell where placing a curr_plane-owner stone there would create 2
    or more distinct empty cells that would each independently complete a
    WIN_LENGTH-in-a-row on the following move (a "double threat" / fork).

    O(num_lines) instead of the naive O(num_empty_cells * num_lines)
    (simulate-every-candidate-then-rescan-everything) approach -- the naive
    version was the dominant cost of MCTS search on the 10x10/11x11 boards
    (measured: fork detection alone was ~60% of total simulation time).
    Numerically verified identical to the naive simulate-per-candidate
    version on 600+ random boards plus hand-built edge cases (including a
    board with pre-existing separate threats before any hypothetical
    placement, which the naive version's ">=2 total lit cells" definition
    also counts).

    Key fact this relies on: if a line already has exactly one empty cell
    and candidate cell A (currently empty) belongs to that line, A MUST BE
    that one empty cell -- so any line through A that is already an
    immediate-win line has A itself as its target. That means placing at A
    only removes A from the pre-existing lit set (if A itself was already
    a winning move) and otherwise leaves every other pre-existing threat
    untouched; the only NEW threats placing at A can create come from
    lines that currently have exactly WIN_LENGTH-2 of curr's stones and
    exactly 2 empty cells (one of which is A) -- filling A then completes
    that line to WIN_LENGTH-1 with the OTHER empty cell as the new target.

    2026-09-16: the per-candidate combination used to be a Python loop over
    every empty cell (up to ~120 on a sparse board) that rebuilt a new set
    via `(baseline_lit - {a}) | threats.get(a, set())` on EVERY iteration --
    profiled as the single largest cost in the whole dataset-generation
    pipeline (~74s of a 158s run, ~47%; board_to_input_array as a whole was
    77%), because set construction/union has real per-call overhead even
    when the sets involved are tiny. Rewritten to be fully vectorized: the
    identity len((baseline_lit - {a}) | T) = baseline_count - [a in
    baseline_lit] + len(T) - len(T & baseline_lit) (valid because `a` is
    never a member of its own threat set T, by construction) is computed as
    plain numpy array arithmetic over all cells at once -- the only
    remaining Python loop is over `threats.items()`, bounded by ~2x the
    count of "two-empty-cell" lines, not by empty-cell count. Verified
    numerically identical to the original on 966+ boards (random densities
    0-120 stones + hand-built edge cases including multiple simultaneous
    baseline threats); measured ~1.25x faster in isolation, and this
    function's overall dominance in board encoding means that carries
    through to end-to-end dataset generation and self-play.
    """
```

**`shared/fast_board.py`** sims_for_state (docstring)

```text
    """
    A sim-budget diagnostic on this project's 5x5 board (see training
    conversation) found a fixed budget badly under-converged on wide-open
    boards but already converged on narrower ones: at 25 empty cells, the
    top move's visit share was still climbing at 1600 sims (49% -> 79% from
    100 -> 1600); at 14 empty cells, it was flat from 100 all the way to
    1600 sims (~38%, unchanged). So only spend extra search while the board
    is still wide open -- past boost_threshold empty cells, more search
    doesn't change the answer, it just costs time.

    boost_when_narrow=False (default) keeps that exact behavior. On 3x3,
    a diagnosed critical defensive move stayed stuck at ~1/100 visits all
    session, in the mid/late game (few empty cells) -- the opposite end
    from what the 5x5 finding above boosts, since 3x3's branching factor
    (<=9) is already narrower than 5x5's "already converged" 14-empty case
    everywhere. boost_when_narrow=True inverts the comparison so extra sims
    go to low-num_empty states instead, targeting that specific bug.

    That inversion is all-or-nothing (narrow XOR open), so once 5x5 started
    passing boost_when_narrow=True (for its own late-game precision), it
    silently lost the original wide-open boost above -- confirmed still
    needed by a live diagnostic: at 23 empty cells a 100-sim search picked a
    move MCTS itself ranks far worse than the alternative it converges to
    given 600+ sims (visit share 24% -> 92% for the better move as sims
    100->2400, its q swinging from -0.37 to -0.90 in the eventual mover's
    own favor), and that misjudged move was directly upstream of a loss.
    So independently of the boost_when_narrow branch above, also always
    boost the wide-open end (num_empty > NUM_CELLS - 10, i.e. before the
    10th stone is placed) to boosted_sims -- covers both ends at once
    instead of forcing a choice between them.
    """
```

**`shared/mcts_cpp.py`** PooledEvaluator (docstring)

```text
    """
    Combines eval_batch_fn calls from multiple concurrent self-play worker
    threads into fewer, larger GPU forward passes. Measured on this machine
    (GTX 1050): 6 threads each issuing independent batch=8 calls only hit
    2.17x aggregate throughput (concurrent GPU access partly serializes,
    unlike CPU cores) -- pooling those into shared batch~48 calls amortizes
    per-call overhead (kernel launch, .cpu() sync) across more samples.

    One dedicated background thread owns every model() call, so there is no
    lock contention or race on who runs a batch. Worker threads call
    submit() and block on a per-request threading.Event; the background
    thread accumulates requests from the queue until either target_batch
    samples are pending or max_wait seconds pass since the last flush, then
    runs one batched forward pass and wakes every waiting thread with its
    slice of the result.

    Safe to construct once per training run (not per-epoch): it holds a
    reference to the model object, not a snapshot of its weights, so it
    keeps seeing the current weights across epochs as optimizer.step()
    mutates the same nn.Module in place. Call stop() only at process exit.
    """
```

**`shared/mcts_cpp.py`** MCTS.__init__ (docstring)

```text
        """
        eval_batch_size=1 (default) runs the original one-leaf-per-simulation
        search, unchanged. eval_batch_size>1 groups that many simulations'
        selection phases together (via standard virtual loss) and sends their
        leaves through the model in one batched forward call -- measured
        ~1.5-2x faster self-play throughput at batch_size=4-8. This trades a
        small amount of search-order fidelity for speed (a simulation's path
        selection can see virtual-loss-inflated visit counts from other
        in-flight simulations in the same group), so it is only wired into
        the training self-play/vs-Det generation paths, not the formal
        evaluate_vs_deterministic metric or the human-play API, which stay on
        the exact eval_batch_size=1 sequential path.

        shared_cache=None (default) keeps the original behavior: a private
        cache cleared at the start of every search() call. Pass a dict to
        share it across many search() calls instead (e.g. one dict per
        training epoch, reused across every game and every move generated in
        that epoch) -- valid because (board, player) -> (probs, value) is a
        pure function of the model's weights, which are frozen for the whole
        self-play/vs-Det generation phase each epoch. The caller is
        responsible for building a fresh dict every epoch, since
        optimizer.step() changes the weights and would otherwise make the
        cache stale.
        """
```

**`shared/mcts_cpp.py`** MCTS._cpp_eval_fn (docstring)

```text
        """Single-board eval callback for _mcts_core.mcts_search. Routes
        through self._eval_cache first so eval_batch_size==1 callers keep the
        exact same per-call-cleared-cache semantics as the pure-Python path
        (no eval_batch_size==1 call site ever passes a shared_cache, so this
        cache is always freshly cleared for this search() call anyway).

        mcts_core.pyd (currently built from cpp_mcts\\mcts_core_tree_reuse.cpp
        -- a stateful MCTSEngine with persistent-tree reuse, itself stacked
        on cpp_mcts\\mcts_core_pruned_deploy.cpp) hands the leaf board to
        Python as a (BOARD_SIZE, BOARD_SIZE) numpy int8 array (0=empty, 1=X, 2=O)
        instead of a nested list of "X"/"O"/" " strings, so this routes
        through board_to_input_array_from_numeric (integer comparisons)
        instead of board_to_input_tensor (numpy string-equality), and the
        cache key is board_arr.tobytes() instead of a tuple-of-tuples of
        Python strings -- much cheaper to build/hash for this hot per-leaf
        path (measured ~2.8x faster leaf-conversion in isolation)."""
```

**`shared/mcts_cpp.py`** MCTS._search_cpp (docstring)

```text
        """Delegates the whole search to the persistent C++ engine
        (self._cpp_engine, one MCTSEngine per MCTS object -- see __init__),
        then reconstructs a plain MCTSNode tree (root + direct children only
        -- the only depth every call site actually reads) so every caller
        of MCTS.search() keeps working unchanged.

        Calling search()/search_batched() on the SAME self._cpp_engine every
        ply (rather than a stateless module-level function) is what makes
        tree reuse work: the engine itself decides, from a board diff
        against what it last saw, whether to promote an already-explored
        subtree or start fresh -- transparent to this method and to every
        caller. See MCTSEngine's class docstring in
        cpp_mcts/mcts_core_tree_reuse.cpp for the full design."""
```

**`shared/mp_eval.py`** <module> (docstring)

```text
"""
Multiprocessing-based formal evaluation (evaluate_vs_deterministic) for the
11x11 hybrid MCTS pipeline.

Why this exists: evaluate_vs_deterministic's existing ThreadPoolExecutor +
PooledEvaluator implementation caps effective CPU usage at roughly 1 of 8
logical cores in production, because deterministic_play (the pure-Python
minimax opponent every game plays against) holds the GIL for the entirety of
its own call and therefore fully serializes across worker threads -- the C++
MCTS engine itself already releases the GIL correctly during its own tree
search (verified directly against the deployed mcts_core.pyd source), so that
part was never the bottleneck. Giving each concurrent game a real OS process
instead of a thread routes around the GIL entirely for the part that was
actually serializing. Measured on this hardware (GTX 1050, 8 logical cores):
~2.5x wall-clock speedup on a realistic vs-Deterministic game-generation
workload at the live sim budget, with bit-for-bit-identical search outputs
(deterministic/no-noise/greedy mode is a pure function of model weights +
opening + settings, so crossing a process boundary changes no numeric value).

Architecture: a worker-pool of up to os.cpu_count() processes each running
worker_loop, pulling (game_idx, mcts_plays_x, opening) jobs from a shared
queue until exhausted -- this avoids spawning more OS processes than there
are cores when num_games (e.g. eval_games=20) exceeds the core count. Every
worker's own C++ tree search (mcts_core.mcts_search_batched) and
board-to-plane conversion happen locally (CPU-bound, parallel across
processes, no CUDA). NN evaluation is routed to ONE centralized GPU-owning
evaluator process over a persistent shared-memory channel per worker
(see WorkerChannel/ChannelHandle), mirroring PooledEvaluator's own
accumulate-until-target_batch-or-max_wait batching loop but across a process
boundary instead of a thread boundary.

Safety: evaluate_vs_deterministic_mp raises on ANY setup failure or a worker
dying without completing its assigned games (rather than hanging or silently
returning wrong stats), so its caller (evaluate_vs_deterministic in
naps_vsdet.py) can fall back to the existing, long-proven threaded
implementation. On an unattended 20000-epoch job, a periodic eval call
failing over to the slower-but-solid path is fine; a hang is not.

Not yet ported: generate_mcts_self_play_data, generate_mcts_vs_deterministic_data,
generate_mcts_vs_league_data, and run_gating_check still use the threaded
implementation -- their game-generation loops need per-move policy targets
(training data) or a model-vs-model matchup (no GIL-heavy minimax opponent in
the loop, so the same speedup would not obviously apply), which this module
does not attempt. See the investigation report this was built from for the
full reasoning on why those were left out of this pass.
"""
```

**`shared/naps_gate.py`** run_gating_check (docstring)

```text
    """
    Plays the currently-training `model` against the tracked best checkpoint
    across all board-symmetry openings x both sides (fair games). If model's
    score (win=1, draw=0.5) reaches GATE_PROMOTION_THRESHOLD, model's weights
    become the new best. Bootstraps on first call (no best yet). `gen_model`
    (the frozen model self-play/vs-Det/vs-League data generation actually
    uses) is synced to the tracked best's weights on every promotion,
    including the bootstrap case, so generation always reflects the latest
    validated checkpoint.

    Games run concurrently across a ThreadPoolExecutor (same pattern as
    evaluate_vs_deterministic / the three generate_mcts_*_data functions).
    Two frozen models are in play here (`model`, `best_model`), unlike the
    single-model generation functions, so each gets its OWN PooledEvaluator
    and its OWN shared eval_cache -- a cache entry or pooled batch is only
    ever valid for the exact weights that produced it, so model's and
    best_model's calls are never pooled or cached together. Behavior-
    preserving for game OUTCOMES (see evaluate_vs_deterministic docstring
    for why); only wall-clock time changes.
    runner (2026-09-27): optional selfplay_pool task runner -- plays
    the 20 gate games in the persistent multi-process pool (the two models
    under separate eval tags/caches, same isolation as the two
    PooledEvaluators below); any failure falls back to the threaded path.
    Returns (win_rate, draw_rate, loss_rate, promoted).
    """
```

**`shared/naps_hard_examples.py`** get_fork_mistake_samples (docstring)

```text
    """Per-epoch DEDICATED draw from the Hard-7 (general, template-free
    fork) slice of the hard-mined pool, separate from
    get_hard_example_samples' whole-pool draw.

    2026-09-23: Hard-7 is currently a small fraction of the shared 8000-cap
    pool (261/8000 the first time it was measured), so
    get_hard_example_samples' n_positions=46 draw only pulls ~1-2 Hard-7
    examples per epoch by chance -- measured directly
    (hard7_adaptation_test.py): the live model still walked into 18/20
    freshly-mined Hard-7 positions one epoch after mining, essentially no
    real training exposure yet. This dedicated draw guarantees Hard-7 gets
    real per-epoch volume regardless of how large the rest of the pool
    (mostly Hard-5, outcome-based) grows around it. sample_weight=2.0
    (higher than the shared draw's 1.5) since Hard-7 carries a confident
    value=-1.0 label (see relabel_hard_examples), not a policy-only one.
    """
```

**`shared/naps_hard_examples.py`** comment at line 148

```text
# Adds a second, outcome/disagreement-based mining criterion on top of the
# pattern-only scan below (threat_gaps mirrors that scan's own "4-of-5+gap"
# test exactly, factored out so both share one implementation). Approved
# design (see this session's diagnosis workflow, agent "diag:hard-mining-spec"):
#   Hard-1: a winning move was available (own threat_gaps) and not played.
#   Hard-2: opponent had an immediate winning threat and it was not blocked.
#   Hard-3: a fork (fork_geometries) was available/threatened and not taken/denied.
#   Hard-5: position within the last HARD5_LOOKBACK of the mover's own plies
#           in a game that mover actually lost.
#   Hard-6 (an addition, not in the original spec): the search's own
#           visit distribution came out suspicion-of-degenerate flat for its
#           candidate count -- see entropy_complexity_probe.py's finding that
#           ~19% of the live buffer is EXACTLY uniform over its candidate set,
#           a PUCT round-robin degeneracy when prior+value are both flat.
# Relabeling deliberately does NOT imitate the opponent's move (that review
# rejected minimax-as-teacher: a shallow, fixed depth-2 opponent is most
# likely to be WRONG exactly where the model's own search disagrees with it
# most). Instead every flagged position is re-searched with the model's OWN
# MCTS at a boosted sim count and the resulting visit distribution becomes
# the policy target -- purely a bigger dose of the exact fix already applied
# this morning (more search), targeted at the positions where today's smaller
# production budget still isn't enough.
```

**`shared/naps_hard_examples.py`** comment at line 328

```text
        # 1.0 -> 0.95 (label smoothing). value_overfit_check.py
        # found the live value head badly overconfident on FRESH,
        # never-trained-on self-play games (held-out MSE 1.36 --
        # worse than a constant-0 predictor; 61% of predictions
        # saturated past |0.9|; the model's own "very confident"
        # bucket only actually won ~67% of the time, not ~100%).
        # Confident artificial labels (this class and Hard-7) are
        # the most likely amplifier -- softening them off the
        # tanh's saturated tail should reduce that overconfidence
        # without giving up the corrective signal's direction.
```

**`shared/naps_hard_examples.py`** comment at line 340

```text
        # Unlike Hard-1/5, this is a confident NEGATIVE
        # label, not a policy-only class -- letting `other` reach an
        # unstoppable fork is a near-certain loss for `player`
        # regardless of what happens afterward (session evidence:
        # test_fork_prevention.py found 0/3 real HLT forks
        # preventable even with a fresh 10000-sim search, so this
        # needs a direct value correction, not just more search at
        # relabel time). -1.0 -> -0.95: same label-smoothing
        # rationale as Hard-1 above.
        # REMOVED -- policy-only now (value=None). The
        # board stored here is the position BEFORE the defender
        # moves, where fork-safe replies exist, so "near-certain
        # loss" was the value of the position AFTER a bad reply, not
        # this one. pessimism_probe (14 Hard-7 positions, safe reply
        # then best-vs-best playouts) measured raw value -0.78 vs an
        # empirical outcome of -0.04: this label was teaching the
        # value head that any fork threat means the game is lost.
```

**`shared/naps_league.py`** pick_league_opponent (docstring)

```text
    """Picks a frozen league opponent for this epoch's vs-League games.

    With probability best_checkpoint_prob (checked first), loads the
    tracked BEST_CHECKPOINT_PATH -- the strongest validated checkpoint,
    always kept up to date by run_gating_check's promotion logic -- instead
    of sampling from the snapshot pool at all. This guarantees a real,
    strong sparring partner shows up in league play regardless of how the
    plain recency-weighted draw below happens to fall.

    Otherwise (or if that file doesn't exist), with probability
    deep_history_prob (once epoch >= deep_history_min_epoch) samples
    UNIFORMLY from the FULL snapshot history instead of just the most
    recent pool_size -- the recency-weighted pool below permanently
    excludes a snapshot once pool_size*10 epochs have passed it, so without
    this the model would only ever face styles from its own recent past,
    never its much-earlier (structurally weaker/more exploratory) self.
    Kept low-probability and floored at deep_history_min_epoch so it stays
    a small, occasional addition, not a replacement for the recency-weighted
    pool that already does most of the work.

    Otherwise, falls back to the existing snapshot pool: a random past
    checkpoint snapshot from the most recent pool_size snapshots BY EPOCH
    NUMBER, not filesystem mtime -- mtime can get scrambled by anything
    that touches the files without touching training progress, e.g. a
    backup pass or several snapshots written in quick succession right
    around a restart. Sampling within that pool is recency-weighted
    (linearly favoring the newest snapshot), not uniform, since the newest
    snapshots are the most representative of the model's current style --
    the whole point of league play is testing against strategies close to
    what self-play might otherwise overfit to.

    Returns (league_model, path), or None if the best-checkpoint load
    failed, or if fewer than 2 snapshots exist yet -- too early in training
    for league play to add anything over the fixed deterministic opponent
    already in use."""
```

**`shared/naps_league.py`** play_one_mcts_vs_league_game (docstring)

```text
    """
    Like play_one_mcts_vs_deterministic_game, but the opponent is a frozen
    past checkpoint of this same model instead of deterministic_play. Both
    sides' moves are recorded (the league opponent runs a real, always-full
    search -- see league_sims below -- so its visit distribution is just as
    valid a policy target as the live model's own): the live model was
    previously only ever training on its own moves against the league pool,
    never on the moves of the checkpoints that actually beat it, which
    starved league-derived data of the one thing it could uniquely teach.

    opening (see play_one_mcts_vs_deterministic_game): pre-places X's first
    move there before normal play resumes. The league side's own moves use
    add_noise=False (greedy), so when it moves first, every game with the
    same side assignment would otherwise replay the identical opening move.

    league_pooled_evaluator/league_eval_cache (2026-09-26, unified MP
    game pool): route the LEAGUE side's leaf evaluations through a separate
    evaluator/cache -- a worker process holds no weights, so it needs a
    proxy that the GPU coordinator maps to the league model. Both default
    to None, which keeps the original behavior (league side calls
    league_model directly with a private per-search cache).

    Returns (history, outcome, mcts_result): mcts_result is
    'mcts'/'league'/'draw' from the live model's perspective.
    """
```

**`shared/naps_openings.py`** sample_tactical_start (docstring)

```text
    """
    Builds one REACHABLE board matching a random tactical curriculum pattern
    (same "3-in-a-row + 1 gap" shapes as generate_tactical_examples), for
    self-play to occasionally start a game from instead of always from an
    empty board -- so self-play actually encounters this tactical family
    (and must resolve it with its own search) more often than it arises
    naturally from full-game rollouts. Unlike tactical_clutter, clutter
    here is piece-count constrained: exactly 2 stones of the color NOT used
    to fill the window, so count(X) == count(O) after placement -- the
    parity a real alternating game reaching this position (with 'X' to
    move next) requires. Mirrors _get_tactical_tensors' own convention
    (board_to_input_tensor(board, 'X', ...)): 'X' is always the mover,
    whether about to block an 'O' open line or complete its own.

    This changes what positions self-play explores FROM, not what move is
    labeled correct -- the move actually played is still whatever this
    epoch's MCTS search (with its own noise/exploration) decides, same as
    any other self-play game.

    Returns (board, 'X'), or None if no valid placement is found within
    max_attempts (caller should fall back to a normal empty-board start).
    """
```

**`shared/naps_selfplay.py`** play_one_mcts_game (docstring)

```text
    """
    Plays a single MCTS self-play game to completion and returns
    (history, outcome), where history is a list of
    (board_3d, player_turn, target_policy_map) tuples, pre-augmentation.
    With return_ply_info=True it returns (history, outcome, ply_indices,
    total_plies) instead: ply_indices[k] is the ply number (0-based, counted
    from this game's start position) at which history[k] was recorded, and
    total_plies is how many plies the game actually lasted. Needed for the
    V3 survival target once Playout Cap Randomization skips recording
    fast-search plies -- len(history) - k then undercounts the true
    plies-to-end.

    Builds its own MCTS() instance, but eval_cache (when passed) is shared
    across every game and move in the calling epoch -- safe to call
    concurrently from multiple worker threads because it's read-mostly
    (dict get/set is GIL-atomic) and a pure function of the frozen weights
    for the whole generation phase.

    start_board/start_player (both or neither): start from this board
    instead of empty -- see sample_tactical_start, used to occasionally
    start self-play from a reachable "hard" tactical position instead of
    always from scratch, so self-play encounters that family of positions
    (and must resolve it with its own search) more often than it would
    arise naturally from full-game rollouts.

    full_search_prob/fast_sims: Playout Cap Randomization (KataGo). At each
    move, with probability full_search_prob run the normal (possibly
    boosted) search and record it as a training target; otherwise run a
    cheap fast_sims-simulation search just to pick a move and advance the
    game, without recording it (a low-sim search's visit distribution is
    too noisy to be a good policy target). Same total game count spends
    less average compute per move, so the freed budget can go to more
    games/more full searches elsewhere. full_search_prob=1.0 (default)
    disables this -- every move is a full search and recorded, matching
    prior behavior exactly.

    2026-09-26 PCR fixes (no effect at full_search_prob=1.0):
      * temperature sampling is keyed on the number of plies actually played
        (`ply`), not len(history) -- with PCR, len(history) only counts the
        recorded (full-search) plies, which would have stretched the
        temperature_moves exploratory window over ~1/full_search_prob times
        as many real moves.
      * fast searches run without Dirichlet noise (KataGo: fast searches
        exist to play strong moves cheaply, not to explore -- they are never
        recorded as training targets).
    """
```

**`shared/naps_tactical_refresh.py`** refresh_mined_tactical_pool (docstring)

```text
    """
    Periodic top-up for the real-game-mined tactical pool (see
    _get_mined_tactical_tensors), called once per tactical_refresh cycle
    (right before tactical_refresh itself) so the pool keeps tracking the
    model's CURRENT play style instead of staying frozen at whatever epoch
    it was first mined at.

    Why this matters: the mined pool's board positions come from games the
    model itself plays, so a one-time-only pool reflects only that
    snapshot's blind spots (a self-referential limitation -- see project
    discussion). As self-play's own exploration (Dirichlet noise + MCTS's
    PUCT exploration bonus, which visits under-prior-weighted moves and
    feeds back into training when they turn out good) gradually corrects
    those blind spots over many epochs, re-mining periodically lets the
    tactical curriculum's data source ride along with that improvement
    instead of being stuck at its original snapshot forever.

    Plays num_selfplay_games + num_vsdet_games real games with the CURRENT
    model (threaded, mirroring generate_mcts_self_play_data's own
    ThreadPoolExecutor pattern -- torch releases the GIL during the forward
    pass so this is real concurrency), scans every ply of every game for
    the same "4-of-5+gap" pattern tactical_windows() defines, appends any
    matches to the on-disk pool at MINED_TACTICAL_EXAMPLES_PATH (capped at
    `cap` total examples -- oldest examples dropped first once exceeded, so
    the pool doesn't grow unboundedly over a 20000-epoch run), and
    invalidates the in-memory tensor cache so the very next
    _get_tactical_tensors() call (from the tactical_refresh() call right
    after this one) picks up the fresh pool instead of stale cached tensors.

    Uses a modest sims count (150, not the live training sim-ramp budget)
    since realistic stone clustering matters more here than search
    strength -- matches the original one-time mining's own choice (see
    scratch/multiproc_investigation/mine_tactical_examples.py).

    2026-09-11: the same games also feed a SECOND, outcome-based mining pass
    (classify_hard_examples / relabel_hard_examples) writing to the
    separate HARD_EXAMPLE_POOL_PATH -- see the block of comments/functions
    just above this one for the full design. Kept in the same function
    (rather than a second refresh call) so the expensive game-generation
    itself is only paid once per tactical_refresh cycle.

    runner (2026-09-27): optional selfplay_pool task runner. When
    given, the mining games (+ their per-game hard-example classification)
    and the relabel searches run in the persistent multi-process game pool
    instead of threads in this process; any runner failure falls back to
    the original threaded/sequential code below for that step.
    """
```

**`shared/naps_tactical_refresh.py`** get_open_three_injection_samples (docstring)

```text
    """Per-epoch draw of open-three + open-3-3-fork (+ diagnosed-position)
    examples for the regular replay buffer, same pattern as
    get_hard_example_samples right below this.

    2026-09-22: replaces tactical_refresh()'s periodic burst supervised
    fine-tune as this curriculum's delivery mechanism. Isolated before/after
    testing (scratch/.../tactical_refresh_destroy_test.py) found ONE
    tactical_refresh() call immediately dropped vs-Det win rate 65%->25%
    (a single 20-game/600-sim sample -- noisy, but the direction matches the
    loss-spike pattern already documented in tactical_refresh's own
    docstring at ep22/ep43/ep64). A separate retention check
    (scratch/.../tactical_retention_check.py) found the ~100% tactical
    accuracy right after a refresh had already decayed to ~70% just 2
    epochs later, before the next refresh even fired -- so each burst was
    paying a real, measured cost for a gain that mostly didn't survive to
    the next cycle anyway. Blending a smaller amount of this curriculum into
    EVERY epoch's ordinary training batch (like the hard-mined pool
    already does) avoids both problems: no isolated-batch overfit spike,
    and no burst to decay away between refreshes since there's no longer a
    gap between applications.

    n_examples=120 positions x up to 8-way dihedral augmentation ~= up to
    ~960 samples/epoch -- deliberately a larger share of the epoch's buffer
    than the hard-mined pool's ~368/epoch (n_positions=46), since this is
    now the sole delivery path for the open-three curriculum rather than a
    supplement to periodic bursts.
    """
```

**`shared/naps_tactical_refresh.py`** _get_tactical_tensors (docstring)

```text
    """
    Builds this refresh's tactical training pool as an EQUAL-WEIGHTED mix:
    ALL currently-available real mined examples (naturally-clustered
    positions mined from actual played games) combined with a FRESH random
    subsample of the synthetic pool of equal size (drawn without
    replacement, independently on every call -- not cached/fixed). This is
    "candidate d" from the held-out generalization study (Phase 2:
    scratch/multiproc_investigation/tactical_mix_experiment.py; Phase 3
    independent re-verification: scratch/multiproc_investigation/
    phase3_independent_reverify.py) -- the only mix among 5 tested that
    materially improved held-out-REAL argmax accuracy (Phase 2: 0.5079 ->
    0.6947; independent re-verification with a different seed/split:
    0.3135 -> 0.6239) with no measurable cost to held-out-SYNTHETIC
    accuracy (both runs: within ~0.2pp of the synthetic-only baseline).
    Naive full-concatenation and real-upsampled-to-synthetic-volume mixes
    were both measurably WORSE than synthetic-only on held-out real
    accuracy in Phase 2 -- do not "simplify" this back to a plain
    concatenation.

    Re-drawing the synthetic subsample fresh every call (rather than fixing
    it once) means the full synthetic pool's window/direction/color
    coverage guarantee is still achieved in expectation across many refresh
    cycles (every tactical_refresh_interval epochs), while any single
    refresh's pool stays balanced 50/50 instead of being diluted the way a
    naive full-concatenation was measured to be.
    """
```

**`shared/naps_tactical_refresh.py`** _multi_label_targets_from_rows (docstring)

```text
    """2026-09-14 multi-label fix. rows_by_key: dict mapping a board(+player)
    key -> list of ((gap_r, gap_c), mover_completes) tuples all scanned from
    that EXACT SAME input position (it commonly matches more than one
    tactical window at once, e.g. two simultaneous open threats). The old
    scheme emitted one single-hot row per matching window, so the identical
    network input got DIFFERENT "correct" one-hot targets across rows --
    an argmax target that no single-output model can ever satisfy for all
    of them. Measured on the live mined pool (2026-09-14,
    scratch/multiproc_investigation/mine_tactical_conflict_check.py): 27.1%
    of unique boards carried >=2 conflicting single-cell labels, capping the
    theoretical MAX achievable argmax accuracy on the raw pool at 64.6% --
    this is very likely the dominant reason tactical_refresh's own reported
    accuracy has been flat at ~80% for weeks (98% synthetic + ~65% real,
    averaged in the deployed 50/50 mix, ~= 80%).

    Fix: one row per unique board(+player). If ANY matching window for that
    board is a provable mover_completes=True win, the target concentrates
    ONLY on those winning cells (playing a non-winning move when a win is
    available is never correct, so diluting the target with defensive
    cells would be actively wrong). Otherwise the target spreads uniformly
    over every distinct labeled (defensive) gap cell -- all of them are
    legitimately urgent, and this removes the artificial ceiling without
    claiming false certainty about which single one is "the" answer.
    Yields (key, target_vec[NUM_CELLS] summing to 1, has_winning_cell)."""
```

**`shared/naps_tactical_refresh.py`** tactical_refresh (docstring)

```text
    """
    Runs a few epochs of supervised fine-tuning on the tactical curriculum.

    Uses its OWN local Adam (2026-09-10) rather than the main RL loop's
    optimizer. Sharing it meant 8 supervised epochs on a different objective
    wrote into the shared Adam moment estimates, and every refresh was
    followed by a 2-3 epoch loss spike as the main loop recovered (ep22:
    P 0.98->2.04, V 0.06->0.26; ep43 and ep64 the same shape) -- roughly 14%
    of all epochs spent undoing it. Meant to be called
    periodically (every N epochs) during train_mcts_alpha_zero so this
    signal doesn't get diluted back out by the much larger volume of
    ordinary self-play samples between refreshes.

    Policy loss is computed over the whole batch, as before. Value loss is
    added ONLY over the subset of the batch carrying a real value label --
    a per-row float value_target (NaN = no label, e.g. the pure-block-only
    case, which gets none: "the defender played correctly" alone doesn't
    prove a value by itself). 0.97 (not literal 1.0) stays off the tanh
    output's saturated-gradient tail.
    """
```

**`shared/naps_tactical_refresh.py`** _get_mined_tactical_tensors (docstring)

```text
    """Loads and tensorizes the real mined examples (board, current_player,
    (gap_r, gap_c), mover_completes) tuples once and caches. Unlike the
    synthetic boards (always encoded from 'X' mover's perspective), each
    mined example carries its own real current_player -- passed through
    as-is to board_to_input_tensor, whose perspective-relative encoding
    (curr_plane = current_player's stones) needs no color canonicalization.

    2026-09-09: an 8-way dihedral-augmented version of this (same
    augment_sample every other data source gets) was tried and reverted --
    it reproducibly stalled the live process for 2+ hours at the third
    tactical_refresh cycle (ep63; the first two, ep21/ep42, ran fine), most
    likely GPU-memory fragmentation from repeatedly building/discarding an
    8x-larger (150K+) tensor cache every refresh compounding across cycles.
    Not re-attempted here; a safer version would need to build the
    per-example encoding via board_to_input_array (pure CPU numpy, no
    per-item .to(device) round-trip) instead of board_to_input_tensor, and
    ideally batch rather than rebuild-from-scratch every refresh.

    Returns empty tensors (falls back to synthetic-only) if the pickle is
    missing, so a fresh checkout without the mined-examples file still runs."""
```

**`shared/naps_tactical_refresh.py`** comment at line 228

```text
        # generate_fork_examples() (the open-3-3 / double-open-
        # three geometry -- see fork_geometries) added to this SAME
        # continuous pool. It was already implemented and already used to
        # get delivered via tactical_refresh()'s burst fine-tune, but that
        # call was removed from the main loop this session (see the
        # tactical_refresh_interval guard's comment) and nothing replaced
        # it for this specific curriculum the way open-three's own got
        # migrated here -- so it silently went from
        # "burst-delivered" to "never delivered". A held-out T-shape probe
        # this session (test_t_shape_33fork_realistic.py) found this
        # measurably regressing over the ~200 epochs since: raw policy
        # rank for the correct denial cell 3rd/121 at the Gen12 reset
        # point -> 47th/121 at the current live checkpoint, MCTS solving
        # it correctly at epoch 0 and failing every checkpoint since --
        # i.e. real forgetting of knowledge the supervised-pretrain
        # checkpoint already had, not just slow acquisition. No new
        # examples are being authored here -- this is reconnecting an
        # existing generator to an existing delivery mechanism.
```

**`shared/naps_tactics.py`** generate_double_four_examples (docstring)

```text
    """
    OFFENSE: the mover ('X') owns all 6 stones (3 per direction); playing
    the shared center cell completes a four in both directions at once --
    each threatens a win next move via its own open end, and the defender
    can only block one, a strictly more decisive threat than the open-three
    fork (generate_fork_examples) even though, like that fork, it is one
    move short of an outright win itself. DEFENSE: 'O' owns all 6 stones
    (would create the double-four if 'O' got to move next); the mover ('X')
    must occupy the shared center FIRST to deny it -- same target cell
    either way. mover_completes is always False for the same reason
    generate_fork_examples' is: completing/denying this shape isn't itself
    a provable 5-in-a-row.

    Each geometry gets a clean variant plus a couple of clutter variants
    (reusing tactical_clutter, same pattern as generate_fork_examples),
    excluding both the stone cells and the required-open end cells so
    clutter can't collide with either.
    """
```

**`shared/naps_tactics.py`** generate_open_three_examples (docstring)

```text
    """
    Reuses _closed_three_spans()'s exact 5-cell (k=-1..3) windows, but
    treats BOTH end cells (index 0 and index 4) as empty and equally
    correct, instead of one being an occupied blocker and the other the
    sole target -- so each variant yields TWO rows for the SAME board (one
    per valid end), which _multi_label_targets_from_rows groups back into a
    single 50/50 target over both cells, exactly like the two symmetric
    block cells in generate_tactical_examples' gap family.

    OFFENSE: mover ('X') owns the three-stone line; either end extends it
    into an open four. DEFENSE: 'O' owns the three-stone line; the mover
    ('X') must occupy either end to prevent the open four. Neither is a
    provable 5-in-a-row on its own, so mover_completes is always False --
    same policy-only convention as generate_closed_three_examples/
    generate_fork_examples/generate_double_four_examples.
    """
```

**`shared/naps_tactics.py`** comment at line 358

```text
# Position A (epoch291 HLT loss) is a real open-three-defense
# instance whose raw POLICY prior (not just the value head/MCTS search) has
# drifted onto the historically-wrong move after repeated self-play/HLT
# reinforcement -- confirmed by direct re-check against the live checkpoint
# (rank 1/121, 12.6% prior on the wrong cell, vs the two correct blocks at
# 3-5% each). A diffuse generic curriculum (generate_open_three_examples
# above) teaches the general shape but has no special weight on this EXACT,
# already-entrenched association -- unlearning it needs direct, repeated
# counter-supervision on the specific pattern, not just broader coverage.
# Position B (live test game, Claude vs the epoch462-promoted
# checkpoint) -- a DIAGONAL open-three missed in a denser, real-game-like
# position (5 X + 4 O stones already on the board, unlike Position A's
# emptier board) despite the model correctly blocking a near-identical
# horizontal open-three earlier in the SAME game. Confirms the open-three
# fix from generate_open_three_examples works on the shape in isolation but
# does not yet fully generalize to busier boards -- clutter density was
# widened 3-36 (was 3-6, see tactical_clutter calls throughout this file)
# specifically to close this gap generically, and this exact position is
# also hard-mined directly, same rationale as Position A.
```

**`shared/naps_tactics.py`** comment at line 58

```text
# Two more tactical blind spots, diagnosed separately (see
# scratch/multiproc_investigation/closed_three_diagnostic_newrun.py and
# double_three_fork_diagnostic.py) as weak even on the latest checkpoint,
# and NOT covered by the 4-of-5-gap family above (that family is always 4
# stones + 1 gap; both shapes here are one filled square short of it):
#   - closed-three: 3-in-a-row blocked on one end, open (with room to
#     extend) on the other. Offense = extend into the open end. Defense =
#     block the open end first.
#   - double-three-fork: a single point that completes an open-three in
#     two directions at once (unstoppable -- the defender can only block
#     one line). Offense = play the shared point to create the fork.
#     Defense = play it first to deny the opponent's fork.
# Unlike the 4-of-5-gap family, NEITHER pattern's correct move is ever a
# provable win on its own (extending/blocking a closed-three, or
# creating/denying a fork, doesn't complete a 5-in-a-row) -- so both
# offense and defense are policy-only here, mover_completes always False.
# This keeps the two new curricula symmetric by construction, unlike the
# existing family's offense/defense value-supervision asymmetry.
```

**`shared/naps_tactics.py`** comment at line 322

```text
            # 3->8 variants (1 clean + more clutter diversity) --
            # open-three defense is the pattern the passive open3 tracker
            # has shown persistently flat/noisy (0.2-0.7 miss rate) despite
            # training, so this deliberately over-represents it relative to
            # the other four tactical families in the combined curriculum.
            # 8->14 -- tactical_refresh() (the periodic burst
            # supervised re-injection this curriculum used to feed) was
            # removed the same day: isolated testing
            # (scratch/.../tactical_refresh_destroy_test.py) measured ONE
            # refresh call immediately dropping vs-Det win rate 65%->25%,
            # and a separate retention check
            # (scratch/.../tactical_retention_check.py) found that gain
            # decayed from ~100% back to ~70% tactical accuracy within just
            # 2 epochs anyway -- net negative. Replaced by continuous
            # per-epoch injection into the ordinary replay buffer (see
            # get_open_three_injection_samples), so this pool now needs
            # more inherent variant diversity since it's no longer refreshed
            # by a dedicated fine-tuning pass, just sampled from directly.
```

**`shared/naps_tactics.py`** comment at line 274

```text
# (added after re-diagnosing the HLT-loss value/MCTS blind spot --
# see scratch/multiproc_investigation/rediag_value_blindspot.py, run against
# the live epoch443 checkpoint): a genuine OPEN three (3-in-a-row with BOTH
# immediately-flanking cells empty) is a shape this curriculum never actually
# covered. generate_tactical_examples' "4-of-5-gap" family is always one move
# from a proven outcome (mover_completes or a single forced block);
# generate_closed_three_examples' shape has only ONE open end. An open three
# has TWO equally-correct response cells (either flank creates/denies an open
# four), and re-testing 6 real/synthetic open-three positions against the
# live checkpoint found the model still gets roughly half of them wrong --
# MCTS converging on an unrelated move despite the raw policy prior usually
# still ranking at least one true response cell near the top -- the same
# "good-ish prior, MCTS/value diverges" pattern diagnosed earlier, just never
# targeted by a dedicated curriculum shape until now.
```

**`shared/naps_threats.py`** fork_safe_cells (docstring)

```text
    """Every empty cell where `player` moving there does NOT hand `other` an
    unstoppable next-move fork (winning outright always counts as safe too)
    -- the Hard-7 analogue of threat_gaps' g_p/g_o: a verified, brute-force
    ground truth for "which replies are actually correct here", not just
    "is a fork possible". Used by relabel_hard_examples to force the
    Hard-7 policy target onto a real safe cell when the re-searched MCTS
    visit distribution doesn't already favor one -- mirrors how hard_class
    1/2 force their policy onto g_p/g_o. Without this, Hard-7's policy
    target was just whatever a RELABEL_SIMS-budget search happened to
    converge to, which an artificial verified-answer puzzle test
    (2026-09-24 session) showed can itself fail to concentrate visits on
    the best-Q move -- i.e. the relabel step could silently inherit and
    reinforce the same policy-prior blind spot it's meant to correct.
    O(BOARD_SIZE^2) candidate moves x opponent_can_fork_next's own
    O(BOARD_SIZE^2)-worst-case scan; only run inside
    relabel_hard_examples's already-capped (RELABEL_CAP) pass, not the
    self-play hot loop."""
```

**`shared/naps_threats.py`** _immediate_win_cells (docstring)

```text
    """Every empty cell where placing `color` right now completes 5-in-a-row.
    Template-free (unlike fork_hits/fork_geometries, which only match the
    clean symmetric double-open-three shape) -- brute-forces every empty
    cell through fast_check_win, so it also catches irregular, real-game
    multi-threat shapes fork_geometries structurally can't represent.
    len(...) >= 2 means an UNSTOPPABLE fork for whoever must move next (no
    single move blocks two simultaneous winning replies).

    2026-09-23: added after 4 straight HLT (human) losses this session, all
    ending in this exact kind of fork, NONE of which were caught by the
    existing (narrow, template-based) Hard-3 detector below -- see the
    session's find_fork_points.py/test_fork_prevention.py scratch analysis.
    O(BOARD_SIZE^2) fast_check_win calls per invocation; only run on
    already-mined game plies (a few hundred per refresh cycle), not inside
    the hot self-play loop itself.
    """
```

**`shared/naps_trainer.py`** comment at line 203

```text
    # Pools eval_batch_fn calls from concurrent self-play worker threads into
    # fewer, larger GPU forward passes -- see PooledEvaluator docstring.
    # Constructed once (not per-epoch): holds a reference to `gen_model`, not
    # a weight snapshot, so it keeps working correctly as run_gating_check's
    # promotions mutate those same weights in place across epochs.
    # target_batch=24, max_wait=0.05 (was 48/0.005): live stats
    # logging showed this PooledEvaluator hitting its target batch 0% of the
    # time across 300K+ flushes (avg batch ~4, timing out every single time)
    # -- PooledEvaluator._run's `timeout = self.max_wait if pending else None`
    # re-arms a fresh wait window on every arrival (a silence-gap debounce,
    # not a deadline from the first pending item), and with 10 CPU-bound
    # self-play worker threads the gap between arrivals almost never stays
    # under the old 5ms window. Benchmarked on this exact self-play path
    # (scratch/multiproc_investigation/bench_pooled_evaluator.py, 3-run
    # average): target_batch=24/max_wait=0.015 grows the achieved batch
    # ~3.8->5.0 (+32%) and self-play games/sec +19% over the old operating
    # point; CPU-bound MCTS tree search, not GPU eval, still dominates epoch
    # wall-clock, so this is a real but modest gain, not a step-change. Kept
    # consistent with evaluate_vs_deterministic / run_gating_check's own
    # scoped PooledEvaluators below, which use the same values.
```

**`shared/naps_trainer.py`** comment at line 565

```text
            # Prioritized replay (Schaul et al. 2015), adapted to this
            # project's full-buffer-pass-per-epoch training loop instead
            # of a sampled minibatch replay: reweight each batch's loss
            # by its own current policy/value-prediction "surprise"
            # (computed fresh from this same forward pass, no stored/stale
            # priority table needed) so samples the model is most surprised
            # by contribute more gradient. b_weight is each sample's static,
            # source-based weight (self-play's mover-won bonus, fixed
            # vs-Det/vs-League weights -- see the raw_samples.append call
            # sites) and is always applied; the surprise term on top of it
            # only kicks in when use_prioritized_replay is True. NOTE: when
            # prioritized, weights correlate with p_loss_per_sample/
            # v_loss_per_sample by construction (weight = f(loss)), so the
            # weighted mean is *not* comparable to the plain mean -- it's
            # mechanically inflated (Cov(weight, loss) > 0). That's
            # expected/intended (harder samples deliberately dominate the
            # gradient), not a regression -- hence tracking the unweighted
            # loss above separately for logging.
```

**`shared/naps_trainer.py`** comment at line 109

```text
    # lr-fix restart: base_lr 0.001->0.0001 and T_0 450->150.
    # Weight-drift measurement (see conversation/workflow findings) showed
    # lr=0.001 drifts the post-supervised-pretrain checkpoint 1.86x farther
    # (matched-epoch comparison) than lr=0.0001, and drift never reverses
    # even during low-instantaneous-LR windows of the T_0=450 schedule --
    # consistent with the gate-check having stayed stuck at ~0% for 564+
    # epochs after the Gen10 supervised-pretrain restart. The prior Gen7->
    # Gen8 supervised-pretrain restart broke through gate-check right around
    # the point its T_0=150 schedule decayed to ~1.2e-4 -- this restart
    # starts there directly instead of spending hundreds of epochs decaying
    # down to it.
    # Gate stuck 15 checks (650-720) while losses kept falling ->
    # halve the LR band (1e-4..5e-5 -> 5e-5..1e-5) to consolidate instead of drifting.
    # ep937 cooldown ended -> LR jumped 1e-5 -> 4.4e-5 and ep940 fell to ~1,092 Rapfi nodes.
    # Cosine peak 5e-5 -> 2e-5 (band 2e-5..1e-5) to polish rather than shake before the festival.
    # LR band per run (run_config.py): 11x11 NAPS 2e-5..1e-5, 11x11_v2 5e-5..1e-5, 15x15 1e-4..2e-5
    # (the 2e-5 polish band of the mature 11x11 net kept the young 15x15 net flat).
```

**`shared/naps_vsdet.py`** play_one_mcts_vs_deterministic_game (docstring)

```text
    """
    Plays one training game between the in-memory model (MCTS with noise
    and temperature sampling -- same exploration settings as self-play) and
    deterministic_play's minimax AI. Only the MCTS side's decisions are
    recorded, in the same (board_3d, player_turn, target_policy_map) format
    play_one_mcts_game produces, so callers can mix both into one buffer.

    Pure self-play never has to defend against or attack the specific
    tactics deterministic_play actually uses, so the model can stay blind to
    patterns that opponent exploits no matter how long training runs.
    Mixing in games against it directly closes that loop.

    deterministic_play is deterministic, so without `opening` every game
    with the same mcts_plays_x replays the same line -- see
    _play_vs_deterministic's docstring. Passing a (row, col) `opening`
    pre-places X's first move there before normal play resumes.

    det_max_depth (see sample_vs_det_max_depth) sets the deterministic
    opponent's minimax search depth for this game; 0 means it plays a
    uniformly random legal move instead of calling minimax at all.

    Returns (history, outcome, mcts_result): outcome uses the same
    X=1/O=-1/draw=0 convention as self-play; mcts_result is
    'mcts'/'det'/'draw' from the trained model's perspective.
    """
```

**`shared/naps_vsdet.py`** evaluate_vs_deterministic (docstring)

```text
    """
    Plays num_games against deterministic_play (minimax/alpha-beta), alternating
    who moves first, and returns (win_rate, draw_rate, loss_rate) from the
    trained model's perspective. This is the direct metric for the research
    goal ("beat the deterministic algorithm").

    Tries the multiprocessing implementation (mp_eval.evaluate_vs_deterministic_mp)
    first -- each game gets a real OS process instead of a thread, which
    matters because deterministic_play (the pure-Python minimax opponent)
    holds the GIL for its whole call and was found to fully serialize the
    threaded implementation down to ~1 CPU core regardless of thread count
    (the C++ MCTS engine itself already releases the GIL correctly and was
    not the bottleneck). Verified against the threaded path at the real
    production shape (20 games, 600/1350 sims, live checkpoint): identical
    win/draw/loss rates (deterministic/no-noise/greedy + fixed openings is a
    pure function of model weights, so crossing a process boundary changes
    no numeric value), 2.35x wall-clock speedup. Falls back to the original
    threaded implementation on ANY exception from the mp path (process spawn
    failure, a worker dying mid-game, GPU OOM in the evaluator process,
    etc.) so a transient multiprocessing hiccup degrades this call to
    "slower" rather than crashing or hanging an unattended training run.

    pooled_evaluator is only used by the threaded fallback path; the mp path
    has no equivalent parameter (see _evaluate_vs_deterministic_threaded).
    """
```

**`shared/naps_vsdet.py`** _evaluate_vs_deterministic_threaded (docstring)

```text
    """
    Original ThreadPoolExecutor + PooledEvaluator implementation, kept as the
    safety-net fallback for evaluate_vs_deterministic (see its docstring).

    Games run concurrently across a ThreadPoolExecutor with a shared
    eval_cache and PooledEvaluator, the same pattern
    generate_mcts_vs_deterministic_data already uses (see its docstring) --
    this is behavior-preserving for game OUTCOMES: no-noise/greedy MCTS vs
    deterministic minimax is a pure function of (model weights, opening,
    settings), and eval_cache only memoizes that pure function (a cache hit
    returns the exact same value a fresh compute would have) -- only
    wall-clock time changes.

    pooled_evaluator: reuse a caller-owned PooledEvaluator wrapping this
    exact `model` object if the caller already has one running; when
    omitted (the default), a fresh one scoped to just this call is
    constructed and torn down before returning, so this function stays
    usable standalone. NOTE: the training loop's own pooled_evaluator is
    scoped to gen_model, not model, so the existing call site cannot reuse
    it.
    """
```

**`shared/nets.py`** comment at line 129

```text
# Experimental fused-block architecture, built alongside (not
# replacing) alphago_lee.py's V1 network (GomokuNetV1). Imports everything except
# the model classes from board_encoding -- fork-mask helpers, board_to_input_array,
# board_to_input_tensor, BOARD_SIZE/NUM_CELLS/NUM_INPUT_CHANNELS -- so there is
# exactly one source of truth for the board/tensor representation; only the
# network internals differ here.
#
# Combines three changes discussed in chat, each independently motivated:
# 1. Depthwise-separable convolutions (MobileNetV3/EfficientNet-style) instead
#    of full 3x3 convs -- ~8-9x fewer multiply-adds per conv layer at this
#    channel count (256), directly targeting this project's recurring GPU
#    bottleneck.
# 2. GroupNorm instead of BatchNorm -- GroupNorm normalizes per-sample (no
#    batch-wide running statistics), so it behaves identically regardless of
#    batch size or train()/eval() mode. This eliminates an entire bug class
#    this project has hit more than once (the 5x5 gate-check BatchNorm
#    train-mode bug and others referenced in this session) at
#    the source, rather than continuing to rely on callers remembering to
#    call .eval() before every MCTS leaf batch.
# 3. Squeeze-and-Excitation channel attention (KataGo uses this in its own
#    trunk) -- lets the network dynamically emphasize/suppress channels based
#    on GLOBAL board context, which suits whole-board pattern reasoning
#    (open-three / fork detection) better than treating every channel with
#    fixed importance regardless of position.
#
# This combination (depthwise-separable conv + SE, "MBConv"-style) is not
# novel -- it is the standard building block of MobileNetV3 and EfficientNet,
# chosen here because it is well-validated in the literature, not because it
# is untested.
#
# IMPORTANT: this is architecturally incompatible with existing checkpoints
# (nets.GomokuNetV1's ResBlock uses full Conv2d + BatchNorm2d
# with different parameter shapes) -- a model built from this file must be
# trained from scratch, it cannot load model/best_mcts_11x11.pth or any
# existing checkpoint. Nothing under 11x11_naps/model/ is touched by this file
# merely existing; it only takes effect wherever something explicitly
# imports GomokuNetV2 instead of the original.
```

**`shared/nets.py`** comment at line 321

```text
# V3 = V2 (GroupNorm + SE, nets.py) + an auxiliary
# "survival" head that predicts how many plies remain until the game ends.
#
# Why: in 11x11 free-style gomoku the first player wins almost every game
# (elo_roundrobin.py: 68/72), so nearly every defender-side training
# position carries the same -1 value label no matter how well it was
# defended -- the value head has no reason to tell a good defence from a
# bad one, and defence quality measurably fell during self-play
# (elo_defense.py: the supervised start model survived 21.5 plies as the
# defender, the self-play models 11-14). Predicting remaining plies gives
# the shared trunk a dense per-position signal that does distinguish
# "losing slowly" from "losing fast" (KataGo-style auxiliary target).
#
# Compatibility:
# - forward(x) still returns (policy, value) exactly like V2, so every
#   existing call site (MCTS eval callbacks, gating, league, self-play pool,
#   test scripts) works unchanged. forward_all(x) additionally returns the
#   survival prediction and is used only by the training step.
# - load_state_dict() accepts V2 checkpoints: the survival_head.* keys are
#   allowed to be missing (the head keeps its fresh init); any OTHER
#   missing or unexpected key still raises, so a genuinely wrong checkpoint
#   is not silently accepted.
# - V3 checkpoints carry extra survival_head.* keys, so they do NOT load
#   into the V2 class with strict=True. The last V2 best is kept at
#   model/v2_final_best_ep284.pth for old-vs-new matches.
#
# Survival target: min(remaining_plies, SURVIVAL_CAP) / SURVIVAL_CAP, in
# [0, 1]; the head ends in a sigmoid.
```

**`shared/selfplay_pool.py`** <module> (docstring)

```text
"""
selfplay_pool (2026-09-26)
================================
One persistent multi-process pool that plays ALL of an epoch's training
games at once -- self-play (including curriculum/opening starts),
vs-Deterministic and vs-League -- instead of three back-to-back phases.

Built from selfplay_pool_fallback.py (kept unchanged as the fallback path). Why:

* The old sequence ran self-play (multi-process), then vs-Det (threaded),
  then vs-League (threaded). Each phase ends only when its slowest game
  does, so the tail of every phase left most workers idle, and the
  2-game vs-League phase kept 8 of 10 workers and most of the GPU idle
  for its whole duration (live log: evaluator avg batch 30 -> 22 during
  it, "hit target" falling to 57%).
* Here every game of the epoch goes onto ONE task queue that 10 worker
  processes pull from (longest games first), so workers stay busy until
  the queue drains and the GPU coordinator keeps getting full batches.

What changed relative to selfplay_pool_fallback.py
---------------------------------------
* Workers accept three task kinds ('self', 'vsdet', 'league') and call the
  UNMODIFIED production game functions for each (same principle as before:
  no hand-ported game logic, so nothing can silently drift).
* Eval requests carry a model TAG ('gen' or 'league'). The coordinator
  groups a flush by tag and runs one forward pass per model, so a league
  game's opponent side is evaluated by the league checkpoint while
  everything else uses the generation model. Workers still hold no
  weights; the league model is handed to the coordinator per call.
* Each worker keeps two process-local eval caches (gen / league), both
  reset per call (= per epoch), so cache validity is unchanged.
* Samples are aggregated exactly like the production functions
  (value = outcome from the mover's side, weights 1.15 mover-won /
  1.5 vs-Det / 1.25 vs-League, survival label for self-play only) and in
  the same order (self-play, then vs-Det, then vs-League).

Safety (same model as selfplay_pool_fallback.py)
-------------------------------------
Any failure (worker death, coordinator/GPU error, IPC error, stall) tears
the pool down and the epoch is re-run through `fallback()` -- the caller's
original three-phase sequence -- so a bug here costs time, never the run.
After MAX_STRIKES failures this path disables itself for the process.
Kill switch mid-run: create PROD_DIR\\MP_UNIFIED_OFF (checked each epoch).
"""
```

**`shared/selfplay_pool_fallback.py`** <module> (docstring)

```text
"""
generate_mcts_self_play_data_mp
===============================
Multi-PROCESS drop-in replacement for
``naps_trainer.generate_mcts_self_play_data`` (the
ThreadPoolExecutor-based self-play game generator).

Derived from the "pipe_batch_local" candidate, which was selected over the
"queue_pool" and "shared_memory" candidates. See WINNER_deploy_notes.md for
the full selection rationale and -- importantly -- for the honest status of
the PERFORMANCE claim, which is summarized here too because nobody should
deploy this without reading it:

    NONE of the three candidates demonstrated a reproducible speedup.
    Every benchmark was taken on a machine whose GPU and all 8 logical
    cores were simultaneously saturated by the live production training
    run, and the run-to-run noise (+-40%) was larger than any effect.
    Independent re-measurements of this candidate landed at 0.78x and
    1.21x -- i.e. indistinguishable from parity. This file is therefore a
    DELIBERATE EXPERIMENT, not a proven optimization. It is built to be
    safe to try in the live loop and trivial to back out, and it
    instruments itself so the A/B answer can finally be measured under
    real single-tenant conditions instead of under benchmark contention.

============================================================
WHY THIS ARCHITECTURE (and why it beat the other two)
============================================================
The diagnosed problem: the "glue" work around each leaf evaluation (numpy
board encoding + the evaluator submit/wait round trip) does NOT release the
GIL, while both the GPU forward pass and the C++ MCTS tree walk do. With
max_workers=10 self-play THREADS in one process, CPython serializes that
glue across all of them.

Corroborating evidence that is NOT wall-clock (and therefore not corrupted
by benchmark contention) -- straight out of the live training log:

    [PooledEvaluator stats] 13500 flushes, avg batch 5.0, max 20,
    target_batch=24, 0/13500 (0%) hit target vs timed out short

Every single flush times out short, at an average batch of 5 against a
target of 24. Ten concurrent self-play threads essentially never have more
than one eval request pending at the same instant. That is the GIL
serialization, visible in production, right now. The shared_memory
candidate independently measured GPU utilization rising from 10-40%
(threaded) to 55-76% (multi-process), confirming the architecture does fix
the batching. What remains unproven is whether fixing it actually buys
wall-clock, since the real bottleneck may simply be the CPU-bound C++ tree
walk on 8 already-oversubscribed cores.

This candidate's structure:

1. Worker processes (persistent pool, spawned ONCE per training run, never
   respawned per epoch). Each is a plain CPU-only OS process with NO CUDA
   context. It plays games by calling the production
   ``play_one_mcts_game`` COMPLETELY UNMODIFIED, imported read-only from
   the live tree. This is the decisive advantage over the other two
   candidates, both of which hand-ported the game loop, the board encoding
   and/or the augmentation logic into their own files -- creating a second
   copy of production logic that would silently drift the moment anyone
   edits the real one. Here, the only thing that differs from production is
   the *evaluator object* plugged into MCTS; every rule, every curriculum
   branch, every sampling decision is literally the same bytes.

2. The evaluator plugged into each worker's ``MCTS(pooled_evaluator=...)``
   is ``_PipeEvalProxy``, a duck-type of ``PooledEvaluator.submit()``. It
   sends ONE numpy array per already-batched leaf group down a dedicated
   ``multiprocessing.Pipe`` and blocks for the paired result.

   The "batch_local" part comes free: the C++ engine's ``search_batched()``
   already groups eval_batch_size=16 simulations' leaves per round via
   virtual loss BEFORE calling back into Python. So one pipe message
   already carries up to 16 boards -- roughly ceil(sims/16) round trips per
   ply, not one per simulation.

   A dedicated Pipe per worker (rather than one shared Queue) means writers
   never contend on a shared lock; the coordinator reads all of them with a
   single multiplexed ``multiprocessing.connection.wait()``.

3. Exactly ONE thing owns the GPU: ``PipePooledEvaluator``, a background
   THREAD in the MAIN process -- the same process that already holds
   ``model`` on ``device``. No second CUDA context, no checkpoint
   save/reload handshake, no weight-shipping, and therefore ZERO weight
   staleness: workers never hold weights at all. (This is the other reason
   this beat "shared_memory", which needed a separate GPU-owning process
   and a per-call torch.save/torch.load round trip -- extra VRAM on a 4GB
   GTX 1050 and real per-call latency.)

   Its accumulate-until-target_batch-or-silence-gap logic is a direct port
   of ``PooledEvaluator._run``, including that implementation's
   silence-gap-not-deadline quirk, so the two are apples-to-apples at the
   same target_batch/max_wait operating point.

4. Finished games come back over an ordinary Queue -- once per game, not
   once per simulation, so pickling cost there is amortized to nothing.

============================================================
SAFETY DESIGN (what makes this safe to gamble on in a live run)
============================================================
This is going into a 20000-epoch run that must not die. Every known
failure mode degrades to "the original threaded implementation runs this
epoch instead", never to a crash or a hang:

* ANY exception from the MP path (worker death, GPU OOM in the coordinator,
  IPC breakage, timeout) is caught, the pool is torn down, and the epoch is
  transparently re-run on the ORIGINAL production threaded function. The
  epoch still produces correct data; it just doesn't get the experiment.
* After ``MAX_MP_STRIKES`` such failures the MP path disables itself for
  the rest of the process. A recurring problem degrades to "we're just
  running the old code", quietly and permanently.
* Kill switch without editing any code: set ``MP_SELFPLAY_DISABLE=1`` in
  the environment and this becomes a pure pass-through to the original.
* Stall detection is PROGRESS-based, not a fixed deadline: as long as the
  coordinator keeps flushing eval batches or games keep finishing, the call
  is considered healthy. A genuinely wedged pool is caught in ~3 minutes
  and falls back, instead of hanging for the 24 minutes a naive
  games x per-game deadline would have allowed.
* If the coordinator thread ever dies mid-flush, every worker blocked on
  ``recv()`` is explicitly woken with an error sentinel so it fails fast,
  rather than blocking forever on a pipe nobody will ever write to again.
* Workers are daemon processes AND are shut down via atexit, so they cannot
  outlive the trainer.

============================================================
KNOWN, DELIBERATE DEVIATIONS FROM THE ORIGINAL
============================================================
* ``eval_cache`` is accepted but IGNORED. There is no way to share the
  caller's in-process dict with separate processes. Each worker keeps its
  own process-local cache, reset when it sees a new call_id (== a new
  epoch's frozen weights), so cache VALIDITY is identical -- a cached entry
  is still a pure function of frozen weights. What is reduced is the DEGREE
  of cross-game de-dup: in the production shape (num_games=8 spread over 10
  workers) most workers see exactly one game, so they get little cross-game
  reuse, whereas the original's single shared dict saw every game's boards
  AND was additionally shared with the vs-Det / vs-League phases that still
  run threaded. Cost: some extra GPU forward passes. Never a wrong value.
  This is a real, unquantified handicap against the threaded baseline and
  it is included in every measured number.
* ``pooled_evaluator`` is accepted but IGNORED: the MP path must use its
  own cross-process evaluator. The caller's PooledEvaluator is untouched
  and continues serving the still-threaded vs-Det / vs-League phases.
* Per-game randomness comes from each worker's own unseeded RNG stream
  rather than one shared in-process stream. Self-play was never
  reproducible run-to-run either way.
* Only ``generate_mcts_self_play_data`` is replaced. vs-Deterministic and
  vs-League generation stay on the threaded path.
"""
```

**`shared/selfplay_pool_fallback.py`** generate_mcts_self_play_data_mp (docstring)

```text
    """Multi-process drop-in for
    ``naps_trainer.generate_mcts_self_play_data``.

    Identical call signature (plus target_batch/max_wait, which the original
    hard-codes at its PooledEvaluator construction site rather than exposing)
    and identical ``(raw_samples, outcome_counts)`` return contract:
    raw_samples is a list of ``(aug_board_3d, aug_policy_flat, final_val,
    sample_weight)`` rows, 8x dihedral-augmented, and outcome_counts is
    ``{'x_win', 'o_win', 'draw'}`` summing to num_games.

    ``eval_cache`` and ``pooled_evaluator`` are accepted for signature
    compatibility but ignored on the MP path -- see the module docstring.
    They ARE forwarded intact to the threaded implementation whenever this
    falls back, so fallback behavior is byte-for-byte the original.

    On ANY multi-process failure this transparently re-runs the epoch on the
    original threaded implementation, so a bug here can cost time but can
    never cost the training run.
    """
```
