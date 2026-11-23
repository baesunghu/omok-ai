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
