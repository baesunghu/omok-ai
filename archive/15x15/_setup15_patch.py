"""2026-10-14: point the copied tools at 15x15 files (one-off patch)."""
import shutil


def R(f, a, b, all_=False):
    s = open(f, encoding='utf-8').read()
    assert a in s, (f, a)
    s = s.replace(a, b) if all_ else s.replace(a, b, 1)
    open(f, 'w', encoding='utf-8').write(s)


shutil.copy('_restart_train.py', '_restart_train15.py')
f = 'rapfi_ladder_watch.py'
R(f, 'train_state_11x11.json', 'train_state_15x15.json', True)
R(f, 'best_mcts_11x11.pth', 'best_mcts_15x15.pth', True)
R(f, 'RAPFI_LEVELS="1000,2000,3000,5000,10000"', 'RAPFI_LEVELS="300,1000,3000,5000,10000"  # 2026-10-14: 15x15 (epoch 0 = ~766)')
R('watchdog_.py', 'TRAIN_SCRIPT_MARKER = "_restart_train.py"', 'TRAIN_SCRIPT_MARKER = "_restart_train15.py"  # 2026-10-14: distinct from the 11x11 trainer')
R('watchdog_.py', "-ArgumentList '-u _restart_train.py'", "-ArgumentList '-u _restart_train15.py'")
f = 'promotion_sms.py'
R(f, '_report_11x11.txt', '_report_15x15.txt', True)
R(f, 'return f"NAPS ep{ep} ', 'return f"15x15 ep{ep} ')
R(f, 'return f"NAPS {name} ', 'return f"15x15 {name} ')
f = 'ladder_promo.py'
R(f, '    if g is None:\n        return msg + " -> no promotion (grade the graduation problems first)"',
  '    if g is None and ref.get("grad") is not None:   # 2026-10-14 (15x15): no graduation problems yet\n'
  '        return msg + " -> no promotion (grade the graduation problems first)"')
R(f, '    if gb is not None and g[0] < gb:', '    if gb is not None and g is not None and g[0] < gb:')
print("patched")
