"""Every path the scripts need, in one place.

The repository root is the laptop's old `final/` folder, so the layout is the same everywhere:

    <ROOT>/shared/         code that is the same for every run (one copy)
    <ROOT>/11x11_naps/     11x11 NAPS run (until 2026-11 the folder was 10x10/, from the 10x10-board days)
    <ROOT>/11x11_v2/       v2 run (VCF + C++ self-play); v3 lives in 11x11_v2/v3/
    <ROOT>/15x15/          15x15 run
    <ROOT>/archive/        one-off scripts kept for the record
    <ROOT>/cpp_mcts/       C++ MCTS engine sources (the built module is placed next to them)
    <ROOT>/engines/rapfi/  Rapfi engine -- NOT in the repository, download it separately
    <ROOT>/scratch/        throwaway outputs -- NOT in the repository

A run folder holds only its settings (run_config.py: board size, gate, LR, engine, watchers) and its own scripts;
all shared logic lives once in shared/. A run-folder script makes shared/ importable with

    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shared"))

and a shared tool works on a run folder chosen by run_dir(): run it from inside the run folder
(`cd 11x11_v2 && python ../shared/ladder_ci.py`) or set OMOK_RUN.

Anything outside the repository can be moved with an environment variable:

    OMOK_RUN           run folder a shared tool works on        (default: the current directory)
    OMOK_RAPFI_DIR     folder that holds the Rapfi executable   (default <ROOT>/engines/rapfi)
    OMOK_RAPFI_EXE     Rapfi executable file name               (default pbrain-rapfi-windows-avx2.exe)
    OMOK_SCRATCH_DIR   scratch outputs                          (default <ROOT>/scratch)
    OMOK_DOWNLOADS     folder with the paper drafts (.docx)     (default ~/Downloads)
"""
import glob
import importlib.machinery
import os
import sys


def _env_dir(name, default):
    return os.path.abspath(os.path.expanduser(os.environ.get(name) or default))


ROOT = os.path.dirname(os.path.abspath(__file__))

SHARED_DIR = os.path.join(ROOT, "shared")
NAPS_DIR = os.path.join(ROOT, "11x11_naps")          # 11x11 NAPS run
V2_DIR = os.path.join(ROOT, "11x11_v2")
V3_DIR = os.path.join(V2_DIR, "v3")
CPP_MCTS_DIR = os.path.join(ROOT, "cpp_mcts")

RAPFI_DIR = _env_dir("OMOK_RAPFI_DIR", os.path.join(ROOT, "engines", "rapfi"))
RAPFI_EXE = os.path.join(RAPFI_DIR, os.environ.get("OMOK_RAPFI_EXE") or "pbrain-rapfi-windows-avx2.exe")
SCRATCH_DIR = _env_dir("OMOK_SCRATCH_DIR", os.path.join(ROOT, "scratch"))
DOWNLOADS_DIR = _env_dir("OMOK_DOWNLOADS", os.path.join("~", "Downloads"))

# Data with one copy in the repository, used by every 11x11 run (11x11_naps, 11x11_v2, v3):
PROBLEMS_FILE = os.path.join(NAPS_DIR, "tactical_problems", "web_game58_problems.json")   # 28 problems, 11x11 only
WEB_GAMES_LOG = os.path.join(NAPS_DIR, "hlt", "web", "games.jsonl")   # visitors' games on the web board (not in the repo)


def _is_run(d):
    return os.path.isfile(os.path.join(d, "run_config.py"))   # every run folder has its own run_config.py


def run_dir():
    """The run folder (11x11_naps / 11x11_v2 / 15x15) that shared code works on.

    $OMOK_RUN if set, else the current directory if it is a run folder, else the run folder already on sys.path
    (shared code imported by a run-folder script). Exits with a message when there is none.
    """
    if os.environ.get("OMOK_RUN"):
        d = _env_dir("OMOK_RUN", "")
        if not _is_run(d):
            sys.exit(f"OMOK_RUN={d} is not a run folder (no run_config.py)")
        return d
    if _is_run(os.getcwd()):
        return os.path.abspath(os.getcwd())
    for p in sys.path:
        if p and _is_run(p):
            return os.path.abspath(p)
    sys.exit("run this from a run folder (cd 11x11_naps / 11x11_v2 / 15x15) or set OMOK_RUN")


def enter_run():
    """For a shared tool run as a script: pick the run folder (run_dir), make it the working directory and put it
    first on sys.path (its run_config.py and built engines), and return it."""
    d = run_dir()
    if d not in sys.path:
        sys.path.insert(0, d)
    os.chdir(d)
    return d


def find_extension(folder, name):
    """Path of a built C++ module `name` in `folder` (.pyd on Windows, .so elsewhere), or None."""
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        p = os.path.join(folder, name + suffix)
        if os.path.exists(p):
            return p
    hits = sorted(glob.glob(os.path.join(folder, name + ".*.pyd")) + glob.glob(os.path.join(folder, name + "*.so")))
    return hits[0] if hits else None
