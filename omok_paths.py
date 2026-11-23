"""Every path the scripts need, in one place.

The repository root is the laptop's old `final/` folder, so the layout is the same everywhere:

    <ROOT>/10x10/          11x11 NAPS run (the folder name is historical; the board is 11x11)
    <ROOT>/11x11_v2/       v2 run (VCF + C++ self-play); v3 lives in 11x11_v2/v3/
    <ROOT>/15x15/          15x15 run
    <ROOT>/cpp_mcts/       C++ MCTS engine sources (the built module is placed next to them)
    <ROOT>/engines/rapfi/  Rapfi engine -- NOT in the repository, download it separately
    <ROOT>/scratch/        throwaway outputs -- NOT in the repository

Anything outside the repository can be moved with an environment variable:

    OMOK_RAPFI_DIR     folder that holds the Rapfi executable   (default <ROOT>/engines/rapfi)
    OMOK_RAPFI_EXE     Rapfi executable file name               (default pbrain-rapfi-windows-avx2.exe)
    OMOK_SCRATCH_DIR   scratch outputs                          (default <ROOT>/scratch)
    OMOK_DOWNLOADS     folder with the paper drafts (.docx)     (default ~/Downloads)

Scripts inside the repository import this module after adding the repository root to sys.path, e.g.

    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(HERE))          # repository root -> omok_paths
    from omok_paths import RAPFI_DIR

Paths that belong to the script's own run folder are written relative to HERE, so a copied run folder
(10x10 -> 15x15) keeps pointing at its own files.
"""
import glob
import importlib.machinery
import os


def _env_dir(name, default):
    return os.path.abspath(os.path.expanduser(os.environ.get(name) or default))


ROOT = os.path.dirname(os.path.abspath(__file__))

NAPS_DIR = os.path.join(ROOT, "10x10")          # 11x11 NAPS run
V2_DIR = os.path.join(ROOT, "11x11_v2")
V3_DIR = os.path.join(V2_DIR, "v3")
P15_DIR = os.path.join(ROOT, "15x15")
CPP_MCTS_DIR = os.path.join(ROOT, "cpp_mcts")
DOCS_DIR = os.path.join(ROOT, "docs")

RAPFI_DIR = _env_dir("OMOK_RAPFI_DIR", os.path.join(ROOT, "engines", "rapfi"))
RAPFI_EXE = os.path.join(RAPFI_DIR, os.environ.get("OMOK_RAPFI_EXE") or "pbrain-rapfi-windows-avx2.exe")
SCRATCH_DIR = _env_dir("OMOK_SCRATCH_DIR", os.path.join(ROOT, "scratch"))
DOWNLOADS_DIR = _env_dir("OMOK_DOWNLOADS", os.path.join("~", "Downloads"))


def board_suffix(board_size):
    """File-name suffix the training code gives every model/report file, e.g. '_11x11'."""
    return "" if board_size == 3 else f"_{board_size}x{board_size}"


def find_extension(folder, name):
    """Path of a built C++ module `name` in `folder` (.pyd on Windows, .so elsewhere), or None."""
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        p = os.path.join(folder, name + suffix)
        if os.path.exists(p):
            return p
    hits = sorted(glob.glob(os.path.join(folder, name + ".*.pyd")) + glob.glob(os.path.join(folder, name + "*.so")))
    return hits[0] if hits else None
