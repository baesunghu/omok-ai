@echo off
REM build_mcts_core_solver.bat
REM
REM Builds cpp_mcts\mcts_core_solver.cpp (mcts_core_tree_reuse.cpp +
REM mate-distance terminal values) into a SEPARATE .pyd. Nothing live is
REM touched. Output module/file name comes from the first argument so the
REM same source can be built as a side-by-side test module
REM (mcts_core_mate_distance) or, after testing, as the deploy build
REM (mcts_core) -- the PYBIND11_MODULE name in the .cpp must match.
REM
REM Usage: build_mcts_core_solver.bat <output_basename>
REM
REM vswhere.exe PATH fix: see build_mcts_core_tree_reuse.bat.

setlocal

set "OUT_NAME=%~1"
if "%OUT_NAME%"=="" set "OUT_NAME=mcts_core_solver"

REM Python and pybind11 paths are read from the python on PATH (set PYTHON=<path\to\python.exe> to pick another).
REM VCVARS defaults to VS 2022 Build Tools in its standard location; set VCVARS=... to override.
if not defined PYTHON set "PYTHON=python"
set "VSWHERE_DIR=C:\Program Files (x86)\Microsoft Visual Studio\Installer"
if not defined VCVARS set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
for /f "usebackq delims=" %%i in (`%PYTHON% -c "import sysconfig; print(sysconfig.get_paths()['include'])"`) do set "PY_INCLUDE=%%i"
for /f "usebackq delims=" %%i in (`%PYTHON% -c "import os, sys; print(os.path.join(sys.base_prefix, 'libs'))"`) do set "PY_LIBS=%%i"
for /f "usebackq delims=" %%i in (`%PYTHON% -c "import pybind11; print(pybind11.get_include())"`) do set "PYBIND_INCLUDE=%%i"
if not defined PYBIND_INCLUDE (
    echo ERROR: pybind11 not found for %PYTHON% -- run: %PYTHON% -m pip install pybind11
    exit /b 1
)
set "SRC_DIR=%~dp0"

set "PATH=%VSWHERE_DIR%;%PATH%"
call "%VCVARS%"
if errorlevel 1 (
    echo ERROR: vcvars64.bat failed even after PATH fix.
    exit /b 1
)

echo.
echo === Compiling mcts_core_solver.cpp -^> %OUT_NAME%.pyd ===
cd /d "%SRC_DIR%"
cl.exe /nologo /LD /EHsc /std:c++17 /O2 /MD ^
    /I"%PY_INCLUDE%" ^
    /I"%PYBIND_INCLUDE%" ^
    /DMCTS_MODULE_NAME=%OUT_NAME% ^
    "%SRC_DIR%mcts_core_solver.cpp" ^
    /Fo:"%SRC_DIR%%OUT_NAME%.obj" ^
    /link /LIBPATH:"%PY_LIBS%" /OUT:"%SRC_DIR%%OUT_NAME%.pyd"

if errorlevel 1 (
    echo ERROR: compile/link failed.
    exit /b 1
)

echo.
echo === Build succeeded: %SRC_DIR%%OUT_NAME%.pyd ===
endlocal
