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

set "VSWHERE_DIR=C:\Program Files (x86)\Microsoft Visual Studio\Installer"
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "PY_INCLUDE=C:\Users\My PC\AppData\Local\Programs\Python\Python311\Include"
set "PY_LIBS=C:\Users\My PC\AppData\Local\Programs\Python\Python311\libs"
set "PYBIND_INCLUDE=C:\Users\My PC\AppData\Local\Programs\Python\Python311\Lib\site-packages\pybind11\include"
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
