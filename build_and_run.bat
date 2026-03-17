@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cargo build
if %ERRORLEVEL% EQU 0 (
    echo Build successful! Running tests...
    cargo test
) else (
    echo Build failed!
)
pause
