@echo off
setlocal EnableExtensions
title RAG4Prefa - 停止服务

rem ============================================================
rem  停止 RAG4Prefa（默认端口 8899）
rem
rem  为什么需要它：点黑窗口右上角的 X 只保证 cmd 没了，Python 进程树不一定
rem  跟着退（.venv 里的 python.exe 是启动器，真正监听端口的是它拉起的子进程）。
rem  端口没释放，再双击 start.bat 就只剩一句"已经在监听"。
rem
rem  用法：
rem      双击 stop.bat           停默认端口 8899
rem      stop.bat 8900           停指定端口
rem      stop.bat 8899 /force    健康检查没回应（服务卡死）时强制停
rem
rem  先认身份再动手：连 /api/health 看它自报是不是 rag4prefa，认不出来就
rem  不杀，只把 PID 和命令打给你。误杀别人家的服务比停不掉更麻烦。
rem ============================================================

cd /d "%~dp0"

set "PORT=8899"
if not "%~1"=="" set "PORT=%~1"
set "FORCE="
if /i "%~2"=="/force" set "FORCE=--force"
if /i "%~2"=="--force" set "FORCE=--force"

echo.
echo ============================================
echo   RAG4Prefa  停止服务
echo ============================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 找不到 .venv\Scripts\python.exe
    echo.
    echo 没有虚拟环境就直接用系统命令：
    echo     netstat -ano ^| findstr ":%PORT%"
    echo     taskkill /PID ^<上面查到的PID^> /T /F
    goto :bail
)

.venv\Scripts\python.exe -m rag4prefa stop --port %PORT% %FORCE%
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
    echo 结束。要重新启动就双击 start.bat。
) else (
    echo 没有停。按上面的提示处理，别手动 taskkill 别人的进程。
)
goto :bail

:bail
echo.
pause
if not defined RC set "RC=1"
rem endlocal 之后再取 %RC% 就是空的了，所以在同一行展开
endlocal & exit /b %RC%
