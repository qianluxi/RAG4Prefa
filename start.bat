@echo off
setlocal EnableExtensions
title RAG4Prefa - 工程文档知识库

rem ============================================================
rem  RAG4Prefa 一键启动
rem
rem  注意事项：
rem    1. 数据目录以脚本所在目录为基准，R4P_DATA_DIR=data 这类相对路径
rem       会落在本目录；换个目录双击等于打开另一个空库——页面提示"没有资料"
rem       不代表数据丢了，先看是不是在别的目录里启动的。
rem    2. 一律用 .venv 里的 python：系统 python 没装依赖。
rem    3. 端口不写死：被占时要么说清楚谁占着，要么换端口。
rem
rem  用法：双击运行；start.bat 8000 指定端口，默认 8899。
rem
rem  2026-10-26 改：之前是「先 echo 地址再 serve」，端口被占时
rem  uvicorn 报 [Errno 10048]，脚本走到 :end 没有 pause，黑窗口一闪而过，
rem  用户只看到一行地址就去浏览器，打不开当然拒绝连接，于是报障"双击没反应"。
rem  现在改成「先探端口，能起才打印地址，失败必 pause」。
rem
rem  2026-10-27 改：端口被占时把「谁占着」的 PID 打出来，并指向 stop.bat。
rem  用户点黑窗口右上角的 X 之后，再双击还是同一句"已经在监听"——因为点 X
rem  只关掉了 cmd，Python 进程还在监听（venv 里的 python.exe 是启动器，
rem  真正监听端口的是它拉起的子进程）。关窗口不等于关服务，提示里要写明白，
rem  并且给出一个能做到这件事的入口：stop.bat。
rem ============================================================

rem %~dp0 是脚本所在目录，尾部带反斜杠，保证 cwd 正确
cd /d "%~dp0"

set "HOST=127.0.0.1"
set "PORT=8899"
if not "%~1"=="" set "PORT=%~1"

echo.
echo ============================================
echo   RAG4Prefa  工程文档知识库
echo ============================================
echo.

rem 端口必须是十进制数字：netstat 探的是服务端口，uvicorn 的报错也不是人话。
rem 先检查再转数字——cmd 的 if / set / a 会把 08000 当成八进制解释。
echo %PORT%| findstr /R "^[1-9][0-9]*$" >nul
if errorlevel 1 (
    echo [错误] 端口 "%PORT%" 不合法，要 1-65535 的十进制数字。
    echo        用法：start.bat 8000
    goto :bail
)
if %PORT% GTR 65535 (
    echo [错误] 端口 %PORT% 超出范围（最大 65535）。
    goto :bail
)

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 找不到 .venv\Scripts\python.exe
    echo.
    echo 先建虚拟环境并装依赖：
    echo     python -m venv .venv
    echo     .venv\Scripts\python.exe -m pip install -e ".[milvus,dev]"
    goto :bail
)

if not exist ".env" (
    echo [错误] 找不到 .env
    echo.
    echo 从模板复制一份，填上自己的 Key：
    echo     copy .env.example .env
    goto :bail
)

if not exist "data\app.db" (
    echo [提示] data\app.db 还不存在，将打开一个空库。
    echo        先去导入文档，例如：
    echo        .venv\Scripts\python.exe -m rag4prefa ingest ^<文档目录^> --project 我的项目
    echo.
)

rem ---- 端口预检：不确认能绑上，就别谈"地址是什么" ----
rem 顺序是有意义的：地址只能出现在"本次可用"之后，不能出现在"已经开着"之后。
call :port_listening
if not errorlevel 1 goto :port_taken

:ready
echo   启动命令已经发出，看到这一行才算真的起来了：
echo       Application startup complete
echo   此时再打开浏览器：
echo.
echo       http://%HOST%:%PORT%
echo.
echo   停止：在本窗口按 Ctrl+C。
echo   窗口已经关掉、但浏览器还能打开时，说明进程没死透：
echo       双击 stop.bat（或 stop.bat %PORT%）把它停干净。
echo.

rem 起服务前先自检：缺 Key 这类问题早点说，别等提问了才发 503
.venv\Scripts\python.exe -m rag4prefa doctor >nul 2>&1
if errorlevel 1 (
    echo [注意] 自检没通过，原因如下：
    echo.
    .venv\Scripts\python.exe -m rag4prefa doctor
    echo.
    echo 服务仍会启动，但入库/问答可能不可用。
    echo.
)

.venv\Scripts\python.exe -m rag4prefa serve --host %HOST% --port %PORT%
set "RC=%ERRORLEVEL%"

rem Ctrl+C 中断时 cmd 返回 0xC000013A，按有符号解释就是下面的负数
if "%RC%"=="0" goto :end
if "%RC%"=="-1073741510" (
    echo.
    echo 已正常停止。
    goto :end
)

echo.
echo [错误] 服务异常退出（返回码 %RC%）。
echo.
echo   排查顺序
echo     1. 如果报错里有 Errno 10048，说明端口刚被别人抢走，换个端口：
echo            start.bat 8000
echo     2. 看自检报告：
echo            .venv\Scripts\python.exe -m rag4prefa doctor
echo     3. R4P_DATA_DIR 若指向别的路径，数据不在本目录（脚本已 cd 到脚本目录）。
echo.
pause
goto :end

:port_taken
echo [提示] 端口 %PORT% 已经在监听，本次不再启动第二个实例。
echo        这不是启动失败，而是这个端口上已经有一个在跑。
echo        多半是之前那个黑色窗口没关干净：点右上角的 X 只会关掉窗口，
echo        Python 进程可能还在后台监听。
echo.
call :listening_pid
if defined OWNER (
    echo   占着端口的是：PID %OWNER%
    for /f "tokens=1 delims=," %%N in ('tasklist /FI "PID eq %OWNER%" /FO CSV /NH 2^>nul') do if not defined OWNER_NAME set "OWNER_NAME=%%~N"
)
if defined OWNER_NAME echo   进程名：%OWNER_NAME%
echo.
echo   如果那就是本程序，直接打开：
echo        http://%HOST%:%PORT%
echo   想把它停干净（下次能正常双击启动）：
echo        双击 stop.bat
echo        或者回到那个黑窗口按 Ctrl+C
echo.
set /a "NEXT=PORT+1"
echo   如果那是别的软件占着，换一个端口：
echo        start.bat %NEXT%
echo   想自己确认是谁占着：
echo        netstat -ano ^| findstr ":%PORT%"
echo        tasklist /FI "PID eq %OWNER%"
echo.
pause
endlocal
exit /b 1

:bail
echo.
pause
endlocal
exit /b 1

:end
echo.
endlocal
exit /b 0

rem ------------------------------------------------------------
rem  子过程：端口是否已被监听。TCP 连接数 >0、有服务返回 1
rem  netstat 里本地地址后面一定跟空格，用 ":PORT " 匹配可以避免
rem  把 88990 误判成 8899。只看 LISTENING：已建立的连接不算占用。
rem ------------------------------------------------------------
:port_listening
set "BUSY="
for /f "delims=" %%L in ('netstat -ano -p TCP ^| findstr /C:":%PORT% " ^| findstr /C:"LISTENING"') do set "BUSY=1"
if defined BUSY (
    set "BUSY="
    exit /b 0
)
exit /b 1

rem ------------------------------------------------------------
rem  子过程：取监听该端口的 PID（netstat 第 5 列），供占位提示显示
rem ------------------------------------------------------------
:listening_pid
set "OWNER="
set "OWNER_NAME="
for /f "tokens=5" %%P in ('netstat -ano -p TCP ^| findstr /C:":%PORT% " ^| findstr /C:"LISTENING"') do if not defined OWNER set "OWNER=%%P"
exit /b 0
