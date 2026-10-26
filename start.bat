@echo off
setlocal EnableExtensions
title RAG4Prefa - 工程文档知识库

rem ============================================================
rem  RAG4Prefa 一键启动
rem
rem  这个脚本存在的理由不是"少打一行命令"，而是把几个容易踩的
rem  约束固化下来：
rem    1. 必须在项目根目录启动。R4P_DATA_DIR=data 是相对路径，
rem       从别处启动会安静地新建一个空库——页面上显示没有语料，
rem       其实数据还在 data\app.db 里，只是没读到。
rem    2. 必须用 .venv 里的 python。系统 python 没装依赖。
rem    3. 端口不用每次记；端口被占时必须说清楚，而不是打完地址就崩。
rem
rem  用法：双击，或 start.bat 8000 指定端口（默认 8899）
rem
rem  2026-10-26 改：之前是「先 echo 地址，再 serve」。端口被占时
rem  uvicorn 报 [Errno 10048] 后走到 :end，没有 pause，窗口直接关掉；
rem  用户只看到闪过一行地址，去浏览器打开当然拒绝连接，看起来就像
rem  "双击没反应"。现在改成「先探端口，空了才打印地址；失败必 pause」。
rem ============================================================

rem %~dp0 是脚本所在目录（带尾部反斜杠），保证 cwd 正确
cd /d "%~dp0"

set "HOST=127.0.0.1"
set "PORT=8899"
if not "%~1"=="" set "PORT=%~1"

echo.
echo ============================================
echo   RAG4Prefa  工程文档知识库
echo ============================================
echo.

rem 端口必须是十进制数字：netstat 探不了非数字，uvicorn 的报错也看不懂。
rem 显式排除前导零——cmd 的 if / set /a 会把 08000 当八进制解析。
echo %PORT%| findstr /R "^[1-9][0-9]*$" >nul
if errorlevel 1 (
    echo [错误] 端口 "%PORT%" 不合法，要 1-65535 的十进制数字。
    echo        用法：start.bat 8000
    goto :bail
)
if %PORT% GTR 65535 (
    echo [错误] 端口 %PORT% 超出范围，最大 65535。
    goto :bail
)

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 找不到 .venv\Scripts\python.exe
    echo.
    echo 先建虚拟环境再装依赖：
    echo     python -m venv .venv
    echo     .venv\Scripts\python.exe -m pip install -e ".[milvus,dev]"
    goto :bail
)

if not exist ".env" (
    echo [错误] 找不到 .env
    echo.
    echo 从模板复制一份，再填两个 Key：
    echo     copy .env.example .env
    goto :bail
)

if not exist "data\app.db" (
    echo [提示] data\app.db 不存在，这是一个空库。
    echo        先去把文档灌进来，例如：
    echo        .venv\Scripts\python.exe -m rag4prefa ingest ^<文档目录^> --project 我的项目
    echo.
)

rem ---- 端口预检：先确认能绑上，再谈"地址是什么" ----
rem 顺序在这里是有意义的：地址只能代表"即将使用"，不能代表"已经就绪"。
call :port_listening
if not errorlevel 1 goto :port_taken

:ready
echo   即将启动。等下面出现下面这一行
echo       Application startup complete
echo   才说明服务起来了，这时再打开浏览器：
echo.
echo       http://%HOST%:%PORT%
echo.
echo   停止：在本窗口按 Ctrl+C
echo.

rem 启动前跑一次自检：缺 Key、索引不一致在这里就报出来，
rem 而不是等点了提问才发现 503。
.venv\Scripts\python.exe -m rag4prefa doctor >nul 2>&1
if errorlevel 1 (
    echo [注意] 自检未通过，原因如下：
    echo.
    .venv\Scripts\python.exe -m rag4prefa doctor
    echo.
    echo 服务仍会启动，但入库/问答大概率不可用。
    echo.
)

.venv\Scripts\python.exe -m rag4prefa serve --host %HOST% --port %PORT%
set "RC=%ERRORLEVEL%"

rem Ctrl+C 中断时 cmd 给的是 0xC000013A（按有符号解释），不算错误。
if "%RC%"=="0" goto :end
if "%RC%"=="-1073741510" (
    echo.
    echo 服务已停止。
    goto :end
)

echo.
echo [错误] 服务异常退出，返回码 %RC%。
echo.
echo   排查顺序：
echo     1. 若上面出现 Errno 10048，说明端口刚被别人抢走，换个端口：
echo            start.bat 8000
echo     2. 完整自检报告：
echo            .venv\Scripts\python.exe -m rag4prefa doctor
echo     3. R4P_DATA_DIR 是相对路径，必须在本目录启动（本脚本已 cd）。
echo.
pause
goto :end

:port_taken
echo [提示] 端口 %PORT% 已经在监听，本次不再启动第二个实例。
echo        这种情况多半是之前开的窗口还开着，不是启动失败。
echo.
echo   如果那就是本程序，直接打开：
echo        http://%HOST%:%PORT%
echo   要停掉它，去那个黑色窗口按 Ctrl+C。
echo.
set /a "NEXT=PORT+1"
echo   如果那是别的软件占着，换一个端口：
echo        start.bat %NEXT%
echo.
echo   看看究竟是谁占着：
echo        netstat -ano ^| findstr ":%PORT%"
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
endlocal
exit /b 0

rem ------------------------------------------------------------
rem  子过程：端口已有 TCP 监听返回 0，空闲返回 1
rem  netstat 里本地地址后面一定有空格，用 ":PORT " 匹配可以避免
rem  把 88990 误判成 8899。只看 LISTENING，已建立的连接不算占用。
rem ------------------------------------------------------------
:port_listening
set "BUSY="
for /f "delims=" %%L in ('netstat -ano -p TCP ^| findstr /C:":%PORT% " ^| findstr /C:"LISTENING"') do set "BUSY=1"
if defined BUSY (
    set "BUSY="
    exit /b 0
)
exit /b 1
