@echo off
setlocal
rem ============================================================
rem  RAG4Prefa 一键启动
rem
rem  这个脚本存在的理由不是"少打一行命令"，而是把三个容易踩的
rem  约束固化下来：
rem    1. 必须在项目根目录启动。R4P_DATA_DIR=data 是相对路径，
rem       从别处启动会安静地新建一个空库——页面上显示没有语料，
rem       其实数据还在 data\app.db 里，只是没读到。
rem    2. 必须用 .venv 里的 python。系统 python 没装依赖。
rem    3. 端口不用每次记。
rem
rem  用法：双击，或 start.bat 8000 指定端口
rem
rem  这里刻意不自动打开浏览器：延迟开浏览器要靠嵌套 start，
rem  在非交互环境（计划任务、IDE 的终端）里会静默失败甚至带崩
rem  整个脚本。地址已经打印在下面，复制一下而已。
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

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 找不到 .venv\Scripts\python.exe
    echo.
    echo 请先建虚拟环境并装依赖：
    echo     python -m venv .venv
    echo     .venv\Scripts\python.exe -m pip install -e ".[milvus,dev]"
    goto :fail
)

if not exist ".env" (
    echo [错误] 找不到 .env
    echo.
    echo 配置样例在 .env.example，复制一份再填 Key：
    echo     copy .env.example .env
    goto :fail
)

if not exist "data\app.db" (
    echo [提示] data\app.db 不存在，这是一个空库。
    echo        先导入文档再提问，例如：
    echo        .venv\Scripts\python.exe -m rag4prefa ingest 工程文档 --project 我的项目
    echo.
)

echo   地址   http://%HOST%:%PORT%
echo   停止   Ctrl+C
echo.

rem 启动前自检：缺 Key 或索引不一致，这里就报出来，
rem 而不是等用户点了提问才在页面上看到 503
.venv\Scripts\python.exe -m rag4prefa doctor >nul 2>&1
if errorlevel 1 (
    echo [注意] 自检未通过，原因如下：
    echo.
    .venv\Scripts\python.exe -m rag4prefa doctor
    echo.
    echo 服务仍会启动，但检索/问答可能不可用。
    echo.
)

.venv\Scripts\python.exe -m rag4prefa serve --host %HOST% --port %PORT%

goto :end

:fail
echo.
pause
endlocal
exit /b 1

:end
endlocal
