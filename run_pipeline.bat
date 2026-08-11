@echo off
REM Scheduled-task entry point. Appends everything to pipeline.log.
REM
REM %~dp0 is this file's own directory with a trailing backslash. Paths are built
REM from it because Task Scheduler makes no promise about the working directory.

cd /d "%~dp0"
echo. >> pipeline.log
echo ===== %DATE% %TIME% ===== >> pipeline.log
"%~dp0venv\Scripts\python.exe" "%~dp0scraper.py" >> pipeline.log 2>&1
