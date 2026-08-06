@echo off
REM Entry point for the scheduled task; appends all output to pipeline.log.
REM
REM %~dp0 expands to this file's own directory with a trailing backslash. Paths
REM are built from it rather than assumed relative, because Task Scheduler does
REM not guarantee the working directory it supplies.

cd /d "%~dp0"
echo. >> pipeline.log
echo ===== %DATE% %TIME% ===== >> pipeline.log
"%~dp0venv\Scripts\python.exe" "%~dp0scraper.py" >> pipeline.log 2>&1
