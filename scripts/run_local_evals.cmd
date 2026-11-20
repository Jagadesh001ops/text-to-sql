@echo off
REM Run the retry evaluations + a voting timing check on this machine's GPU, detached
REM from any terminal/editor. Progress is appended to eval\results\local_evals.status.
REM Start with:  powershell -Command "Start-Process cmd.exe -ArgumentList '/c','scripts\run_local_evals.cmd' -WindowStyle Hidden"
cd /d "%~dp0.."
set HF_HUB_OFFLINE=1
set FIRST=eval\results\hazel_finetuned_predictions.jsonl
echo started %date% %time% > eval\results\local_evals.status

venv\Scripts\python.exe eval\run_baseline.py --mode finetuned --retries 2 --retry-mode feedback --first-attempts %FIRST% --tag ft-retry-feedback > eval\results\ft-retry-feedback.log 2>&1
echo feedback done, exit %errorlevel% at %time% >> eval\results\local_evals.status

venv\Scripts\python.exe eval\run_baseline.py --mode finetuned --retries 2 --retry-mode sample --first-attempts %FIRST% --tag ft-retry-sample > eval\results\ft-retry-sample.log 2>&1
echo sample done, exit %errorlevel% at %time% >> eval\results\local_evals.status

venv\Scripts\python.exe eval\run_baseline.py --mode finetuned --limit 10 --vote-samples 8 --first-attempts %FIRST% --tag vote-timing > eval\results\vote-timing.log 2>&1
echo vote-timing done, exit %errorlevel% at %time% >> eval\results\local_evals.status

echo ALL DONE at %time% >> eval\results\local_evals.status
