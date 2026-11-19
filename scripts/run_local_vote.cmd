@echo off
REM Full 8-candidate voting evaluation on this machine's GPU, detached from any terminal.
REM Reuses Hazel's greedy answers as candidate #1 (greedy decoding is deterministic).
REM Start with:  powershell -Command "Start-Process cmd.exe -ArgumentList '/c','scripts\run_local_vote.cmd' -WindowStyle Hidden"
cd /d "%~dp0.."
set HF_HUB_OFFLINE=1
echo started %date% %time% > eval\results\local_vote.status
venv\Scripts\python.exe eval\run_baseline.py --mode finetuned --vote-samples 8 --first-attempts eval\results\hazel_finetuned_predictions.jsonl --tag ft-vote8 > eval\results\ft-vote8.log 2>&1
echo vote8 done, exit %errorlevel% at %time% >> eval\results\local_vote.status
