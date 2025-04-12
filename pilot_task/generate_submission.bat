@echo off
REM -----------------------------------------------------------------
REM generate_submission.bat
REM No args needed. Processes 3 hard‑coded folders and produces:
REM   interactions_run1.json, results_run1.json
REM   interactions_run2.json, results_run2.json
REM   interactions_run3.json, results_run3.json
REM -----------------------------------------------------------------

setlocal EnableDelayedExpansion

set count=1

for %%F in (
  "D:\SRC\DS@GT\eRisk25\erisk-2025\pilot_task\transcripts\Claude-3.7-sonnet"
  "D:\SRC\DS@GT\eRisk25\erisk-2025\pilot_task\transcripts\gemini-2.5-pro-exp-03-25"
  "D:\SRC\DS@GT\eRisk25\erisk-2025\pilot_task\transcripts\gemini-2.0-flash"
) do (
  echo --------------------------------------------------------
  echo Run !count!: Processing %%~F

  echo 1^) Extracting conversations...
  python "D:\SRC\DS@GT\eRisk25\erisk-2025\pilot_task\extract_conversations_json_v2.py" "%%~F" -o "interactions_run!count!.json"
  if errorlevel 1 (
    echo   ERROR: conversation extraction failed for %%~F
  ) else (
    echo   Wrote interactions_run!count!.json
  )

  echo 2^) Extracting BDI scores...
  python "D:\SRC\DS@GT\eRisk25\erisk-2025\pilot_task\extract_bdi_scores_v2.py" "%%~F" -o "results_run!count!.json"
  if errorlevel 1 (
    echo   ERROR: score extraction failed for %%~F
  ) else (
    echo   Wrote results_run!count!.json
  )

  set /a count+=1
)

echo --------------------------------------------------------
echo All done. Processed %count%-1 runs.
endlocal
