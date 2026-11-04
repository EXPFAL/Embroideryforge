$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'D:\WORK\TEST\Project 1'
if (-not (Test-Path -LiteralPath '.\.venv\Scripts\python.exe')) {
    py -3 -m venv .venv
}
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
& '.\.venv\Scripts\python.exe' -m streamlit run '.\app\streamlit_app.py'
