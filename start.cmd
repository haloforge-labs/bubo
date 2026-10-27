@echo off
rem Bubo for Windows.
rem   start.cmd            open the app (bin\Bubo.Control.Web.exe)
rem   start.cmd [OPTIONS]  in a folder without the app: bin\bubo.exe serve [OPTIONS]
setlocal
set "BUBO_VC_CHECK=%~dp0tools\windows\vc-runtime.ps1"
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -Command ". $env:BUBO_VC_CHECK; if (-not (Confirm-BuboVcRuntime)) { exit 4 }"
if errorlevel 1 (
  pause
  exit /b 4
)
if exist "%~dp0bin\Bubo.Control.Web.exe" (
  start "" "%~dp0bin\Bubo.Control.Web.exe"
  exit /b 0
)
"%~dp0bin\bubo.exe" serve %*
exit /b %errorlevel%
