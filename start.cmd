@echo off
rem Gufo on Windows: pick a model and its features with number keys.
rem   start.cmd          menu        start.cmd -Last     the previous choice again
rem   start.cmd -DryRun  print the command instead of starting the server
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\windows\launcher.ps1" %*
