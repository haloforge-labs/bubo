Bubo for Windows
================

You need:
  - Windows 11, an AMD Strix Halo (gfx1151) GPU and a current AMD Adrenalin driver.
  - The Microsoft Visual C++ x64 runtime: https://aka.ms/vs/17/release/vc_redist.x64.exe
  - For the default model, about 113 GiB of free disk space. It downloads 8 files, about 108.6 GiB in total.

Open the app: double-click start.cmd (or bin\Bubo.Control.Web.exe).
The app downloads the AMD files and a model, and starts the server.

Or use the command line. Open a terminal in the bin folder:
  .\bubo.exe prepare   Shows the download plan. Type y to download the AMD files and the default model.
  .\bubo.exe serve     Starts the server at http://127.0.0.1:8080/v1. Press Ctrl+C to stop it.
  .\bubo.exe --help    Shows all commands.

Not enough space on C:? Add --output E:\Bubo to bubo prepare.
Behind a proxy? Set HTTPS_PROXY=http://host:port before you run bubo prepare.
