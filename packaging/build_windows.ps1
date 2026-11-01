# Build Deltachrome for Windows: dist\Deltachrome\ and a zip of it.
#
#     pwsh packaging\build_windows.ps1
#
# Uses whichever `python` is first on PATH, which must be Python 3.13.
# OpenCV is the official wheel: on Windows its FFmpeg is a separate plugin that
# the spec leaves out, so no build of our own is needed here.
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
Set-Location (Join-Path $PSScriptRoot "..")

Remove-Item -Recurse -Force build\release-venv -ErrorAction SilentlyContinue
python -m venv build\release-venv
$V = "build\release-venv\Scripts"
& "$V\python" -m pip install -q --upgrade pip
& "$V\pip" install -q -r packaging\requirements-lock.txt opencv-python-headless==5.0.0.93 pyinstaller==6.22.3
& "$V\pip" install -q --no-deps .

$Version = (& "$V\python" -c "import deltachrome; print(deltachrome.__version__)").Trim()
& "$V\python" packaging\make_icon.py
& "$V\python" packaging\fetch_colprof.py
& "$V\python" packaging\collect_notices.py build\THIRD_PARTY_NOTICES.txt
& "$V\python" -m PyInstaller --noconfirm --log-level WARN `
    --workpath build\pyinstaller --distpath dist packaging\deltachrome.spec

# No FFmpeg may ship: see the spec.
$ffmpeg = Get-ChildItem dist\Deltachrome -Recurse -Filter "*ffmpeg*"
if ($ffmpeg) { throw "FFmpeg found in the bundle: $($ffmpeg.FullName)" }

# The bundle must answer as the source does. Piping makes PowerShell wait for a
# windowed executable and hands it a standard output to write to.
$answer = (& dist\Deltachrome\Deltachrome.exe --cli --version | Out-String).Trim()
if ($answer -ne "deltachrome $Version") { throw "Bundle answered '$answer'" }

# colprof must be where the application looks, and must run. Asked for nothing,
# it prints its usage and exits non-zero.
$colprof = "dist\Deltachrome\_internal\argyll\bin\colprof.exe"
$PSNativeCommandUseErrorActionPreference = $false
$usage = (& $colprof 2>&1 | Out-String)
$PSNativeCommandUseErrorActionPreference = $true
# Its usage exits non-zero on purpose; left standing, that exit code becomes
# the script's own and fails the build after everything succeeded.
$global:LASTEXITCODE = 0
if ($usage -notmatch "Create ICC profile") { throw "Bundled colprof is missing or does not run: $colprof" }

Copy-Item LICENSE dist\Deltachrome\LICENSE.txt
Copy-Item build\THIRD_PARTY_NOTICES.txt dist\Deltachrome\
$zip = "dist\Deltachrome-$Version-windows-x64.zip"
Remove-Item $zip -ErrorAction SilentlyContinue
Compress-Archive -Path dist\Deltachrome -DestinationPath $zip
Write-Host "Built $zip"
