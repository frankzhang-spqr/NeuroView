# NeuroView MRI - Production Build Script

Write-Host "`n--- [STEP 1/4] Cleaning previous builds ---" -ForegroundColor Cyan
Remove-Item -Path "dist", "build", "dist-electron" -Recurse -ErrorAction SilentlyContinue

Write-Host "`n--- [STEP 2/4] Building AI Backend (Milestone Tracker) ---" -ForegroundColor Cyan
$pyArgs = @(
    "--noconfirm", "--onedir", "--windowed",
    "--add-data", "app/static;app/static",
    "--add-data", "app/templates;app/templates",
    "--collect-all", "fastapi",
    "--collect-all", "uvicorn",
    "--hidden-import", "uvicorn.loops.auto",
    "--hidden-import", "uvicorn.protocols.http.auto",
    "--hidden-import", "uvicorn.protocols.websockets.auto",
    "--hidden-import", "uvicorn.lifespan.on",
    "--exclude-module", "tensorflow",
    "--exclude-module", "matplotlib",
    "--exclude-module", "notebook",
    "--exclude-module", "jedi",
    "--name", "neuroview_backend",
    "run.py"
)

$process = Start-Process -FilePath "pyinstaller" -ArgumentList $pyArgs -NoNewWindow -PassThru -RedirectStandardOutput "build_log.txt" -RedirectStandardError "build_err.txt"

$milestones = @{
    "Analyzing modules"               = 10
    "Processing standard module hook" = 20
    "Looking for dynamic libraries"   = 30
    "Looking for ctypes DLLs"         = 40
    "Creating base_library.zip"       = 50
    "Building PYZ"                    = 60
    "Building PKG"                    = 70
    "Building EXE"                    = 80
    "Building COLLECT"                = 90
}

while (!$process.HasExited) {
    if (Test-Path "build_err.txt") {
        $lastLines = Get-Content "build_err.txt" -Tail 10
        foreach ($line in $lastLines) {
            foreach ($m in $milestones.Keys) {
                if ($line -like "*$m*") {
                    Write-Progress -Activity "Building AI Backend (5GB Engine)" -Status "Current: $m" -PercentComplete $milestones[$m]
                }
            }
        }
    }
    Start-Sleep -Seconds 2
}
Write-Progress -Activity "Building AI Backend (5GB Engine)" -Completed

Write-Host "`n--- [STEP 3/4] Preparing App Bundle (Live Copy) ---" -ForegroundColor Cyan
$source = "dist/neuroview_backend"
$dest = "build/app_bundle/neuroview_backend"
if (!(Test-Path "build/app_bundle")) { New-Item -ItemType Directory -Path "build/app_bundle" -Force }

$files = Get-ChildItem -Path $source -Recurse
$totalFiles = $files.Count
$counter = 0

foreach ($file in $files) {
    $targetPath = $file.FullName.Replace((Get-Item $source).FullName, (Get-Item "build/app_bundle").FullName + "\neuroview_backend")
    if ($file.PSIsContainer) {
        if (!(Test-Path $targetPath)) { New-Item -ItemType Directory -Path $targetPath -Force | Out-Null }
    }
    else {
        Copy-Item -Path $file.FullName -Destination $targetPath -Force
    }
    $counter++
    if ($counter % 50 -eq 0) {
        $percent = [math]::Round(($counter / $totalFiles) * 100)
        Write-Progress -Activity "Copying 5GB Backend" -Status "$percent% Complete ($counter/$totalFiles)" -PercentComplete $percent
    }
}
Write-Progress -Activity "Copying 5GB Backend" -Completed

Write-Host "`n--- [STEP 4/4] Creating AAA-Style Multi-Part Installer ---" -ForegroundColor Cyan
Write-Host "Packaging data into a separate high-speed archive (like a 300GB game)..." -ForegroundColor Yellow

# 1. Build the unpacked app folder
npm run pack

# 2. Compress the entire win-unpacked folder into a data.7z file
$7z = "C:\Program Files\7-Zip\7z.exe"
if (!(Test-Path $7z)) { $7z = "7z" } # Fallback to path

Write-Host "Compressing 5GB data. This is the heavy lifting..." -ForegroundColor Yellow
& $7z a -t7z "dist-electron\data.7z" ".\dist-electron\win-unpacked\*"

# 3. Create the small, professional Setup files
Copy-Item "install.ps1" "dist-electron\install.ps1"
Copy-Item "Setup.bat" "dist-electron\Setup.bat"
Copy-Item "C:\Program Files\7-Zip\7z.exe" "dist-electron\7z.exe" -ErrorAction SilentlyContinue
Copy-Item "C:\Program Files\7-Zip\7z.dll" "dist-electron\7z.dll" -ErrorAction SilentlyContinue

Write-Host "`n--- ALL STEPS COMPLETE! ---" -ForegroundColor Green
Write-Host "YOUR GAME-STYLE PACKAGE IS READY in .\dist-electron\" -ForegroundColor Gray
Write-Host "1. install.ps1 (The Setup Script - you can make this an EXE later)" -ForegroundColor White
Write-Host "2. data.7z (The 5GB App Data)" -ForegroundColor White
Write-Host "3. 7z.exe/dll (Helper files)" -ForegroundColor White
Write-Host "`nSend people the folder containing these files. They run install.ps1 with PowerShell." -ForegroundColor Yellow
