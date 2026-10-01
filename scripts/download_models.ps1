# Model Downloader for PaddleModels
# Downloads required YOLO and Paddle weights

$ModelDir = "models/weights"
if (!(Test-Path $ModelDir)) {
    New-Item -ItemType Directory -Path $ModelDir
}

Write-Host "[*] Checking for YOLO layout model..."
$YoloPath = Join-Path $ModelDir "yolov10s_best.pt"
if (!(Test-Path $YoloPath)) {
    Write-Host "[!] YOLO model not found. Please place yolov10s_best.pt in $ModelDir"
    # Placeholder for actual URL if available
    # Invoke-WebRequest -Uri "https://example.com/yolov10s_best.pt" -OutFile $YoloPath
} else {
    Write-Host "[OK] YOLO model exists."
}

Write-Host "[*] Checking for PaddleOCR models..."
# PaddleOCR usually downloads models automatically to ~/.paddleocr
# But we can verify or provide manual download logic here if needed.

Write-Host "[OK] Model check completed."
