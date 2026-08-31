# ==============================================================================
# run_vlm_overnight.ps1
# AIC2026 Master Overnight Image Description & Embedding Generation Pipeline
# ==============================================================================

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "🚀 AIC2026: BẮT ĐẦU CHẠY VLM RETRIEVAL-CAPTIONING TOÀN BỘ DATASET" -ForegroundColor Green
Write-Host "Model: Qwen2.5-VL-7B-Instruct (4-bit NF4 | Batch Size = 6 | No TV Logos)" -ForegroundColor Yellow
Write-Host "Card đồ họa: NVIDIA GeForce RTX 5060 Ti (16GB VRAM)" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_ENABLE_HF_TRANSFER = "0"
$PYTHON = "d:\AIC2026\.venv\Scripts\python.exe"

# 1. Chạy sinh mô tả Dense Retrieval cho toàn bộ 88,899 scenes (872 videos)
Write-Host "`n[1/2] Đang chạy sinh mô tả bằng Qwen2.5-VL-7B trên toàn bộ 872 videos..." -ForegroundColor Green
& $PYTHON "d:\AIC2026\tools\vlm_captioning\generate_descriptions.py" --model-id "Qwen/Qwen2.5-VL-7B-Instruct" --quantization 4bit --batch-size 6 --max-new-tokens 85

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Quá trình sinh mô tả gặp lỗi hoặc bị dừng. Bạn có thể chạy lại script này bất cứ lúc nào để tiếp tục từ vị trí đã dừng (Resume)!" -ForegroundColor Red
    exit $LASTEXITCODE
}

# 2. Gộp Shards và Tính toán ma trận BGE-Large-EN embeddings (88,899 x 1024)
Write-Host "`n[2/2] Đang gộp Shards và sinh ma trận Dense BGE Embeddings (88,899 vectors)..." -ForegroundColor Green
& $PYTHON "d:\AIC2026\tools\vlm_captioning\merge_and_embed.py"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Quá trình gộp và sinh embedding gặp lỗi." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n==================================================================" -ForegroundColor Green
Write-Host "🎉 HOÀN THÀNH TOÀN BỘ PIPELINE THÀNH CÔNG RỰC RỠ!" -ForegroundColor Green
Write-Host "File mô tả mới: d:\AIC2026\data\unified\scene_descriptions.parquet" -ForegroundColor Yellow
Write-Host "Ma trận vector mới: d:\AIC2026\data\unified\description_embeddings.npy" -ForegroundColor Yellow
Write-Host "==================================================================" -ForegroundColor Green
