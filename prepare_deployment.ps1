# CXR-Assistant - Hugging Face Deployment Preparation Script
# Run this script before deploying to Hugging Face Spaces

Write-Host "🫁 CXR-Assistant Deployment Preparation" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check if model file exists
Write-Host "Step 1: Checking model file..." -ForegroundColor Yellow
$modelSource = "..\models\colab_clahe_eff_final.keras"
$modelDest = "models\colab_clahe_eff_final.keras"

if (Test-Path $modelSource) {
    Write-Host "✓ Found model at parent directory" -ForegroundColor Green
    
    if (-not (Test-Path $modelDest)) {
        Write-Host "  Copying model to deployment directory..." -ForegroundColor Yellow
        Copy-Item $modelSource -Destination $modelDest
        Write-Host "✓ Model copied successfully" -ForegroundColor Green
    } else {
        Write-Host "✓ Model already in deployment directory" -ForegroundColor Green
    }
} elseif (Test-Path $modelDest) {
    Write-Host "✓ Model found in deployment directory" -ForegroundColor Green
} else {
    Write-Host "✗ ERROR: Model file not found!" -ForegroundColor Red
    Write-Host "  Please copy your trained model to: $modelDest" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 2: Check Git installation
Write-Host "Step 2: Checking Git installation..." -ForegroundColor Yellow
try {
    $gitVersion = git --version
    Write-Host "✓ Git installed: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ ERROR: Git not installed!" -ForegroundColor Red
    Write-Host "  Download from: https://git-scm.com/" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 3: Check Git LFS installation
Write-Host "Step 3: Checking Git LFS installation..." -ForegroundColor Yellow
try {
    $lfsVersion = git lfs version
    Write-Host "✓ Git LFS installed: $lfsVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ WARNING: Git LFS not installed!" -ForegroundColor Yellow
    Write-Host "  Installing Git LFS is required for the model file" -ForegroundColor Yellow
    Write-Host "  Run: git lfs install" -ForegroundColor Yellow
}
Write-Host ""

# Step 4: Check required files
Write-Host "Step 4: Checking required files..." -ForegroundColor Yellow
$requiredFiles = @(
    "app.py",
    "requirements.txt",
    "README.md",
    ".gitattributes",
    ".gitignore",
    "models\colab_clahe_eff_final.keras"
)

$allFilesPresent = $true
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "  ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $file MISSING" -ForegroundColor Red
        $allFilesPresent = $false
    }
}

if (-not $allFilesPresent) {
    Write-Host ""
    Write-Host "✗ Some required files are missing!" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 5: Check Python dependencies
Write-Host "Step 5: Verifying requirements.txt..." -ForegroundColor Yellow
if (Test-Path "requirements.txt") {
    $content = Get-Content "requirements.txt"
    $requiredPackages = @("streamlit", "tensorflow", "numpy", "opencv-python-headless", "pillow", "matplotlib", "pandas")
    
    foreach ($pkg in $requiredPackages) {
        if ($content -match $pkg) {
            Write-Host "  ✓ $pkg" -ForegroundColor Green
        } else {
            Write-Host "  ✗ $pkg missing" -ForegroundColor Red
        }
    }
}
Write-Host ""

# Step 6: Display next steps
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✓ Preparation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📝 Next Steps for Deployment:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Create a Hugging Face Space:" -ForegroundColor White
Write-Host "   → Go to: https://huggingface.co/new-space" -ForegroundColor Gray
Write-Host "   → Name: CXR-Assistant" -ForegroundColor Gray
Write-Host "   → SDK: Streamlit" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Initialize Git and push to Hugging Face:" -ForegroundColor White
Write-Host "   git init" -ForegroundColor Gray
Write-Host "   git lfs install" -ForegroundColor Gray
Write-Host "   git lfs track `"*.keras`"" -ForegroundColor Gray
Write-Host "   git add ." -ForegroundColor Gray
Write-Host "   git commit -m `"Initial commit: CXR-Assistant v1.0`"" -ForegroundColor Gray
Write-Host "   git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/CXR-Assistant" -ForegroundColor Gray
Write-Host "   git branch -M main" -ForegroundColor Gray
Write-Host "   git push -u origin main" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Wait for Hugging Face to build your Space (3-5 minutes)" -ForegroundColor White
Write-Host ""
Write-Host "📖 For detailed instructions, see: DEPLOYMENT_GUIDE.md" -ForegroundColor Cyan
Write-Host ""
Write-Host "🎉 Good luck with your deployment!" -ForegroundColor Green

