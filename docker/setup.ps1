# Speaker-Profiles Docker Setup Script
# This script helps first-time users set up the Docker environment

param(
    [switch]$SkipBuild,
    [switch]$Help
)

# Colors for output
function Write-ColorOutput($ForegroundColor, $Message) {
    Write-Host $Message -ForegroundColor $ForegroundColor
}

function Show-Help {
    Write-ColorOutput "Green" "🐳 Speaker-Profiles Docker Setup"
    Write-ColorOutput "Green" "================================"
    Write-Host ""
    Write-Host "This script helps you set up the Speaker-Profiles Docker environment."
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -SkipBuild    Skip building Docker images"
    Write-Host "  -Help         Show this help message"
    Write-Host ""
    Write-Host "What this script does:"
    Write-Host "  1. Checks Docker installation"
    Write-Host "  2. Creates necessary directories"
    Write-Host "  3. Builds Docker images (unless -SkipBuild)"
    Write-Host "  4. Sets up environment configuration"
    Write-Host "  5. Provides next steps"
}

if ($Help) {
    Show-Help
    exit 0
}

Write-ColorOutput "Green" "🎉 Welcome to Speaker-Profiles Docker Setup!"
Write-ColorOutput "Green" "============================================"

# Step 1: Check prerequisites
Write-ColorOutput "Cyan" "1️⃣ Checking prerequisites..."

# Check if Docker is installed
try {
    $dockerVersion = docker --version
    Write-ColorOutput "Green" "   ✅ Docker detected: $dockerVersion"
} catch {
    Write-ColorOutput "Red" "   ❌ Docker not found!"
    Write-ColorOutput "Yellow" "   Please install Docker Desktop from: https://www.docker.com/products/docker-desktop"
    exit 1
}

# Check if Docker is running
try {
    docker info | Out-Null
    Write-ColorOutput "Green" "   ✅ Docker is running"
} catch {
    Write-ColorOutput "Red" "   ❌ Docker is not running!"
    Write-ColorOutput "Yellow" "   Please start Docker Desktop and run this script again"
    exit 1
}

# Check Docker Compose
try {
    $composeVersion = docker-compose --version
    Write-ColorOutput "Green" "   ✅ Docker Compose detected: $composeVersion"
} catch {
    Write-ColorOutput "Red" "   ❌ Docker Compose not found!"
    Write-ColorOutput "Yellow" "   Please install Docker Compose or update Docker Desktop"
    exit 1
}

# Step 2: Create directories
Write-ColorOutput "Cyan" "2️⃣ Creating directory structure..."

$directories = @("../audio", "../speakers", "../output")
$created = 0

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        try {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-ColorOutput "Green" "   ✅ Created: $dir"
            $created++
        } catch {
            Write-ColorOutput "Red" "   ❌ Failed to create: $dir"
        }
    } else {
        Write-ColorOutput "Yellow" "   📂 Already exists: $dir"
    }
}

if ($created -gt 0) {
    Write-ColorOutput "Green" "   📁 Created $created new directories"
}

# Step 3: Build Docker images (unless skipped)
if (-not $SkipBuild) {
    Write-ColorOutput "Cyan" "3️⃣ Building Docker images..."
    Write-ColorOutput "Yellow" "   This may take 10-15 minutes on first run..."
    
    try {
        # Build GPU version by default
        & .\build.ps1 -Version gpu
        Write-ColorOutput "Green" "   ✅ Docker images built successfully"
    } catch {
        Write-ColorOutput "Red" "   ❌ Failed to build Docker images"
        Write-ColorOutput "Yellow" "   You can try building manually later with: .\build.ps1"
    }
} else {
    Write-ColorOutput "Yellow" "3️⃣ Skipping Docker image build (as requested)"
}

# Step 4: Environment configuration
Write-ColorOutput "Cyan" "4️⃣ Setting up environment configuration..."

if (Test-Path ".env.docker") {
    Write-ColorOutput "Green" "   ✅ Environment template already exists"
    
    # Check if HuggingFace token is set
    $envContent = Get-Content ".env.docker" -Raw
    if ($envContent -match "your_hugging_face_token_here") {
        Write-ColorOutput "Yellow" "   ⚠️  HuggingFace token needs to be configured"
        $needsTokenConfig = $true
    } else {
        Write-ColorOutput "Green" "   ✅ HuggingFace token appears to be configured"
        $needsTokenConfig = $false
    }
} else {
    Write-ColorOutput "Red" "   ❌ Environment template not found!"
    exit 1
}

# Step 5: Final instructions
Write-ColorOutput "Green" "`n🎯 Setup Summary"
Write-ColorOutput "Green" "================"

Write-ColorOutput "Green" "✅ Docker environment is ready!"
Write-ColorOutput "Cyan" "📋 Directory structure created"
Write-ColorOutput "Cyan" "🐳 Docker images available (if built)"
Write-ColorOutput "Cyan" "⚙️  Environment template configured"

Write-ColorOutput "Yellow" "`n🚀 Next Steps:"

if ($needsTokenConfig) {
    Write-ColorOutput "Red" "1. 🔑 IMPORTANT: Configure your HuggingFace token"
    Write-ColorOutput "Yellow" "   - Visit: https://huggingface.co/settings/tokens"
    Write-ColorOutput "Yellow" "   - Create a token (read permissions sufficient)"
    Write-ColorOutput "Yellow" "   - Edit .env.docker and replace 'your_hugging_face_token_here'"
    Write-ColorOutput "Yellow" "   - Save the file"
    Write-Host ""
}

Write-ColorOutput "Green" "2. 🎵 Add audio files to process"
Write-ColorOutput "Yellow" "   - Copy your audio files to: ../audio/"
Write-ColorOutput "Yellow" "   - Supported formats: .wav, .mp3, .flac"

Write-ColorOutput "Green" "3. 🏃 Run the container"
Write-ColorOutput "Yellow" "   - Interactive mode: .\run.ps1 -Interactive"
Write-ColorOutput "Yellow" "   - Background mode: .\run.ps1 -Background"
Write-ColorOutput "Yellow" "   - Check status: .\run.ps1 -Status"

Write-ColorOutput "Green" "4. 📚 Learn more"
Write-ColorOutput "Yellow" "   - Read README.md for detailed usage"
Write-ColorOutput "Yellow" "   - Check TROUBLESHOOTING.md for common issues"

if ($SkipBuild) {
    Write-ColorOutput "Yellow" "`n⚠️  Remember to build Docker images before running:"
    Write-ColorOutput "Yellow" "   .\build.ps1"
}

Write-ColorOutput "Green" "`n🎉 Setup completed! Happy speaker profiling!"

# Offer to open files for editing
if ($needsTokenConfig) {
    Write-Host ""
    $openEnv = Read-Host "Would you like to open .env.docker for editing now? (y/n)"
    if ($openEnv -eq 'y' -or $openEnv -eq 'Y') {
        try {
            Start-Process notepad.exe ".env.docker"
        } catch {
            Write-ColorOutput "Yellow" "Please edit .env.docker manually"
        }
    }
}