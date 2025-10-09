# Speaker-Profiles Docker Build Script for Windows
# This script builds the Docker images for the Speaker-Profiles project

param(
    [string]$Version = "gpu",  # Options: gpu, cpu, all
    [switch]$NoCache,
    [switch]$Verbose
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Colors for output
function Write-ColorOutput($ForegroundColor, $Message) {
    Write-Host $Message -ForegroundColor $ForegroundColor
}

Write-ColorOutput "Green" "🐳 Speaker-Profiles Docker Build Script"
Write-ColorOutput "Green" "======================================"

# Check if Docker is installed and running
try {
    $dockerVersion = docker --version
    Write-ColorOutput "Cyan" "✅ Docker detected: $dockerVersion"
}
catch {
    Write-ColorOutput "Red" "❌ Docker is not installed or not running"
    Write-ColorOutput "Yellow" "Please install Docker Desktop and ensure it's running"
    exit 1
}

# Check if we're in the correct directory
if (-not (Test-Path "Dockerfile")) {
    Write-ColorOutput "Red" "❌ Dockerfile not found in current directory"
    Write-ColorOutput "Yellow" "Please run this script from the docker folder"
    exit 1
}

# Prepare build arguments
$buildArgs = @()
if ($NoCache) {
    $buildArgs += "--no-cache"
}
if ($Verbose) {
    $buildArgs += "--progress=plain"
}

# Function to build a Docker image
function Build-DockerImage {
    param(
        [string]$ImageName,
        [string]$DockerfileName,
        [string]$Description
    )
    
    Write-ColorOutput "Yellow" "🏗️  Building $Description..."
    Write-ColorOutput "Cyan" "Image: $ImageName"
    Write-ColorOutput "Cyan" "Dockerfile: $DockerfileName"
    
    $buildCommand = @("build") + $buildArgs + @("-f", $DockerfileName, "-t", $ImageName, ".")
    
    try {
        & docker $buildCommand
        Write-ColorOutput "Green" "✅ Successfully built $ImageName"
    }
    catch {
        Write-ColorOutput "Red" "❌ Failed to build $ImageName"
        Write-ColorOutput "Red" $_.Exception.Message
        return $false
    }
    
    return $true
}

# Main build logic
$success = $true

switch ($Version.ToLower()) {
    "gpu" {
        Write-ColorOutput "Cyan" "🚀 Building GPU version only..."
        $success = Build-DockerImage "speaker-profiles:gpu" "Dockerfile" "GPU-enabled version"
    }
    "cpu" {
        Write-ColorOutput "Cyan" "🖥️  Building CPU version only..."
        $success = Build-DockerImage "speaker-profiles:cpu" "Dockerfile.cpu" "CPU-only version"
    }
    "all" {
        Write-ColorOutput "Cyan" "🔄 Building all versions..."
        
        # Build GPU version
        $gpuSuccess = Build-DockerImage "speaker-profiles:gpu" "Dockerfile" "GPU-enabled version"
        
        # Build CPU version
        $cpuSuccess = Build-DockerImage "speaker-profiles:cpu" "Dockerfile.cpu" "CPU-only version"
        
        $success = $gpuSuccess -and $cpuSuccess
    }
    default {
        Write-ColorOutput "Red" "❌ Invalid version specified: $Version"
        Write-ColorOutput "Yellow" "Valid options: gpu, cpu, all"
        exit 1
    }
}

# Show results
Write-ColorOutput "Green" "`n🎯 Build Summary"
Write-ColorOutput "Green" "==============="

if ($success) {
    Write-ColorOutput "Green" "✅ All builds completed successfully!"
    
    # Show built images
    Write-ColorOutput "Cyan" "`n📋 Available images:"
    docker images speaker-profiles --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    
    Write-ColorOutput "Green" "`n🚀 Next steps:"
    Write-ColorOutput "Yellow" "1. Copy .env.docker and set your HUGGING_FACE_ACCESS_TOKEN"
    Write-ColorOutput "Yellow" "2. Create directories: mkdir ../audio ../speakers ../output"
    Write-ColorOutput "Yellow" "3. Run with: docker-compose up speaker-profiles-gpu"
    Write-ColorOutput "Yellow" "   Or CPU version: docker-compose up speaker-profiles-cpu"
    
} else {
    Write-ColorOutput "Red" "❌ One or more builds failed!"
    exit 1
}

Write-ColorOutput "Green" "`n🎉 Build script completed!"