# Speaker-Profiles Docker Run Script for Windows
# This script runs the Speaker-Profiles Docker containers with proper setup

param(
    [string]$Version = "gpu",           # Options: gpu, cpu, dev
    [string]$AudioFile = "",            # Optional: specific audio file to process
    [switch]$Interactive,               # Run in interactive mode
    [switch]$Background,                # Run in background (daemon mode)
    [switch]$Setup,                     # Run initial setup
    [switch]$Status                     # Show container status
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Colors for output
function Write-ColorOutput($ForegroundColor, $Message) {
    Write-Host $Message -ForegroundColor $ForegroundColor
}

Write-ColorOutput "Green" "🚀 Speaker-Profiles Docker Run Script"
Write-ColorOutput "Green" "====================================="

# Function to check if Docker is running
function Test-DockerRunning {
    try {
        docker info | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# Function to create necessary directories
function Initialize-Directories {
    Write-ColorOutput "Cyan" "📁 Creating necessary directories..."
    
    $directories = @("../audio", "../speakers", "../output")
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-ColorOutput "Green" "✅ Created directory: $dir"
        } else {
            Write-ColorOutput "Yellow" "📂 Directory exists: $dir"
        }
    }
}

# Function to check environment configuration
function Test-EnvironmentConfig {
    Write-ColorOutput "Cyan" "🔍 Checking environment configuration..."
    
    if (-not (Test-Path ".env.docker")) {
        Write-ColorOutput "Red" "❌ .env.docker not found"
        Write-ColorOutput "Yellow" "Creating .env.docker from template..."
        
        if (Test-Path ".env.docker") {
            Copy-Item ".env.docker" ".env.docker.bak"
        }
        
        Write-ColorOutput "Yellow" "⚠️  Please edit .env.docker and set your HUGGING_FACE_ACCESS_TOKEN"
        return $false
    }
    
    $envContent = Get-Content ".env.docker" -Raw
    if ($envContent -match "your_hugging_face_token_here") {
        Write-ColorOutput "Red" "❌ Please set your HUGGING_FACE_ACCESS_TOKEN in .env.docker"
        Write-ColorOutput "Yellow" "Get your token from: https://huggingface.co/settings/tokens"
        return $false
    }
    
    Write-ColorOutput "Green" "✅ Environment configuration looks good"
    return $true
}

# Function to show container status
function Show-ContainerStatus {
    Write-ColorOutput "Cyan" "📊 Container Status"
    Write-ColorOutput "Cyan" "=================="
    
    # Show running containers
    $runningContainers = docker ps --filter "name=speaker-profiles" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    if ($runningContainers) {
        Write-ColorOutput "Green" "🟢 Running containers:"
        Write-Host $runningContainers
    } else {
        Write-ColorOutput "Yellow" "⚪ No Speaker-Profiles containers are currently running"
    }
    
    # Show available images
    $availableImages = docker images speaker-profiles --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
    if ($availableImages) {
        Write-ColorOutput "Cyan" "`n📋 Available images:"
        Write-Host $availableImages
    } else {
        Write-ColorOutput "Red" "❌ No Speaker-Profiles images found. Please run build script first."
    }
}

# Main execution starts here

# Handle status request
if ($Status) {
    Show-ContainerStatus
    exit 0
}

# Check if Docker is running
if (-not (Test-DockerRunning)) {
    Write-ColorOutput "Red" "❌ Docker is not running"
    Write-ColorOutput "Yellow" "Please start Docker Desktop and try again"
    exit 1
}

# Check if we're in the correct directory
if (-not (Test-Path "docker-compose.yml")) {
    Write-ColorOutput "Red" "❌ docker-compose.yml not found in current directory"
    Write-ColorOutput "Yellow" "Please run this script from the docker folder"
    exit 1
}

# Run setup if requested
if ($Setup) {
    Write-ColorOutput "Cyan" "🔧 Running initial setup..."
    Initialize-Directories
    
    if (-not (Test-EnvironmentConfig)) {
        Write-ColorOutput "Yellow" "⚠️  Setup completed, but environment configuration needs attention"
        exit 1
    }
    
    Write-ColorOutput "Green" "✅ Setup completed successfully!"
    exit 0
}

# Check environment configuration
if (-not (Test-EnvironmentConfig)) {
    Write-ColorOutput "Yellow" "Run with -Setup flag to initialize the environment"
    exit 1
}

# Ensure directories exist
Initialize-Directories

# Validate version
$validVersions = @("gpu", "cpu", "dev")
if ($Version -notin $validVersions) {
    Write-ColorOutput "Red" "❌ Invalid version: $Version"
    Write-ColorOutput "Yellow" "Valid versions: $($validVersions -join ', ')"
    exit 1
}

# Determine service name
$serviceName = "speaker-profiles-$Version"

# Check if image exists
$imageExists = docker images "speaker-profiles:$Version" --format "{{.Repository}}" | Where-Object { $_ -eq "speaker-profiles" }
if (-not $imageExists) {
    Write-ColorOutput "Red" "❌ Image speaker-profiles:$Version not found"
    Write-ColorOutput "Yellow" "Please run the build script first: .\build.ps1 -Version $Version"
    exit 1
}

Write-ColorOutput "Cyan" "🐳 Starting Speaker-Profiles ($Version version)..."

# Prepare docker-compose command
$composeArgs = @("up")

if ($Background) {
    $composeArgs += "-d"
}

if (-not $Interactive -and -not $Background) {
    # Default to interactive mode with proper TTY
    $composeArgs += "--remove-orphans"
}

$composeArgs += $serviceName

# Execute docker-compose
try {
    Write-ColorOutput "Green" "🚀 Running: docker-compose $($composeArgs -join ' ')"
    
    if ($Interactive) {
        # Run interactively
        & docker-compose $composeArgs
    } elseif ($Background) {
        # Run in background
        & docker-compose $composeArgs
        Write-ColorOutput "Green" "✅ Container started in background"
        Write-ColorOutput "Cyan" "📊 Check status with: .\run.ps1 -Status"
        Write-ColorOutput "Cyan" "🔗 Access container: docker exec -it $serviceName bash"
    } else {
        # Run with proper handling for audio file processing
        if ($AudioFile) {
            Write-ColorOutput "Cyan" "🎵 Processing audio file: $AudioFile"
            # TODO: Implement specific audio file processing
        }
        & docker-compose $composeArgs
    }
    
    Write-ColorOutput "Green" "✅ Container operation completed"
}
catch {
    Write-ColorOutput "Red" "❌ Failed to run container: $($_.Exception.Message)"
    Write-ColorOutput "Yellow" "💡 Try running: docker-compose logs $serviceName"
    exit 1
}

Write-ColorOutput "Green" "🎉 Run script completed!"