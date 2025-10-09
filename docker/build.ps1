# Build script for Speaker-Profiles Docker images (PowerShell)

param(
    [switch]$GpuOnly,
    [switch]$CpuOnly,
    [string]$Tag = "speaker-profiles",
    [switch]$Push,
    [string[]]$BuildArg = @(),
    [switch]$Help
)

# Colors for output
$Red = "Red"
$Green = "Green"
$Yellow = "Yellow"
$Blue = "Blue"

function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $Red
}

function Show-Usage {
    Write-Host "Usage: .\build.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Build Docker images for Speaker-Profiles"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -GpuOnly            Build only GPU version"
    Write-Host "  -CpuOnly            Build only CPU version"
    Write-Host "  -Tag PREFIX         Tag prefix (default: speaker-profiles)"
    Write-Host "  -Push               Push images to registry after building"
    Write-Host "  -BuildArg ARG       Pass build argument to docker build"
    Write-Host "  -Help               Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\build.ps1                                    # Build both GPU and CPU versions"
    Write-Host "  .\build.ps1 -GpuOnly                          # Build only GPU version"
    Write-Host "  .\build.ps1 -CpuOnly                          # Build only CPU version"
    Write-Host "  .\build.ps1 -Tag 'myrepo/speaker-profiles' -Push"
}

# Show help if requested
if ($Help) {
    Show-Usage
    exit 0
}

# Set build flags
$BuildGpu = $true
$BuildCpu = $true

if ($GpuOnly) {
    $BuildCpu = $false
}
if ($CpuOnly) {
    $BuildGpu = $false
}

# Check if we're in the docker directory
if (!(Test-Path "Dockerfile") -or !(Test-Path "docker-compose.yml")) {
    Write-Error "Please run this script from the docker/ directory"
    exit 1
}

Write-Status "Building Speaker-Profiles Docker images..."
Write-Status "GPU version: $BuildGpu"
Write-Status "CPU version: $BuildCpu"
Write-Status "Tag prefix: $Tag"

# Prepare build arguments
$BuildArgString = ""
if ($BuildArg.Count -gt 0) {
    foreach ($arg in $BuildArg) {
        $BuildArgString += "--build-arg $arg "
    }
}

# Build GPU version
if ($BuildGpu) {
    Write-Status "Building GPU version..."
    
    $buildCmd = "docker build $BuildArgString -t ${Tag}:latest -f Dockerfile ."
    Write-Status "Executing: $buildCmd"
    
    $result = Invoke-Expression $buildCmd
    if ($LASTEXITCODE -eq 0) {
        Write-Success "GPU version built successfully: ${Tag}:latest"
        
        if ($Push) {
            Write-Status "Pushing GPU version..."
            docker push "${Tag}:latest"
            if ($LASTEXITCODE -eq 0) {
                Write-Success "GPU version pushed successfully"
            } else {
                Write-Error "Failed to push GPU version"
                exit 1
            }
        }
    } else {
        Write-Error "Failed to build GPU version"
        exit 1
    }
}

# Build CPU version
if ($BuildCpu) {
    Write-Status "Building CPU version..."
    
    $buildCmd = "docker build $BuildArgString -t ${Tag}:latest-cpu -f Dockerfile.cpu ."
    Write-Status "Executing: $buildCmd"
    
    $result = Invoke-Expression $buildCmd
    if ($LASTEXITCODE -eq 0) {
        Write-Success "CPU version built successfully: ${Tag}:latest-cpu"
        
        if ($Push) {
            Write-Status "Pushing CPU version..."
            docker push "${Tag}:latest-cpu"
            if ($LASTEXITCODE -eq 0) {
                Write-Success "CPU version pushed successfully"
            } else {
                Write-Error "Failed to push CPU version"
                exit 1
            }
        }
    } else {
        Write-Error "Failed to build CPU version"
        exit 1
    }
}

Write-Success "Build process completed!"

# Show next steps
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Test the images:"
Write-Host "     docker run --rm ${Tag}:latest python --version"
Write-Host "     docker run --rm ${Tag}:latest-cpu python --version"
Write-Host ""
Write-Host "  2. Run with docker-compose:"
Write-Host "     docker-compose up"
Write-Host ""
Write-Host "  3. Process an audio file:"
Write-Host "     docker-compose exec speaker-profiles python auto_speaker_tagging_system.py /app/audio_files/your_audio.wav"