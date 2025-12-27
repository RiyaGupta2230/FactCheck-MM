#!/bin/bash
###############################################################################
# FactCheck-MM Production Environment Setup Script
#
# Sets up a production-ready Python environment with all dependencies
# for deploying the FactCheck-MM API. Handles both CPU-only (MacBook M2)
# and GPU-enabled (RTX 2050) environments.
#
# Usage:
#   bash deployment/scripts/setup_production.sh [--cpu-only] [--gpu]
#
# Options:
#   --cpu-only    Force CPU-only installation (skip CUDA)
#   --gpu         Force GPU installation with CUDA support
#   --help        Display this help message
#
# Requirements:
#   - Python 3.10 or higher
#   - pip package manager
#   - (Optional) CUDA 11.8+ for GPU support
###############################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_MIN_VERSION="3.10"
VENV_DIR="venv"
REQUIREMENTS_FILE="deployment/docker/requirements-prod.txt"
FORCE_CPU=false
FORCE_GPU=false

###############################################################################
# Helper Functions
###############################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo "============================================="
    echo "$1"
    echo "============================================="
    echo ""
}

check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

version_compare() {
    # Compare two version strings
    # Returns 0 if $1 >= $2, 1 otherwise
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

###############################################################################
# Argument Parsing
###############################################################################

show_help() {
    grep "^#" "$0" | grep -v "#!/bin/bash" | sed 's/^# //' | sed 's/^#//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu-only)
            FORCE_CPU=true
            shift
            ;;
        --gpu)
            FORCE_GPU=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

###############################################################################
# Main Setup Process
###############################################################################

main() {
    print_header "FactCheck-MM Production Environment Setup"
    
    log_info "Starting production environment setup..."
    log_info "Target: CPU-only: $FORCE_CPU, GPU: $FORCE_GPU"
    
    # Step 1: Check Python version
    check_python_version
    
    # Step 2: Check requirements file
    check_requirements_file
    
    # Step 3: Create virtual environment
    create_virtual_environment
    
    # Step 4: Activate virtual environment
    activate_virtual_environment
    
    # Step 5: Upgrade pip
    upgrade_pip
    
    # Step 6: Install dependencies
    install_dependencies
    
    # Step 7: Verify installation
    verify_installation
    
    # Step 8: Check GPU availability
    check_gpu_availability
    
    # Step 9: Final summary
    print_summary
    
    log_success "Production environment setup completed successfully!"
}

###############################################################################
# Step 1: Check Python Version
###############################################################################

check_python_version() {
    print_header "Step 1: Checking Python Version"
    
    if ! check_command python3; then
        log_error "Python 3 is not installed"
        log_error "Please install Python 3.10 or higher"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    log_info "Found Python version: $PYTHON_VERSION"
    
    if ! version_compare "$PYTHON_VERSION" "$PYTHON_MIN_VERSION"; then
        log_error "Python version $PYTHON_VERSION is below minimum required version $PYTHON_MIN_VERSION"
        log_error "Please upgrade Python to version $PYTHON_MIN_VERSION or higher"
        exit 1
    fi
    
    log_success "Python version $PYTHON_VERSION meets requirements (>= $PYTHON_MIN_VERSION)"
    
    # Store python command
    PYTHON_CMD=$(which python3)
    log_info "Using Python: $PYTHON_CMD"
}

###############################################################################
# Step 2: Check Requirements File
###############################################################################

check_requirements_file() {
    print_header "Step 2: Checking Requirements File"
    
    if [ ! -f "$REQUIREMENTS_FILE" ]; then
        log_error "Requirements file not found: $REQUIREMENTS_FILE"
        log_error "Please ensure you're running this script from the project root"
        exit 1
    fi
    
    log_success "Requirements file found: $REQUIREMENTS_FILE"
    
    # Count dependencies
    DEP_COUNT=$(grep -v "^#" "$REQUIREMENTS_FILE" | grep -v "^$" | wc -l)
    log_info "Found $DEP_COUNT production dependencies"
}

###############################################################################
# Step 3: Create Virtual Environment
###############################################################################

create_virtual_environment() {
    print_header "Step 3: Creating Virtual Environment"
    
    if [ -d "$VENV_DIR" ]; then
        log_warning "Virtual environment already exists at: $VENV_DIR"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Removing existing virtual environment..."
            rm -rf "$VENV_DIR"
        else
            log_info "Keeping existing virtual environment"
            return 0
        fi
    fi
    
    log_info "Creating virtual environment at: $VENV_DIR"
    $PYTHON_CMD -m venv "$VENV_DIR"
    
    if [ $? -eq 0 ]; then
        log_success "Virtual environment created successfully"
    else
        log_error "Failed to create virtual environment"
        exit 1
    fi
}

###############################################################################
# Step 4: Activate Virtual Environment
###############################################################################

activate_virtual_environment() {
    print_header "Step 4: Activating Virtual Environment"
    
    # Determine activation script based on OS
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        ACTIVATE_SCRIPT="$VENV_DIR/Scripts/activate"
    else
        ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"
    fi
    
    if [ ! -f "$ACTIVATE_SCRIPT" ]; then
        log_error "Activation script not found: $ACTIVATE_SCRIPT"
        exit 1
    fi
    
    log_info "Activating virtual environment..."
    source "$ACTIVATE_SCRIPT"
    
    # Verify activation
    ACTIVE_PYTHON=$(which python)
    if [[ "$ACTIVE_PYTHON" == *"$VENV_DIR"* ]]; then
        log_success "Virtual environment activated"
        log_info "Active Python: $ACTIVE_PYTHON"
    else
        log_error "Failed to activate virtual environment"
        exit 1
    fi
}

###############################################################################
# Step 5: Upgrade pip
###############################################################################

upgrade_pip() {
    print_header "Step 5: Upgrading pip"
    
    log_info "Current pip version:"
    pip --version
    
    log_info "Upgrading pip, setuptools, and wheel..."
    pip install --upgrade pip setuptools wheel --quiet
    
    if [ $? -eq 0 ]; then
        log_success "pip upgraded successfully"
        log_info "New pip version:"
        pip --version
    else
        log_warning "pip upgrade had issues, but continuing..."
    fi
}

###############################################################################
# Step 6: Install Dependencies
###############################################################################

install_dependencies() {
    print_header "Step 6: Installing Production Dependencies"
    
    log_info "Installing dependencies from: $REQUIREMENTS_FILE"
    log_warning "This may take several minutes..."
    
    # Determine PyTorch installation based on flags and system
    if [ "$FORCE_CPU" = true ]; then
        log_info "Installing CPU-only PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    elif [ "$FORCE_GPU" = true ]; then
        log_info "Installing GPU-enabled PyTorch (CUDA 11.8)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        log_info "Auto-detecting PyTorch requirements..."
        # Check if NVIDIA GPU is available
        if check_command nvidia-smi; then
            log_info "NVIDIA GPU detected, installing CUDA-enabled PyTorch..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        else
            log_info "No NVIDIA GPU detected, installing CPU-only PyTorch..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        fi
    fi
    
    # Install remaining dependencies
    log_info "Installing remaining production dependencies..."
    
    # Filter out torch-related packages from requirements file
    TEMP_REQ=$(mktemp)
    grep -v "^torch" "$REQUIREMENTS_FILE" > "$TEMP_REQ"
    
    pip install -r "$TEMP_REQ" --quiet
    
    if [ $? -eq 0 ]; then
        log_success "All dependencies installed successfully"
    else
        log_error "Failed to install some dependencies"
        rm -f "$TEMP_REQ"
        exit 1
    fi
    
    rm -f "$TEMP_REQ"
    
    # Install project in editable mode
    log_info "Installing FactCheck-MM package in editable mode..."
    pip install -e . --quiet
    
    if [ $? -eq 0 ]; then
        log_success "FactCheck-MM package installed"
    else
        log_warning "Failed to install FactCheck-MM package (may not have setup.py)"
    fi
}

###############################################################################
# Step 7: Verify Installation
###############################################################################

verify_installation() {
    print_header "Step 7: Verifying Installation"
    
    log_info "Verifying critical packages..."
    
    # Check FastAPI
    python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')" 2>/dev/null
    if [ $? -eq 0 ]; then
        log_success "✓ FastAPI installed"
    else
        log_error "✗ FastAPI not found"
    fi
    
    # Check PyTorch
    python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null
    if [ $? -eq 0 ]; then
        log_success "✓ PyTorch installed"
    else
        log_error "✗ PyTorch not found"
    fi
    
    # Check Transformers
    python -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null
    if [ $? -eq 0 ]; then
        log_success "✓ Transformers installed"
    else
        log_error "✗ Transformers not found"
    fi
    
    # Check other critical packages
    PACKAGES=("numpy" "pandas" "scikit-learn" "psutil")
    
    for pkg in "${PACKAGES[@]}"; do
        python -c "import $pkg" 2>/dev/null
        if [ $? -eq 0 ]; then
            log_success "✓ $pkg installed"
        else
            log_warning "✗ $pkg not found"
        fi
    done
}

###############################################################################
# Step 8: Check GPU Availability
###############################################################################

check_gpu_availability() {
    print_header "Step 8: Checking GPU Availability"
    
    # Check NVIDIA GPU
    if check_command nvidia-smi; then
        log_info "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
        echo ""
    else
        log_info "No NVIDIA GPU detected (CPU-only mode)"
    fi
    
    # Check PyTorch CUDA availability
    log_info "Checking PyTorch CUDA availability..."
    python -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'Device Count: {torch.cuda.device_count()}')
    print(f'Current Device: {torch.cuda.current_device()}')
    print(f'Device Name: {torch.cuda.get_device_name(0)}')
else:
    print('Running in CPU mode')
" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        log_success "PyTorch device check completed"
    else
        log_warning "PyTorch device check failed"
    fi
}

###############################################################################
# Step 9: Print Summary
###############################################################################

print_summary() {
    print_header "Installation Summary"
    
    cat << EOF
Production environment setup completed!

Virtual Environment: $VENV_DIR
Python Version: $PYTHON_VERSION
Requirements File: $REQUIREMENTS_FILE

To activate the environment:
    source $VENV_DIR/bin/activate    # Linux/Mac
    $VENV_DIR\\Scripts\\activate      # Windows

To start the API server:
    uvicorn deployment.api.app:app --host 0.0.0.0 --port 8000

To run with Docker:
    cd deployment/docker
    docker-compose up -d

For more information, see the documentation in docs/

EOF
    
    log_info "Next steps:"
    echo "  1. Activate virtual environment"
    echo "  2. Download/prepare model checkpoints"
    echo "  3. Configure environment variables"
    echo "  4. Run the API server or deploy with Docker"
}

###############################################################################
# Execute Main Function
###############################################################################

main "$@"

exit 0
