#!/usr/bin/env python3
"""
Environment Setup Script for FactCheck-MM
Handles virtual environment creation, dependency installation, and hardware verification.
"""

import os
import sys
import platform
import subprocess
import argparse
import venv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.utils.logging_utils import get_logger


class FactCheckEnvironmentSetup:
    """Complete environment setup for FactCheck-MM project."""
    
    def __init__(self, project_root: Path = None):
        """
        Initialize environment setup.
        
        Args:
            project_root: Path to project root directory
        """
        self.project_root = project_root or Path(__file__).parent.parent
        self.logger = get_logger("EnvironmentSetup")
        self.venv_path = self.project_root / "venv"
        
        # System information
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict[str, str]:
        """Get comprehensive system information."""
        import platform
        
        info = {
            'platform': platform.platform(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation()
        }
        
        # Add macOS specific info
        if platform.system() == 'Darwin':
            try:
                # Check for Apple Silicon
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    info['cpu_brand'] = result.stdout.strip()
                    info['is_apple_silicon'] = 'Apple' in result.stdout
            except:
                pass
        
        return info
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        version = sys.version_info
        
        if version.major != 3 or version.minor < 8:
            self.logger.error(f"Python 3.8+ required, found {version.major}.{version.minor}")
            return False
        
        self.logger.info(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    
    def create_virtual_environment(self, force_recreate: bool = False) -> bool:
        """
        Create virtual environment.
        
        Args:
            force_recreate: Whether to recreate existing environment
            
        Returns:
            True if successful
        """
        if self.venv_path.exists():
            if force_recreate:
                self.logger.info(f"🔄 Recreating virtual environment at {self.venv_path}")
                import shutil
                shutil.rmtree(self.venv_path)
            else:
                self.logger.info(f"✅ Virtual environment already exists at {self.venv_path}")
                return True
        
        try:
            self.logger.info(f"📦 Creating virtual environment at {self.venv_path}")
            venv.create(self.venv_path, with_pip=True)
            
            # Verify creation
            if self._is_venv_valid():
                self.logger.info("✅ Virtual environment created successfully")
                return True
            else:
                self.logger.error("❌ Virtual environment creation failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to create virtual environment: {e}")
            return False
    
    def _is_venv_valid(self) -> bool:
        """Check if virtual environment is valid."""
        if not self.venv_path.exists():
            return False
        
        # Check for activation script
        if platform.system() == 'Windows':
            activate_script = self.venv_path / 'Scripts' / 'activate.bat'
        else:
            activate_script = self.venv_path / 'bin' / 'activate'
        
        return activate_script.exists()
    
    def get_pip_path(self) -> Path:
        """Get pip executable path in virtual environment."""
        if platform.system() == 'Windows':
            return self.venv_path / 'Scripts' / 'pip.exe'
        else:
            return self.venv_path / 'bin' / 'pip'
    
    def get_python_path(self) -> Path:
        """Get Python executable path in virtual environment."""
        if platform.system() == 'Windows':
            return self.venv_path / 'Scripts' / 'python.exe'
        else:
            return self.venv_path / 'bin' / 'python'
    
    def install_dependencies(self, dev_mode: bool = False) -> bool:
        """
        Install project dependencies.
        
        Args:
            dev_mode: Whether to install development dependencies
            
        Returns:
            True if successful
        """
        pip_path = self.get_pip_path()
        
        if not pip_path.exists():
            self.logger.error("❌ Pip not found in virtual environment")
            return False
        
        # Upgrade pip first
        self.logger.info("⬆️ Upgrading pip...")
        result = subprocess.run([str(pip_path), 'install', '--upgrade', 'pip'], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            self.logger.warning(f"⚠️ Pip upgrade warning: {result.stderr}")
        
        # Install core requirements
        requirements_file = self.project_root / 'requirements.txt'
        
        if not requirements_file.exists():
            self.logger.error(f"❌ Requirements file not found: {requirements_file}")
            return False
        
        self.logger.info(f"📥 Installing dependencies from {requirements_file}")
        result = subprocess.run([str(pip_path), 'install', '-r', str(requirements_file)], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            self.logger.error(f"❌ Failed to install dependencies: {result.stderr}")
            return False
        
        # Install platform-specific packages
        platform_packages = self._get_platform_packages()
        if platform_packages:
            self.logger.info(f"📥 Installing platform-specific packages: {platform_packages}")
            result = subprocess.run([str(pip_path), 'install'] + platform_packages, 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.warning(f"⚠️ Some platform packages failed: {result.stderr}")
        
        # Install development dependencies
        if dev_mode:
            dev_packages = ['pytest', 'pytest-cov', 'black', 'flake8', 'mypy', 'jupyter', 'ipykernel']
            self.logger.info(f"📥 Installing development packages: {dev_packages}")
            result = subprocess.run([str(pip_path), 'install'] + dev_packages, 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.warning(f"⚠️ Some dev packages failed: {result.stderr}")
        
        self.logger.info("✅ Dependencies installed successfully")
        return True
    
    def _get_platform_packages(self) -> List[str]:
        """Get platform-specific packages."""
        packages = []
        
        # Audio processing (all platforms)
        packages.extend(['librosa', 'soundfile'])
        
        # Image/video processing
        packages.extend(['opencv-python', 'Pillow'])
        
        # Data augmentation
        packages.extend(['albumentations', 'imgaug'])
        
        # Optional audio augmentation
        try:
            # Test if audiomentations can be installed
            packages.append('audiomentations')
        except:
            self.logger.debug("Skipping audiomentations (may not be available)")
        
        return packages
    
    def verify_pytorch_installation(self) -> Dict[str, bool]:
        """
        Verify PyTorch installation and hardware support.
        
        Returns:
            Dictionary with verification results
        """
        results = {
            'pytorch_installed': False,
            'cuda_available': False,
            'mps_available': False,
            'gpu_count': 0,
            'recommended_device': 'cpu'
        }
        
        try:
            # Import PyTorch
            python_path = self.get_python_path()
            result = subprocess.run([str(python_path), '-c', 
                                   'import torch; print(torch.__version__)'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                results['pytorch_installed'] = True
                torch_version = result.stdout.strip()
                self.logger.info(f"✅ PyTorch installed: {torch_version}")
                
                # Check CUDA
                result = subprocess.run([str(python_path), '-c', 
                                       'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'], 
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    cuda_info = result.stdout.strip().split()
                    results['cuda_available'] = cuda_info[0] == 'True'
                    results['gpu_count'] = int(cuda_info[1]) if len(cuda_info) > 1 else 0
                    
                    if results['cuda_available']:
                        self.logger.info(f"✅ CUDA available with {results['gpu_count']} GPU(s)")
                        results['recommended_device'] = 'cuda'
                        
                        # Get GPU info
                        result = subprocess.run([str(python_path), '-c', 
                                               'import torch; print(torch.cuda.get_device_name(0))'], 
                                              capture_output=True, text=True)
                        if result.returncode == 0:
                            gpu_name = result.stdout.strip()
                            self.logger.info(f"🎮 GPU: {gpu_name}")
                
                # Check MPS (Apple Silicon)
                if platform.system() == 'Darwin':
                    result = subprocess.run([str(python_path), '-c', 
                                           'import torch; print(torch.backends.mps.is_available())'], 
                                          capture_output=True, text=True)
                    
                    if result.returncode == 0 and result.stdout.strip() == 'True':
                        results['mps_available'] = True
                        self.logger.info("✅ MPS (Metal Performance Shaders) available")
                        if not results['cuda_available']:
                            results['recommended_device'] = 'mps'
            
            else:
                self.logger.error(f"❌ PyTorch not properly installed: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to verify PyTorch: {e}")
        
        return results
    
    def install_pytorch(self, force_cpu: bool = False) -> bool:
        """
        Install PyTorch with appropriate backend support.
        
        Args:
            force_cpu: Force CPU-only installation
            
        Returns:
            True if successful
        """
        pip_path = self.get_pip_path()
        
        if force_cpu:
            self.logger.info("📥 Installing PyTorch (CPU-only)")
            cmd = [str(pip_path), 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 
                   'https://download.pytorch.org/whl/cpu']
        else:
            # Auto-detect best PyTorch version
            if platform.system() == 'Darwin':
                # macOS - install with MPS support
                self.logger.info("📥 Installing PyTorch for macOS (with MPS support)")
                cmd = [str(pip_path), 'install', 'torch', 'torchvision', 'torchaudio']
            else:
                # Linux/Windows - try CUDA first
                self.logger.info("📥 Installing PyTorch (CUDA support)")
                cmd = [str(pip_path), 'install', 'torch', 'torchvision', 'torchaudio', '--index-url',
                       'https://download.pytorch.org/whl/cu118']
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            self.logger.error(f"❌ Failed to install PyTorch: {result.stderr}")
            return False
        
        self.logger.info("✅ PyTorch installed successfully")
        return True
    
    def setup_jupyter_kernel(self) -> bool:
        """Setup Jupyter kernel for the virtual environment."""
        python_path = self.get_python_path()
        
        # Install ipykernel in the venv
        result = subprocess.run([str(python_path), '-m', 'pip', 'install', 'ipykernel'], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            self.logger.error(f"❌ Failed to install ipykernel: {result.stderr}")
            return False
        
        # Add kernel to Jupyter
        kernel_name = 'factcheck-mm'
        result = subprocess.run([str(python_path), '-m', 'ipykernel', 'install', '--user', 
                               '--name', kernel_name, '--display-name', 'FactCheck-MM'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            self.logger.info(f"✅ Jupyter kernel '{kernel_name}' installed")
            return True
        else:
            self.logger.warning(f"⚠️ Jupyter kernel setup failed: {result.stderr}")
            return False
    
    def generate_activation_script(self) -> bool:
        """Generate platform-specific activation script."""
        try:
            if platform.system() == 'Windows':
                script_name = 'activate_factcheck.bat'
                script_content = f"""@echo off
echo Activating FactCheck-MM environment...
call "{self.venv_path}\\Scripts\\activate.bat"
cd /d "{self.project_root}"
echo ✅ Environment activated. Project root: {self.project_root}
echo Run: python scripts/setup_environment.py --verify to check setup
"""
            else:
                script_name = 'activate_factcheck.sh'
                script_content = f"""#!/bin/bash
echo "Activating FactCheck-MM environment..."
source "{self.venv_path}/bin/activate"
cd "{self.project_root}"
echo "✅ Environment activated. Project root: {self.project_root}"
echo "Run: python scripts/setup_environment.py --verify to check setup"
"""
            
            script_path = self.project_root / script_name
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Make executable on Unix systems
            if platform.system() != 'Windows':
                import stat
                script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)
            
            self.logger.info(f"✅ Activation script created: {script_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create activation script: {e}")
            return False
    
    def run_setup(self, args: argparse.Namespace) -> bool:
        """
        Run complete environment setup.
        
        Args:
            args: Parsed command-line arguments
            
        Returns:
            True if successful
        """
        self.logger.info("🚀 Starting FactCheck-MM environment setup")
        self.logger.info(f"📍 Project root: {self.project_root}")
        self.logger.info(f"🖥️ System: {self.system_info['platform']}")
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Create virtual environment
        if not self.create_virtual_environment(args.force_recreate):
            return False
        
        # Install PyTorch if requested
        if args.install_pytorch:
            if not self.install_pytorch(args.cpu_only):
                return False
        
        # Install dependencies
        if not self.install_dependencies(args.dev_mode):
            return False
        
        # Verify PyTorch installation
        pytorch_info = self.verify_pytorch_installation()
        
        if pytorch_info['pytorch_installed']:
            device = pytorch_info['recommended_device']
            self.logger.info(f"🎯 Recommended device for training: {device}")
            
            if device == 'cuda':
                self.logger.info("💡 RTX 2050 detected - optimal for multimodal training")
            elif device == 'mps':
                self.logger.info("🍎 Apple Silicon detected - good for text-only baseline")
            else:
                self.logger.info("💻 CPU-only - consider GPU for better performance")
        
        # Setup Jupyter kernel
        if args.setup_jupyter:
            self.setup_jupyter_kernel()
        
        # Generate activation script
        self.generate_activation_script()
        
        self.logger.info("🎉 Environment setup completed successfully!")
        self.logger.info("📚 Next steps:")
        self.logger.info("   1. Activate environment using generated script")
        self.logger.info("   2. Run: python scripts/setup_environment.py --verify")
        self.logger.info("   3. Start training: python scripts/run_full_pipeline.py")
        
        return True
    
    def verify_setup(self) -> bool:
        """Verify environment setup."""
        self.logger.info("🔍 Verifying FactCheck-MM environment setup")
        
        success = True
        
        # Check virtual environment
        if not self._is_venv_valid():
            self.logger.error("❌ Virtual environment not found or invalid")
            success = False
        else:
            self.logger.info("✅ Virtual environment valid")
        
        # Check PyTorch
        pytorch_info = self.verify_pytorch_installation()
        
        if not pytorch_info['pytorch_installed']:
            self.logger.error("❌ PyTorch not installed")
            success = False
        
        # Check project modules
        python_path = self.get_python_path()
        
        modules_to_check = [
            'shared.utils.logging_utils',
            'sarcasm_detection.models',
            'config.base_config'
        ]
        
        for module in modules_to_check:
            result = subprocess.run([str(python_path), '-c', f'import {module}'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"✅ Module {module} importable")
            else:
                self.logger.error(f"❌ Module {module} not importable: {result.stderr}")
                success = False
        
        # Check data directory
        data_dir = self.project_root / 'data'
        if data_dir.exists():
            dataset_count = len([d for d in data_dir.iterdir() if d.is_dir()])
            self.logger.info(f"✅ Data directory found with {dataset_count} datasets")
        else:
            self.logger.warning("⚠️ Data directory not found - datasets may need to be downloaded")
        
        if success:
            self.logger.info("🎉 Environment verification successful!")
            self.logger.info("🚀 Ready to start training FactCheck-MM models")
        else:
            self.logger.error("❌ Environment verification failed")
            self.logger.info("💡 Try running: python scripts/setup_environment.py --force-recreate")
        
        return success


def main():
    """Main entry point for environment setup."""
    parser = argparse.ArgumentParser(description="FactCheck-MM Environment Setup")
    
    parser.add_argument('--force-recreate', action='store_true',
                       help='Force recreation of virtual environment')
    parser.add_argument('--install-pytorch', action='store_true', default=True,
                       help='Install PyTorch (default: True)')
    parser.add_argument('--cpu-only', action='store_true',
                       help='Install CPU-only PyTorch')
    parser.add_argument('--dev-mode', action='store_true',
                       help='Install development dependencies')
    parser.add_argument('--setup-jupyter', action='store_true',
                       help='Setup Jupyter kernel')
    parser.add_argument('--verify', action='store_true',
                       help='Only verify existing setup')
    parser.add_argument('--project-root', type=Path,
                       help='Project root directory (auto-detected if not provided)')
    
    args = parser.parse_args()
    
    # Initialize setup
    setup = FactCheckEnvironmentSetup(args.project_root)
    
    try:
        if args.verify:
            success = setup.verify_setup()
        else:
            success = setup.run_setup(args)
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        setup.logger.info("⏹️ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        setup.logger.error(f"💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
