#!/usr/bin/env python3
"""
Cache Cleaning Script for FactCheck-MM
Cleans __pycache__, checkpoints, cached datasets, logs, and temporary files.
"""

import os
import sys
import shutil
import argparse
import time
from pathlib import Path
from typing import List, Dict, Tuple
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.utils.logging_utils import get_logger


class CacheManager:
    """Manages cache cleaning for FactCheck-MM project."""
    
    def __init__(self, project_root: Path = None, dry_run: bool = False):
        """Initialize cache manager."""
        self.project_root = project_root or Path(__file__).parent.parent
        self.dry_run = dry_run
        self.logger = get_logger("CacheManager")
        
        # Cache patterns to clean
        self.cache_patterns = {
            'python_cache': [
                '**/__pycache__',
                '**/*.pyc',
                '**/*.pyo',
                '**/.pytest_cache'
            ],
            'checkpoints': [
                '**/checkpoints/**/*.pt',
                '**/checkpoints/**/*.pth',
                '**/checkpoints/**/*.ckpt'
            ],
            'logs': [
                '**/logs/**/*.log',
                '**/tensorboard/**/*',
                '**/wandb/**/*',
                '**/.wandb/**/*'
            ],
            'datasets_cache': [
                '**/.cache/**/*',
                '**/cached_datasets/**/*',
                '**/*.cache',
                '**/processed_data/**/*'
            ],
            'build_artifacts': [
                '**/build/**/*',
                '**/dist/**/*',
                '**/*.egg-info/**/*',
                '**/.coverage',
                '**/coverage.xml'
            ],
            'temp_files': [
                '**/.tmp/**/*',
                '**/tmp/**/*',
                '**/*.tmp',
                '**/*.temp'
            ],
            'jupyter': [
                '**/.ipynb_checkpoints/**/*',
                '**/*-checkpoint.ipynb'
            ],
            'model_artifacts': [
                '**/exported_models/**/*',
                '**/*.onnx',
                '**/*.tflite',
                '**/*.mlmodel'
            ]
        }
        
        self.logger.info(f"Initialized cache manager for: {self.project_root}")
    
    def get_cache_size(self, patterns: List[str]) -> Tuple[List[Path], int]:
        """Calculate total size of files matching patterns."""
        files = []
        total_size = 0
        
        for pattern in patterns:
            for path in self.project_root.glob(pattern):
                if path.is_file():
                    try:
                        size = path.stat().st_size
                        files.append(path)
                        total_size += size
                    except (OSError, FileNotFoundError):
                        # Skip files that can't be accessed
                        continue
        
        return files, total_size
    
    def clean_cache_category(self, category: str, patterns: List[str]) -> Dict[str, any]:
        """Clean a specific cache category."""
        self.logger.info(f"🧹 Cleaning {category}...")
        
        # Get files to clean
        files, total_size = self.get_cache_size(patterns)
        
        if not files:
            self.logger.info(f"✅ No {category} cache found")
            return {
                'category': category,
                'files_found': 0,
                'size_mb': 0,
                'files_deleted': 0,
                'errors': []
            }
        
        size_mb = total_size / (1024 * 1024)
        self.logger.info(f"📊 Found {len(files)} files ({size_mb:.2f} MB)")
        
        if self.dry_run:
            self.logger.info("🔍 [DRY RUN] Would delete:")
            for file_path in files[:10]:  # Show first 10 files
                self.logger.info(f"   - {file_path.relative_to(self.project_root)}")
            if len(files) > 10:
                self.logger.info(f"   ... and {len(files) - 10} more files")
            
            return {
                'category': category,
                'files_found': len(files),
                'size_mb': size_mb,
                'files_deleted': 0,
                'errors': []
            }
        
        # Delete files
        deleted_count = 0
        errors = []
        
        for file_path in files:
            try:
                if file_path.is_file():
                    file_path.unlink()
                    deleted_count += 1
                elif file_path.is_dir() and not any(file_path.iterdir()):
                    # Remove empty directory
                    file_path.rmdir()
                    deleted_count += 1
            except Exception as e:
                errors.append(f"{file_path}: {e}")
        
        # Clean empty directories
        self._remove_empty_directories(patterns)
        
        self.logger.info(f"✅ Deleted {deleted_count} files ({size_mb:.2f} MB)")
        
        if errors:
            self.logger.warning(f"⚠️ {len(errors)} errors occurred")
            for error in errors[:5]:  # Show first 5 errors
                self.logger.warning(f"   {error}")
        
        return {
            'category': category,
            'files_found': len(files),
            'size_mb': size_mb,
            'files_deleted': deleted_count,
            'errors': errors
        }
    
    def _remove_empty_directories(self, patterns: List[str]):
        """Remove empty directories that match patterns."""
        directories = set()
        
        for pattern in patterns:
            for path in self.project_root.glob(pattern):
                if path.is_dir():
                    directories.add(path)
                else:
                    # Add parent directories
                    directories.add(path.parent)
        
        # Sort by depth (deepest first) to remove leaf directories first
        sorted_dirs = sorted(directories, key=lambda x: len(x.parts), reverse=True)
        
        for directory in sorted_dirs:
            try:
                if directory.exists() and directory.is_dir():
                    # Check if directory is empty
                    if not any(directory.iterdir()):
                        if not self.dry_run:
                            directory.rmdir()
                        self.logger.debug(f"Removed empty directory: {directory}")
            except Exception as e:
                self.logger.debug(f"Could not remove directory {directory}: {e}")
    
    def clean_pytorch_cache(self):
        """Clean PyTorch-specific cache."""
        self.logger.info("🧹 Cleaning PyTorch cache...")
        
        try:
            import torch
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("✅ Cleared CUDA cache")
            
            # Clear MPS cache (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                self.logger.info("✅ Cleared MPS cache")
                
        except ImportError:
            self.logger.info("⚠️ PyTorch not available - skipping PyTorch cache")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not clear PyTorch cache: {e}")
    
    def clean_transformers_cache(self):
        """Clean Transformers library cache."""
        self.logger.info("🧹 Cleaning Transformers cache...")
        
        try:
            from transformers import utils
            
            # Get cache directory
            cache_dir = Path(utils.TRANSFORMERS_CACHE)
            
            if cache_dir.exists():
                cache_files, cache_size = self.get_cache_size([str(cache_dir / '**' / '*')])
                
                if cache_files:
                    size_mb = cache_size / (1024 * 1024)
                    self.logger.info(f"📊 Transformers cache: {len(cache_files)} files ({size_mb:.2f} MB)")
                    
                    if not self.dry_run:
                        # Ask for confirmation for large caches
                        if size_mb > 1000:  # > 1GB
                            response = input(f"Transformers cache is {size_mb:.2f} MB. Delete? (y/N): ")
                            if response.lower() != 'y':
                                self.logger.info("⏭️ Skipping Transformers cache")
                                return
                        
                        shutil.rmtree(cache_dir)
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        self.logger.info(f"✅ Cleared Transformers cache ({size_mb:.2f} MB)")
                    else:
                        self.logger.info(f"🔍 [DRY RUN] Would clear Transformers cache ({size_mb:.2f} MB)")
                else:
                    self.logger.info("✅ No Transformers cache found")
            else:
                self.logger.info("✅ No Transformers cache directory found")
                
        except ImportError:
            self.logger.info("⚠️ Transformers not available - skipping Transformers cache")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not clear Transformers cache: {e}")
    
    def get_total_cache_size(self) -> Dict[str, float]:
        """Get total cache size by category."""
        sizes = {}
        total_size = 0
        
        for category, patterns in self.cache_patterns.items():
            _, size_bytes = self.get_cache_size(patterns)
            size_mb = size_bytes / (1024 * 1024)
            sizes[category] = size_mb
            total_size += size_mb
        
        sizes['total'] = total_size
        return sizes
    
    def clean_all(self, categories: List[str] = None) -> Dict[str, any]:
        """Clean all cache categories."""
        if categories is None:
            categories = list(self.cache_patterns.keys())
        
        self.logger.info(f"🧹 Starting cache cleanup - Categories: {categories}")
        
        if self.dry_run:
            self.logger.info("🔍 DRY RUN MODE - No files will be deleted")
        
        results = {
            'categories': {},
            'summary': {
                'total_files_found': 0,
                'total_files_deleted': 0,
                'total_size_mb': 0,
                'total_errors': 0
            },
            'execution_time': 0
        }
        
        start_time = time.time()
        
        # Clean each category
        for category in categories:
            if category in self.cache_patterns:
                patterns = self.cache_patterns[category]
                category_result = self.clean_cache_category(category, patterns)
                results['categories'][category] = category_result
                
                # Update summary
                results['summary']['total_files_found'] += category_result['files_found']
                results['summary']['total_files_deleted'] += category_result['files_deleted']
                results['summary']['total_size_mb'] += category_result['size_mb']
                results['summary']['total_errors'] += len(category_result['errors'])
            else:
                self.logger.warning(f"⚠️ Unknown category: {category}")
        
        # Clean special caches
        self.clean_pytorch_cache()
        self.clean_transformers_cache()
        
        results['execution_time'] = time.time() - start_time
        
        # Print summary
        summary = results['summary']
        self.logger.info(f"🎉 Cache cleanup completed in {results['execution_time']:.2f}s")
        self.logger.info(f"📊 Summary:")
        self.logger.info(f"   Files found: {summary['total_files_found']:,}")
        self.logger.info(f"   Files deleted: {summary['total_files_deleted']:,}")
        self.logger.info(f"   Space freed: {summary['total_size_mb']:.2f} MB")
        
        if summary['total_errors'] > 0:
            self.logger.warning(f"⚠️ Errors: {summary['total_errors']}")
        
        return results
    
    def analyze_cache(self) -> Dict[str, float]:
        """Analyze cache without cleaning."""
        self.logger.info("🔍 Analyzing cache usage...")
        
        sizes = self.get_total_cache_size()
        
        self.logger.info("📊 Cache analysis results:")
        
        for category, size_mb in sorted(sizes.items(), key=lambda x: x[1], reverse=True):
            if category == 'total':
                self.logger.info(f"   {'='*40}")
                self.logger.info(f"   📦 TOTAL: {size_mb:.2f} MB")
            else:
                if size_mb > 0:
                    self.logger.info(f"   📁 {category}: {size_mb:.2f} MB")
        
        return sizes


def main():
    """Main entry point for cache cleaning."""
    parser = argparse.ArgumentParser(description="FactCheck-MM Cache Cleaner")
    
    # Operation mode
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze cache without cleaning')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deleted without deleting')
    
    # Categories
    parser.add_argument('--categories', nargs='+',
                       choices=['python_cache', 'checkpoints', 'logs', 'datasets_cache',
                               'build_artifacts', 'temp_files', 'jupyter', 'model_artifacts'],
                       help='Specific categories to clean')
    
    # Exclusions
    parser.add_argument('--keep-checkpoints', action='store_true',
                       help='Keep model checkpoints')
    parser.add_argument('--keep-logs', action='store_true',
                       help='Keep log files')
    
    # Project settings
    parser.add_argument('--project-root', type=Path,
                       help='Project root directory')
    
    args = parser.parse_args()
    
    try:
        # Initialize cache manager
        cache_manager = CacheManager(
            project_root=args.project_root,
            dry_run=args.dry_run
        )
        
        if args.analyze:
            # Analyze only
            cache_manager.analyze_cache()
        else:
            # Determine categories to clean
            categories = args.categories
            
            if categories is None:
                categories = list(cache_manager.cache_patterns.keys())
                
                # Apply exclusions
                if args.keep_checkpoints:
                    categories = [c for c in categories if c != 'checkpoints']
                
                if args.keep_logs:
                    categories = [c for c in categories if c != 'logs']
            
            # Confirm if not dry run
            if not args.dry_run:
                print(f"\n⚠️  This will clean the following categories: {categories}")
                print(f"📂 Project root: {cache_manager.project_root}")
                
                response = input("\nProceed with cleaning? (y/N): ")
                if response.lower() != 'y':
                    print("❌ Cleaning cancelled")
                    sys.exit(0)
            
            # Clean cache
            results = cache_manager.clean_all(categories)
            
            # Show results
            print(f"\n🎉 Cleaning completed!")
            print(f"📊 {results['summary']['total_files_deleted']:,} files deleted")
            print(f"💾 {results['summary']['total_size_mb']:.2f} MB freed")
            
            if results['summary']['total_errors'] > 0:
                print(f"⚠️ {results['summary']['total_errors']} errors occurred")
    
    except KeyboardInterrupt:
        print("\n⏹️ Cleaning interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Cleaning failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
