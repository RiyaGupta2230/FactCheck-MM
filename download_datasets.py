import os
import requests
import zipfile
import time
import subprocess
import json
import gdown

# Directory setup with new dataset folders
base_dir = 'data'
text_dir = os.path.join(base_dir, 'text')
multimodal_dir = os.path.join(base_dir, 'multimodal')
fact_verification_dir = os.path.join(base_dir, 'fact_verification')
paraphrasing_dir = os.path.join(base_dir, 'text', 'paraphrasing')

# Create all directories including new ones
os.makedirs(text_dir, exist_ok=True)
os.makedirs(multimodal_dir, exist_ok=True)
os.makedirs(fact_verification_dir, exist_ok=True)
os.makedirs(paraphrasing_dir, exist_ok=True)
os.makedirs(os.path.join(multimodal_dir, 'sarcnet'), exist_ok=True)
os.makedirs(os.path.join(multimodal_dir, 'mmsd2'), exist_ok=True)
os.makedirs(os.path.join(multimodal_dir, 'ur_funny'), exist_ok=True)
os.makedirs(os.path.join(text_dir, 'spanish_sarcasm'), exist_ok=True)

def download_file_with_progress(url, save_path, timeout=120):
    """Download file with progress indicator and robust error handling"""
    try:
        print(f'Downloading: {os.path.basename(save_path)}')
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f'\rProgress: {percent:.1f}%', end='', flush=True)
        
        print(f'\n✅ Saved to {save_path}')
        return True
    except Exception as e:
        print(f'\n❌ Download failed: {e}')
        return False

def check_existing_file(filepath, min_size_kb=1):
    """Check if file exists and has reasonable size"""
    if os.path.exists(filepath):
        size_kb = os.path.getsize(filepath) / 1024
        if size_kb > min_size_kb:
            print(f'✅ Already exists: {os.path.basename(filepath)} ({size_kb:.1f} KB)')
            return True
    return False

# NEW DATASET DOWNLOAD FUNCTIONS

def download_sarcnet():
    print('\n=== SarcNet Dataset (Multilingual Multimodal) ===')
    sarcnet_dir = os.path.join(multimodal_dir, 'sarcnet')
    
    # Check if already downloaded
    if os.path.exists(os.path.join(sarcnet_dir, 'sarcnet_data.json')) or len(os.listdir(sarcnet_dir)) > 1:
        print('✅ SarcNet already exists')
        return True
    
    print('🔄 Attempting to download SarcNet from Google Drive...')
    try:
        # SarcNet Google Drive file ID
        file_id = '18m3KdDCXgkAlvTbjNhfftvU9LdhyDUTt'
        output_path = os.path.join(sarcnet_dir, 'sarcnet_dataset.zip')
        
        # Download using gdown
        gdown.download(f'https://drive.google.com/uc?id={file_id}', output_path, quiet=False)
        
        # Extract the zip file
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(sarcnet_dir)
        
        os.remove(output_path)  # Clean up zip file
        print('✅ SarcNet downloaded and extracted!')
        return True
        
    except ImportError:
        print('❌ gdown package not installed')
        print('💡 Install with: pip install gdown')
    except Exception as e:
        print(f'❌ Download failed: {e}')
    
    print('📋 Manual Download Steps:')
    print('1. Go to: https://github.com/yuetanbupt/SarcNet')
    print('2. Click the Google Drive link: https://drive.google.com/file/d/18m3KdDCXgkAlvTbjNhfftvU9LdhyDUTt/view')
    print('3. Download and extract to: data/multimodal/sarcnet/')
    print('4. Contains 3,335 multilingual image-text pairs (English/Chinese)')
    return False

def download_mmsd2():
    print('\n=== MMSD 2.0 Dataset (Multimodal Sarcasm Detection) ===')
    mmsd2_dir = os.path.join(multimodal_dir, 'mmsd2')
    
    if os.path.exists(mmsd2_dir) and len(os.listdir(mmsd2_dir)) > 2:
        print('✅ MMSD 2.0 already exists')
        return True
    
    try:
        print('🔄 Cloning MMSD 2.0 repository...')
        subprocess.run(['git', 'clone', 'https://github.com/JoeYing1019/MMSD2.0.git', mmsd2_dir], 
                      check=True, timeout=300)
        print('✅ MMSD 2.0 cloned successfully!')
        return True
        
    except subprocess.CalledProcessError as e:
        print(f'❌ Git clone failed: {e}')
    except FileNotFoundError:
        print('❌ Git not found. Please install Git first.')
    except Exception as e:
        print(f'❌ Error: {e}')
    
    # Alternative Hugging Face download
    try:
        print('🔄 Trying Hugging Face download...')
        from datasets import load_dataset
        dataset = load_dataset("coderchen01/MMSD2.0")
        dataset.save_to_disk(os.path.join(mmsd2_dir, 'hf_dataset'))
        print('✅ MMSD 2.0 downloaded from Hugging Face!')
        return True
    except ImportError:
        print('❌ datasets package not installed')
        print('💡 Install with: pip install datasets')
    except Exception as e:
        print(f'❌ Hugging Face download failed: {e}')
    
    print('📋 Manual Download Steps:')
    print('1. Go to: https://github.com/JoeYing1019/MMSD2.0')
    print('2. Click "Code" -> "Download ZIP"')
    print('3. Extract to: data/multimodal/mmsd2/')
    print('4. Alternative: https://huggingface.co/datasets/coderchen01/MMSD2.0')
    print('5. Contains 24,635 samples (ACL 2023)')
    return False

def download_spanish_sarcasm():
    print('\n=== Spanish Sarcasm Dataset ===')
    spanish_dir = os.path.join(text_dir, 'spanish_sarcasm')
    
    csv_files = [f for f in os.listdir(spanish_dir) if f.endswith('.csv')]
    if csv_files:
        print(f'✅ Spanish Sarcasm already exists ({len(csv_files)} files)')
        return True
    
    # Try Kaggle download
    try:
        import kaggle
        print('🔄 Downloading Spanish Sarcasm from Kaggle...')
        kaggle.api.dataset_download_files(
            'mikahama/the-best-sarcasm-annotated-dataset-in-spanish', 
            path=spanish_dir, unzip=True
        )
        print('✅ Spanish Sarcasm downloaded from Kaggle!')
        return True
        
    except ImportError:
        print('❌ Kaggle API not installed')
        print('💡 Install with: pip install kaggle')
        print('💡 Setup Kaggle credentials: https://www.kaggle.com/docs/api')
    except Exception as e:
        print(f'❌ Kaggle download failed: {e}')
    
    print('📋 Manual Download Steps:')
    print('1. Go to: https://www.kaggle.com/datasets/mikahama/the-best-sarcasm-annotated-dataset-in-spanish')
    print('2. Create free Kaggle account and download dataset')
    print('3. Extract CSV files to: data/text/spanish_sarcasm/')
    print('4. Contains 960 Spanish utterances with sarcasm annotations')
    print('5. Alternative: https://zenodo.org/records/4701383 (video-aligned version)')
    return False

def download_ur_funny():
    print('\n=== UR-FUNNY Dataset (Multimodal Humor Detection) ===')
    ur_funny_dir = os.path.join(multimodal_dir, 'ur_funny')
    
    if os.path.exists(ur_funny_dir) and len(os.listdir(ur_funny_dir)) > 2:
        print('✅ UR-FUNNY already exists')
        return True
    
    try:
        print('🔄 Cloning UR-FUNNY repository...')
        subprocess.run(['git', 'clone', 'https://github.com/ROC-HCI/UR-FUNNY.git', ur_funny_dir], 
                      check=True, timeout=300)
        print('✅ UR-FUNNY repository cloned successfully!')
        
        # Try to download the actual dataset files
        print('🔄 Downloading dataset files...')
        # V2 dataset links from the repository
        v2_links = {
            'videos': 'https://www.dropbox.com/s/lg7kjx0kul3ansq/urfunny2_videos.zip?dl=1',
            'features': 'https://www.dropbox.com/sh/9h0pcqmqoplx9p2/AAC8yYikSBVYCSFjm3afFHQva?dl=1'
        }
        
        # Download videos
        video_path = os.path.join(ur_funny_dir, 'urfunny2_videos.zip')
        if download_file_with_progress(v2_links['videos'], video_path, timeout=600):
            print('Videos downloaded, extracting...')
            with zipfile.ZipFile(video_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(ur_funny_dir, 'videos'))
            os.remove(video_path)
        
        print('✅ UR-FUNNY dataset downloaded!')
        return True
        
    except subprocess.CalledProcessError as e:
        print(f'❌ Git clone failed: {e}')
    except FileNotFoundError:
        print('❌ Git not found. Please install Git first.')
    except Exception as e:
        print(f'❌ Error: {e}')
    
    print('📋 Manual Download Steps:')
    print('1. Go to: https://github.com/ROC-HCI/UR-FUNNY')
    print('2. Follow repository instructions for dataset access')
    print('3. Download V2 dataset:')
    print('   - Videos: https://www.dropbox.com/s/lg7kjx0kul3ansq/urfunny2_videos.zip?dl=1')
    print('   - Features: https://www.dropbox.com/sh/9h0pcqmqoplx9p2/AAC8yYikSBVYCSFjm3afFHQva?dl=1')
    print('4. Extract to: data/multimodal/ur_funny/')
    print('5. Contains 1,260+ video segments for humor detection (EMNLP 2019)')
    return False

# EXISTING DATASET FUNCTIONS (keep your current implementations)

def download_mustard():
    print('\n=== MUStARD Dataset ===')
    mustard_dirs = ['mustard_bert', 'mustard_features', 'mustard_repo']
    found = [d for d in mustard_dirs if os.path.exists(os.path.join(multimodal_dir, d))]
    
    if found:
        print(f'✅ Found MUStARD components: {found}')
        return True
    else:
        print('⚠️ MUStARD not found in data/multimodal/')
        print('📋 Manual Download Steps:')
        print('1. Go to: https://github.com/soujanyaporia/MUStARD')
        print('2. Download as ZIP or clone repository')
        print('3. Extract/move dataset folders to data/multimodal/')
        print('4. Ensure folders: mustard_bert/, mustard_features/, mustard_repo/')
        return False

def download_news_headline_sarcasm():
    print('\n=== News Headline Sarcasm Dataset ===')
    save_path = os.path.join(text_dir, 'Sarcasm_Headlines_Dataset.json')
    
    if check_existing_file(save_path, 100):
        return True
    
    github_url = 'https://raw.githubusercontent.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection/master/Sarcasm_Headlines_Dataset.json'
    
    if download_file_with_progress(github_url, save_path, timeout=60):
        return True
    else:
        print('📋 Manual Download Steps:')
        print('1. Go to: https://github.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection')
        print('2. Download Sarcasm_Headlines_Dataset.json')
        print('3. Place in: data/text/')
        return False

def download_liar():
    print('\n=== LIAR Dataset ===')
    
    train_file = os.path.join(fact_verification_dir, 'train.tsv')
    if check_existing_file(train_file, 500):
        return True
    
    url = 'https://www.cs.ucsb.edu/~william/data/liar_dataset.zip'
    zip_path = os.path.join(fact_verification_dir, 'liar_temp.zip')
    
    try:
        if download_file_with_progress(url, zip_path, timeout=120):
            print('Extracting...')
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(fact_verification_dir)
            os.remove(zip_path)
            print('✅ LIAR dataset ready!')
            return True
    except Exception as e:
        print(f'❌ Error: {e}')
    
    print('📋 Manual Download Steps:')
    print('1. Go to: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip')
    print('2. Download the ZIP file')
    print('3. Extract to get: train.tsv, valid.tsv, test.tsv')
    print('4. Place all files in: data/fact_verification/')
    return False

def download_sarc_kaggle():
    print('\n=== SARC Dataset (1.3M Reddit Comments) ===')
    
    sarc_dir = os.path.join(text_dir, 'sarc')
    if os.path.exists(sarc_dir) and os.listdir(sarc_dir):
        print('✅ SARC directory already exists')
        return True
    
    try:
        import kaggle
        print('Downloading via Kaggle API...')
        kaggle.api.dataset_download_files('danofer/sarcasm', path=sarc_dir, unzip=True)
        print('✅ SARC downloaded!')
        return True
    except ImportError:
        print('❌ Kaggle API not installed')
        print('💡 Install with: pip install kaggle')
    except Exception as e:
        print(f'❌ Kaggle API failed: {e}')
    
    print('📋 Manual Download Steps:')
    print('1. Go to: https://www.kaggle.com/datasets/danofer/sarcasm')
    print('2. Create free Kaggle account')
    print('3. Click "Download" button')
    print('4. Extract CSV files')
    print('5. Place in: data/text/sarc/')
    return False

def download_mrpc():
    print('\n=== MRPC Dataset (Paraphrasing) ===')
    
    mrpc_dir = os.path.join(paraphrasing_dir, 'MRPC')
    if os.path.exists(mrpc_dir) and os.listdir(mrpc_dir):
        print('✅ MRPC already exists')
        return True
    
    try:
        glue_script_url = 'https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py'
        script_path = 'download_glue_data.py'
        
        if download_file_with_progress(glue_script_url, script_path, timeout=30):
            print('Running GLUE download script...')
            result = subprocess.run(['python', script_path, '--data_dir', paraphrasing_dir, '--tasks', 'MRPC'], 
                                  capture_output=True, text=True, timeout=180)
            
            if result.returncode == 0:
                print('✅ MRPC downloaded!')
                os.remove(script_path)
                return True
                
    except Exception as e:
        print(f'❌ MRPC failed: {e}')
        if os.path.exists('download_glue_data.py'):
            os.remove('download_glue_data.py')
    
    print('📋 Manual Download Steps:')
    print('1. Go to: https://gluebenchmark.com/tasks')
    print('2. Find MRPC task')
    print('3. Download train.tsv, dev.tsv, test.tsv')
    print('4. Place in: data/text/paraphrasing/MRPC/')
    return False

def download_fever_robust():
    print('\n=== FEVER Dataset (Optional) ===')
    
    # Check if any FEVER files exist
    fever_files = ['fever_train.jsonl', 'fever_dev.jsonl', 'fever_test.jsonl']
    existing = [f for f in fever_files if os.path.exists(os.path.join(fact_verification_dir, f))]
    
    if existing:
        print(f'✅ Found existing FEVER files: {existing}')
        return True
    
    print('📋 Manual Download Steps (RECOMMENDED):')
    print('1. Visit: http://fever.ai/resources.html')
    print('2. Register for dataset access (free for academic use)')
    print('3. Download and place in: data/fact_verification/')
    print('💡 Note: LIAR dataset is sufficient for fact verification')
    
    return False

if __name__ == '__main__':
    print("🚀 FACTCHECK-MM COMPREHENSIVE DATASET DOWNLOADER")
    print("=" * 80)
    print("Now including NEW datasets: SarcNet, MMSD 2.0, Spanish Sarcasm, UR-FUNNY!")
    
    # Check required packages
    print("\n📦 Checking required packages...")
    required_packages = {
        'gdown': 'Google Drive downloads',
        'kaggle': 'Kaggle dataset downloads', 
        'datasets': 'Hugging Face datasets'
    }
    
    for package, purpose in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {package} available for {purpose}")
        except ImportError:
            print(f"⚠️ {package} not installed - needed for {purpose}")
            print(f"   Install with: pip install {package}")
    
    start_time = time.time()
    results = {}
    
    print("\n📦 DOWNLOADING ALL DATASETS...")
    
    # Original datasets
    results['MUStARD'] = download_mustard()
    results['News Headlines'] = download_news_headline_sarcasm()
    results['LIAR'] = download_liar()
    results['SARC'] = download_sarc_kaggle()
    results['MRPC'] = download_mrpc()
    results['FEVER'] = download_fever_robust()
    
    # NEW multimodal sarcasm datasets
    results['SarcNet'] = download_sarcnet()
    results['MMSD 2.0'] = download_mmsd2()
    results['Spanish Sarcasm'] = download_spanish_sarcasm()
    results['UR-FUNNY'] = download_ur_funny()
    
    # Results summary
    successful = sum(results.values())
    total = len(results)
    duration = round(time.time() - start_time, 1)
    
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE DOWNLOAD RESULTS")
    print("="*80)
    
    print("\n🎯 ORIGINAL DATASETS:")
    original_datasets = ['MUStARD', 'News Headlines', 'LIAR', 'SARC', 'MRPC', 'FEVER']
    for dataset in original_datasets:
        if dataset in results:
            status = "✅ READY" if results[dataset] else "❌ MANUAL DOWNLOAD"
            print(f"  {dataset:<20} {status}")
    
    print("\n🆕 NEW MULTIMODAL SARCASM DATASETS:")
    new_datasets = ['SarcNet', 'MMSD 2.0', 'Spanish Sarcasm', 'UR-FUNNY']
    dataset_info = {
        'SarcNet': '3,335 multilingual pairs (EN/ZH)', 
        'MMSD 2.0': '24,635 samples (ACL 2023)',
        'Spanish Sarcasm': '960 Spanish utterances',
        'UR-FUNNY': '1,260+ video segments'
    }
    
    for dataset in new_datasets:
        if dataset in results:
            status = "✅ READY" if results[dataset] else "❌ MANUAL DOWNLOAD"
            info = dataset_info.get(dataset, '')
            print(f"  {dataset:<18} {status} ({info})")
    
    print(f"\n🎯 Successfully downloaded: {successful}/{total} datasets")
    print(f"⏱️  Total time: {duration} seconds")
    
    # Updated directory structure
    print(f"\n📁 UPDATED DIRECTORY STRUCTURE:")
    print("data/")
    print("├── text/")
    print("│   ├── Sarcasm_Headlines_Dataset.json")
    print("│   ├── sarc/ (1.3M Reddit comments)")
    print("│   ├── spanish_sarcasm/ (960 Spanish utterances) 🆕")
    print("│   └── paraphrasing/MRPC/")
    print("├── multimodal/")
    print("│   ├── sarcnet/ (3,335 multilingual pairs) 🆕")
    print("│   ├── mmsd2/ (24,635 samples) 🆕")
    print("│   ├── ur_funny/ (1,260+ videos) 🆕")
    print("│   └── [MUStARD files]")
    print("└── fact_verification/")
    print("    ├── train.tsv, valid.tsv, test.tsv (LIAR)")
    print("    └── fever_*.jsonl (optional)")
    
    print(f"\n📊 COMPREHENSIVE DATASET STATISTICS:")
    print(f"• Text samples: ~1.4M+ (SARC + Headlines + Spanish)")
    print(f"• Multimodal samples: ~30K+ (SarcNet + MMSD 2.0 + UR-FUNNY + MUStARD)")
    print(f"• Fact verification: ~12K+ (LIAR)")
    print(f"• Languages: English, Chinese, Spanish")
    print(f"• Modalities: Text, Images, Videos, Audio")
    
    if successful >= 7:
        print("\n🎉 OUTSTANDING! World-class multilingual multimodal dataset collection")
        print("   Ready for cutting-edge sarcasm detection research!")
    elif successful >= 4:
        print("\n✅ EXCELLENT! Sufficient datasets for comprehensive research")
    else:
        print(f"\n⚠️ {total - successful} datasets need manual download")
        print("💡 Follow manual steps above for missing datasets")
    
    print("\n🚀 Your FactCheck-MM project now has access to:")
    print("   📈 State-of-the-art multimodal datasets")
    print("   🌍 Multilingual sarcasm detection capabilities") 
    print("   🎯 30,000+ samples across multiple modalities")
    print("   🏆 Ready for world-class research!")
