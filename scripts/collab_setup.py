"""
Google Colab Setup Script for Pusher-Gym Project

This script handles:
- Installation of dependencies
- Mounting Google Drive (for saving models/videos)
- Cloning/syncing the project
- Setting up paths
"""

import os
import sys
from pathlib import Path

def setup_colab_environment(mount_drive=True, project_name="Pusher-Gym"):
    """
    Set up the Colab environment for training.
    
    Args:
        mount_drive: Whether to mount Google Drive
        project_name: Name of the project folder
    
    Returns:
        Dictionary with important paths
    """
    print("🚀 Setting up Colab environment for Pusher-Gym\n")
    print("=" * 80)
    
    # Check if we're in Colab
    try:
        import google.colab
        IN_COLAB = True
        print("✅ Running in Google Colab")
    except ImportError:
        IN_COLAB = False
        print("ℹ️  Running locally")
    
    # Mount Google Drive if requested
    drive_root = None
    if IN_COLAB and mount_drive:
        print("\n📁 Mounting Google Drive...")
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        drive_root = Path('/content/drive/MyDrive')
        print(f"✅ Drive mounted at: {drive_root}")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    
    packages = [
        "gymnasium[mujoco]",
        "torch",
        "numpy",
        "matplotlib",
        "seaborn",
        "pandas",
        "tqdm",
        "imageio",
        "imageio-ffmpeg",
    ]
    
    for package in packages:
        os.system(f"pip install -q {package}")
    
    print("✅ Dependencies installed")
    
    # Set up project structure
    if IN_COLAB:
        project_root = Path('/content') / project_name
        
        # Create project directory
        project_root.mkdir(exist_ok=True)
        
        # Create subdirectories
        (project_root / 'src' / 'agents').mkdir(parents=True, exist_ok=True)
        (project_root / 'src' / 'utils').mkdir(parents=True, exist_ok=True)
        (project_root / 'src' / 'environments').mkdir(parents=True, exist_ok=True)
        (project_root / 'checkpoints').mkdir(exist_ok=True)
        (project_root / 'videos').mkdir(exist_ok=True)
        (project_root / 'logs').mkdir(exist_ok=True)
        (project_root / 'results').mkdir(exist_ok=True)
        
        # If Drive is mounted, create symlinks to save there
        if drive_root:
            drive_project = drive_root / project_name
            drive_project.mkdir(exist_ok=True)
            
            # Create subdirectories in Drive
            (drive_project / 'checkpoints').mkdir(exist_ok=True)
            (drive_project / 'videos').mkdir(exist_ok=True)
            (drive_project / 'logs').mkdir(exist_ok=True)
            (drive_project / 'results').mkdir(exist_ok=True)
            
            print(f"\n💾 Project files will be saved to: {drive_project}")
    else:
        project_root = Path.cwd()
        drive_project = None
    
    # Add project to path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    print(f"\n✅ Project root: {project_root}")
    
    # Create paths dictionary
    paths = {
        'project_root': project_root,
        'drive_root': drive_root,
        'drive_project': drive_project,
        'checkpoints': drive_project / 'checkpoints' if drive_project else project_root / 'checkpoints',
        'videos': drive_project / 'videos' if drive_project else project_root / 'videos',
        'logs': drive_project / 'logs' if drive_project else project_root / 'logs',
        'results': drive_project / 'results' if drive_project else project_root / 'results',
    }
    
    print("\n📋 Project paths:")
    for name, path in paths.items():
        if path is not None:
            print(f"  {name:20s}: {path}")
    
    print("\n" + "=" * 80)
    print("✅ Setup complete! Ready to train.\n")
    
    return paths


def check_gpu():
    """Check GPU availability."""
    import torch
    
    print("\n🖥️  Hardware Check:")
    print("=" * 80)
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU Available: {device_name}")
        print(f"   Memory: {memory:.2f} GB")
        device = "cuda"
    else:
        print("⚠️  No GPU available, using CPU")
        device = "cpu"
    
    print("=" * 80)
    return device


def upload_source_files():
    """
    Upload source files to Colab.
    Prompts user to upload key files if not already present.
    """
    try:
        from google.colab import files
        
        print("\n📤 Upload Source Files")
        print("=" * 80)
        print("Please upload the following files:")
        print("  1. sac.py (SAC agent)")
        print("  2. replay_buffer.py (Replay buffer)")
        print("\nOr skip if files are already in place.")
        print("=" * 80)
        
        uploaded = files.upload()
        
        if uploaded:
            print(f"\n✅ Uploaded {len(uploaded)} file(s)")
            return uploaded
        else:
            print("\n⏭️  Skipped upload")
            return None
            
    except ImportError:
        print("ℹ️  Not in Colab, skipping upload")
        return None


if __name__ == "__main__":
    # Run setup
    paths = setup_colab_environment(mount_drive=True)
    device = check_gpu()
    
    print("\n🎯 Quick Start:")
    print("=" * 80)
    print("1. Upload your source files (sac.py, replay_buffer.py, etc.)")
    print("2. Run training script:")
    print("   python train_sac.py --episodes 2000")
    print("\n3. Or use the training notebook:")
    print("   Open: colab_training.ipynb")
    print("=" * 80)