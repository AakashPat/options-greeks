#!/usr/bin/env python3
"""
Setup script for Options Greeks Calculator & Gamma Scalping Backtester

This script helps set up the development environment and dependencies.
"""

import os
import sys
import subprocess
import shutil

def check_python_version():
    """Check if Python version is 3.8 or higher."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_dependencies():
    """Install required Python packages."""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print(f"   Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        return False

def setup_environment():
    """Set up environment file."""
    env_file = ".env"
    env_example = "env.example"
    
    if os.path.exists(env_file):
        print(f"⚠️  {env_file} already exists, skipping...")
        return True
    
    if os.path.exists(env_example):
        try:
            shutil.copy2(env_example, env_file)
            print(f"✅ Created {env_file} from {env_example}")
            print(f"   Please edit {env_file} with your actual API credentials")
            return True
        except Exception as e:
            print(f"❌ Failed to create {env_file}: {e}")
            return False
    else:
        # Create basic .env file
        env_content = """# FutPrint API Configuration
# Replace these with your actual API credentials

API_OHLCV_URL=https://api.futprint.in/api/historical-data-ohlcv-oi
API_TOKEN=<YOUR_TOKEN>
API_KEY=FutPrintIN
RISK_FREE_RATE=0.06
DEFAULT_LOT_SIZE=50
DEFAULT_FEE_PER_CONTRACT=10.0
DEFAULT_SLIPPAGE_TICKS=1
DEFAULT_TICK_SIZE=1.0
"""
        try:
            with open(env_file, 'w') as f:
                f.write(env_content)
            print(f"✅ Created basic {env_file}")
            print(f"   Please edit {env_file} with your actual API credentials")
            return True
        except Exception as e:
            print(f"❌ Failed to create {env_file}: {e}")
            return False

def verify_installation():
    """Verify that key packages can be imported."""
    print("\n🔍 Verifying installation...")
    packages = ['pandas', 'numpy', 'requests', 'scipy', 'matplotlib', 'dotenv']
    
    for package in packages:
        try:
            if package == 'dotenv':
                __import__('dotenv')
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - import failed")
            return False
    
    return True

def main():
    """Main setup function."""
    print("🚀 Setting up Options Greeks Calculator & Gamma Scalping Backtester")
    print("=" * 65)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\n⚠️  Some packages failed to import. Please check the installation.")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file with your FutPrint API credentials")
    print("2. Run: python main.py")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
