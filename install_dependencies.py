#!/usr/bin/env python3
"""
Install missing dependencies for the adaptive learning system
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    """Install required packages"""
    print("🔧 Installing missing dependencies for adaptive learning...")
    
    packages = [
        "schedule==1.2.0",
        "scikit-learn==1.3.0",
        "tensorflow==2.13.0",
        "joblib==1.3.2"
    ]
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 Installation Summary:")
    print(f"   Successful: {success_count}/{len(packages)}")
    
    if success_count == len(packages):
        print("🎉 All dependencies installed successfully!")
        print("\nYou can now restart the backend server:")
        print("   cd backend")
        print("   python app.py")
    else:
        print("⚠️  Some packages failed to install. Please install them manually:")
        for package in packages:
            print(f"   pip install {package}")

if __name__ == "__main__":
    main()