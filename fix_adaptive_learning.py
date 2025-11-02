#!/usr/bin/env python3
"""
Quick fix script for adaptive learning system issues
"""

import os
import sys

def check_file_exists(filepath):
    """Check if a file exists and is not empty"""
    if not os.path.exists(filepath):
        return False, f"File does not exist: {filepath}"
    
    if os.path.getsize(filepath) == 0:
        return False, f"File is empty: {filepath}"
    
    return True, f"File OK: {filepath}"

def check_imports():
    """Check if all required modules can be imported"""
    print("🔍 Checking imports...")
    
    try:
        import schedule
        print("✅ schedule module imported successfully")
    except ImportError:
        print("❌ schedule module not found - run: pip install schedule")
        return False
    
    try:
        import sklearn
        print("✅ scikit-learn imported successfully")
    except ImportError:
        print("❌ scikit-learn not found - run: pip install scikit-learn")
        return False
    
    try:
        import tensorflow
        print("✅ tensorflow imported successfully")
    except ImportError:
        print("❌ tensorflow not found - run: pip install tensorflow")
        return False
    
    try:
        import joblib
        print("✅ joblib imported successfully")
    except ImportError:
        print("❌ joblib not found - run: pip install joblib")
        return False
    
    return True

def check_files():
    """Check if all required files exist"""
    print("\n📁 Checking files...")
    
    required_files = [
        "backend/ml_models/online_learning.py",
        "backend/ml_models/adaptive_learning.py", 
        "backend/ml_models/continuous_learning.py",
        "backend/database/mongodb.py",
        "backend/app.py"
    ]
    
    all_good = True
    for filepath in required_files:
        exists, message = check_file_exists(filepath)
        if exists:
            print(f"✅ {message}")
        else:
            print(f"❌ {message}")
            all_good = False
    
    return all_good

def test_imports_in_files():
    """Test if the imports in the files work"""
    print("\n🧪 Testing file imports...")
    
    # Test if we can import the modules
    sys.path.append('backend')
    
    try:
        from ml_models.online_learning import OnlineSGDRegressor
        print("✅ OnlineSGDRegressor imported successfully")
    except Exception as e:
        print(f"❌ Failed to import OnlineSGDRegressor: {e}")
        return False
    
    try:
        from ml_models.adaptive_learning import AdaptiveLearningManager
        print("✅ AdaptiveLearningManager imported successfully")
    except Exception as e:
        print(f"❌ Failed to import AdaptiveLearningManager: {e}")
        return False
    
    try:
        from ml_models.continuous_learning import ContinuousLearningManager
        print("✅ ContinuousLearningManager imported successfully")
    except Exception as e:
        print(f"❌ Failed to import ContinuousLearningManager: {e}")
        return False
    
    try:
        from database.mongodb import MongoDB
        print("✅ MongoDB imported successfully")
    except Exception as e:
        print(f"❌ Failed to import MongoDB: {e}")
        return False
    
    return True

def main():
    """Run all checks"""
    print("🔧 Adaptive Learning System - Quick Fix Check")
    print("=" * 50)
    
    # Check imports
    imports_ok = check_imports()
    
    # Check files
    files_ok = check_files()
    
    # Test file imports
    file_imports_ok = test_imports_in_files()
    
    print("\n" + "=" * 50)
    print("📊 Summary:")
    print(f"   Dependencies: {'✅ OK' if imports_ok else '❌ ISSUES'}")
    print(f"   Files: {'✅ OK' if files_ok else '❌ MISSING'}")
    print(f"   File Imports: {'✅ OK' if file_imports_ok else '❌ ISSUES'}")
    
    if imports_ok and files_ok and file_imports_ok:
        print("\n🎉 All checks passed! The adaptive learning system should work now.")
        print("\nNext steps:")
        print("1. Restart the backend server: cd backend && python app.py")
        print("2. Open the frontend: http://localhost:3000")
        print("3. Navigate to 'Adaptive Learning' page")
        print("4. Try registering a symbol (AAPL)")
    else:
        print("\n⚠️  Issues detected. Please fix the following:")
        if not imports_ok:
            print("   - Install missing Python packages")
        if not files_ok:
            print("   - Ensure all required files exist")
        if not file_imports_ok:
            print("   - Fix import errors in the files")
        
        print("\nTo install missing packages, run:")
        print("   python install_dependencies.py")

if __name__ == "__main__":
    main()