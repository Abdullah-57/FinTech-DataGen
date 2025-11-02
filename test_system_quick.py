#!/usr/bin/env python3
"""
Quick test to verify the adaptive learning system is working
"""

import requests
import json
import time

def test_backend_health():
    """Test if backend is running"""
    try:
        response = requests.get("http://localhost:5000/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running and healthy")
            return True
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Is it running on port 5000?")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_registration():
    """Test symbol registration"""
    print("\n🔄 Testing symbol registration...")
    
    payload = {
        "symbol": "AAPL",
        "model_types": ["sgd", "lstm", "ensemble"]
    }
    
    try:
        response = requests.post(
            "http://localhost:5000/api/adaptive/register",
            json=payload,
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Registration successful!")
            print(json.dumps(data, indent=2))
            return True
        else:
            print(f"❌ Registration failed")
            try:
                error_data = response.json()
                print(json.dumps(error_data, indent=2))
            except:
                print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ Registration error: {e}")
        return False

def test_system_status():
    """Test system status"""
    print("\n🔄 Testing system status...")
    
    try:
        response = requests.get("http://localhost:5000/api/adaptive/status", timeout=10)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ System status retrieved!")
            print(json.dumps(data, indent=2))
            return True
        else:
            print(f"❌ System status failed")
            try:
                error_data = response.json()
                print(json.dumps(error_data, indent=2))
            except:
                print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ System status error: {e}")
        return False

def test_database_stats():
    """Test database statistics"""
    print("\n🔄 Testing database statistics...")
    
    try:
        response = requests.get("http://localhost:5000/api/adaptive/stats", timeout=10)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Database stats retrieved!")
            print(json.dumps(data, indent=2))
            return True
        else:
            print(f"❌ Database stats failed")
            try:
                error_data = response.json()
                print(json.dumps(error_data, indent=2))
            except:
                print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ Database stats error: {e}")
        return False

def main():
    """Run quick tests"""
    print("🧪 Quick Adaptive Learning System Test")
    print("=" * 50)
    
    # Test backend health
    if not test_backend_health():
        print("\n❌ Backend is not running. Please start it with:")
        print("   cd backend")
        print("   python app.py")
        return
    
    # Test registration
    registration_ok = test_registration()
    
    # Test system status
    status_ok = test_system_status()
    
    # Test database stats
    db_stats_ok = test_database_stats()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Backend Health: ✅ OK")
    print(f"   Registration: {'✅ OK' if registration_ok else '❌ FAILED'}")
    print(f"   System Status: {'✅ OK' if status_ok else '❌ FAILED'}")
    print(f"   Database Stats: {'✅ OK' if db_stats_ok else '❌ FAILED'}")
    
    if registration_ok and status_ok and db_stats_ok:
        print("\n🎉 All tests passed! The system is working correctly.")
        print("\nYou can now:")
        print("1. Open http://localhost:3000 in your browser")
        print("2. Navigate to 'Adaptive Learning' page")
        print("3. Try all the features in the showcase")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
        print("Common issues:")
        print("- Missing Python packages (run: python install_dependencies.py)")
        print("- MongoDB not running")
        print("- Import path issues")

if __name__ == "__main__":
    main()