#!/usr/bin/env python3
"""
Quick API Test Script for Adaptive Learning System

This script tests all the adaptive learning API endpoints to verify
the system is working correctly. Run this after starting the backend server.

Usage: python test_adaptive_api.py
"""

import requests
import json
import time
import sys
from datetime import datetime

class AdaptiveLearningAPITester:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.symbol = "AAPL"
        self.model_types = ["sgd", "lstm", "ensemble"]
        
    def print_section(self, title):
        """Print a formatted section header"""
        print("\n" + "="*60)
        print(f" {title}")
        print("="*60)
    
    def print_result(self, response, title="Response"):
        """Print formatted API response"""
        print(f"\n{title}:")
        print(f"Status Code: {response.status_code}")
        
        try:
            data = response.json()
            print(json.dumps(data, indent=2, default=str))
        except:
            print(response.text)
        
        return response.status_code == 200
    
    def test_health_check(self):
        """Test backend health"""
        self.print_section("1. HEALTH CHECK")
        
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=5)
            success = self.print_result(response, "Health Check Result")
            
            if success:
                print("✅ Backend is healthy and connected to database")
            else:
                print("❌ Backend health check failed")
                
            return success
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to backend. Make sure it's running on port 5000")
            return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_registration(self):
        """Test symbol registration"""
        self.print_section("2. SYMBOL REGISTRATION")
        
        payload = {
            "symbol": self.symbol,
            "model_types": self.model_types
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/adaptive/register",
                json=payload,
                timeout=10
            )
            
            success = self.print_result(response, "Registration Result")
            
            if success:
                print(f"✅ Successfully registered {self.symbol} for adaptive learning")
            else:
                print(f"❌ Failed to register {self.symbol}")
                
            return success
        except Exception as e:
            print(f"❌ Registration error: {e}")
            return False
    
    def test_training(self):
        """Test model training"""
        self.print_section("3. MODEL TRAINING")
        
        training_results = {}
        
        for model_type in self.model_types:
            print(f"\n🔄 Training {model_type.upper()} model...")
            
            payload = {
                "symbol": self.symbol,
                "model_type": model_type
            }
            
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/api/adaptive/train",
                    json=payload,
                    timeout=120  # Longer timeout for training
                )
                end_time = time.time()
                
                success = self.print_result(response, f"{model_type.upper()} Training Result")
                training_time = end_time - start_time
                
                if success:
                    data = response.json()
                    if data.get('status') == 'success':
                        metrics = data.get('metrics', {})
                        print(f"✅ {model_type.upper()} trained successfully in {training_time:.2f}s!")
                        print(f"   Version: {data.get('version', 'N/A')}")
                        print(f"   RMSE: {metrics.get('rmse', 'N/A')}")
                        print(f"   MAE: {metrics.get('mae', 'N/A')}")
                        print(f"   MAPE: {metrics.get('mape', 'N/A')}")
                        training_results[model_type] = True
                    else:
                        print(f"❌ {model_type.upper()} training failed: {data.get('message', 'Unknown error')}")
                        training_results[model_type] = False
                else:
                    print(f"❌ {model_type.upper()} training request failed")
                    training_results[model_type] = False
                    
            except Exception as e:
                print(f"❌ {model_type.upper()} training error: {e}")
                training_results[model_type] = False
        
        return training_results
    
    def test_predictions(self):
        """Test model predictions"""
        self.print_section("4. PREDICTIONS")
        
        horizons = [1, 5, 10]
        prediction_results = {}
        
        for model_type in self.model_types:
            print(f"\n📈 Testing {model_type.upper()} predictions...")
            prediction_results[model_type] = {}
            
            for horizon in horizons:
                payload = {
                    "symbol": self.symbol,
                    "model_type": model_type,
                    "horizon": horizon
                }
                
                try:
                    response = requests.post(
                        f"{self.base_url}/api/adaptive/predict",
                        json=payload,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('status') == 'success':
                            predictions = data.get('predictions', [])
                            print(f"   ✅ {horizon}-day forecast: {len(predictions)} predictions")
                            if predictions:
                                print(f"      First prediction: ${predictions[0]:.2f}")
                                if len(predictions) > 1:
                                    print(f"      Last prediction: ${predictions[-1]:.2f}")
                            prediction_results[model_type][horizon] = True
                        else:
                            print(f"   ❌ {horizon}-day forecast failed: {data.get('message', 'Unknown error')}")
                            prediction_results[model_type][horizon] = False
                    else:
                        print(f"   ❌ {horizon}-day forecast request failed (status: {response.status_code})")
                        prediction_results[model_type][horizon] = False
                        
                except Exception as e:
                    print(f"   ❌ {horizon}-day forecast error: {e}")
                    prediction_results[model_type][horizon] = False
        
        return prediction_results
    
    def test_system_status(self):
        """Test system status"""
        self.print_section("5. SYSTEM STATUS")
        
        try:
            response = requests.get(f"{self.base_url}/api/adaptive/status", timeout=10)
            success = self.print_result(response, "System Status")
            
            if success:
                data = response.json()
                print(f"✅ System status retrieved successfully")
                print(f"   Continuous learning running: {data.get('is_running', False)}")
                print(f"   Registered models: {data.get('registered_models', 0)}")
                
                models = data.get('models', {})
                if models:
                    print(f"   Active models: {len(models)}")
                    for model_key, model_info in models.items():
                        print(f"     {model_key}: v{model_info.get('current_version', 0)}")
            else:
                print("❌ Failed to get system status")
                
            return success
        except Exception as e:
            print(f"❌ System status error: {e}")
            return False
    
    def test_performance_monitoring(self):
        """Test performance monitoring"""
        self.print_section("6. PERFORMANCE MONITORING")
        
        performance_results = {}
        
        for model_type in self.model_types:
            print(f"\n📊 Getting {model_type.upper()} performance...")
            
            try:
                response = requests.get(
                    f"{self.base_url}/api/adaptive/performance/{self.symbol}/{model_type}",
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'success':
                        summary = data.get('performance_summary', {})
                        print(f"   ✅ Performance data retrieved")
                        print(f"      Total versions: {summary.get('total_versions', 0)}")
                        print(f"      Current version: {summary.get('current_version', 0)}")
                        print(f"      Best RMSE: {summary.get('best_rmse', 'N/A')}")
                        print(f"      Latest RMSE: {summary.get('latest_rmse', 'N/A')}")
                        
                        history = data.get('version_history', [])
                        if history:
                            print(f"      Version history: {len(history)} versions")
                        
                        performance_results[model_type] = True
                    else:
                        print(f"   ❌ Performance data failed: {data.get('message', 'Unknown error')}")
                        performance_results[model_type] = False
                else:
                    print(f"   ❌ Performance request failed (status: {response.status_code})")
                    performance_results[model_type] = False
                    
            except Exception as e:
                print(f"   ❌ Performance monitoring error: {e}")
                performance_results[model_type] = False
        
        return performance_results
    
    def test_database_stats(self):
        """Test database statistics"""
        self.print_section("7. DATABASE STATISTICS")
        
        try:
            response = requests.get(f"{self.base_url}/api/adaptive/stats", timeout=10)
            success = self.print_result(response, "Database Statistics")
            
            if success:
                data = response.json()
                print("✅ Database statistics retrieved successfully")
                print(f"   Model versions: {data.get('total_model_versions', 0)}")
                print(f"   Training events: {data.get('total_training_events', 0)}")
                print(f"   Active models: {data.get('active_models', 0)}")
                print(f"   Recent versions (7d): {data.get('recent_versions', 0)}")
            else:
                print("❌ Failed to get database statistics")
                
            return success
        except Exception as e:
            print(f"❌ Database statistics error: {e}")
            return False
    
    def test_continuous_learning_control(self):
        """Test continuous learning start/stop"""
        self.print_section("8. CONTINUOUS LEARNING CONTROL")
        
        # Test start
        print("🚀 Testing start continuous learning...")
        try:
            response = requests.post(f"{self.base_url}/api/adaptive/start", timeout=10)
            start_success = self.print_result(response, "Start Continuous Learning")
            
            if start_success:
                print("✅ Continuous learning started successfully")
            else:
                print("❌ Failed to start continuous learning")
        except Exception as e:
            print(f"❌ Start continuous learning error: {e}")
            start_success = False
        
        time.sleep(2)  # Brief pause
        
        # Test stop
        print("\n⏸️  Testing stop continuous learning...")
        try:
            response = requests.post(f"{self.base_url}/api/adaptive/stop", timeout=10)
            stop_success = self.print_result(response, "Stop Continuous Learning")
            
            if stop_success:
                print("✅ Continuous learning stopped successfully")
            else:
                print("❌ Failed to stop continuous learning")
        except Exception as e:
            print(f"❌ Stop continuous learning error: {e}")
            stop_success = False
        
        return start_success and stop_success
    
    def run_complete_test(self):
        """Run all tests"""
        print("🧠 ADAPTIVE LEARNING API TEST SUITE")
        print("=" * 60)
        print("Testing all adaptive learning API endpoints...")
        print(f"Backend URL: {self.base_url}")
        print(f"Test Symbol: {self.symbol}")
        print(f"Model Types: {', '.join(self.model_types)}")
        
        results = {}
        
        # Run tests in sequence
        results['health'] = self.test_health_check()
        
        if not results['health']:
            print("\n❌ CRITICAL: Backend health check failed. Cannot continue testing.")
            print("Please ensure:")
            print("1. Backend server is running: python backend/app.py")
            print("2. MongoDB is running and accessible")
            print("3. No firewall blocking port 5000")
            return False
        
        results['registration'] = self.test_registration()
        results['training'] = self.test_training()
        results['predictions'] = self.test_predictions()
        results['status'] = self.test_system_status()
        results['performance'] = self.test_performance_monitoring()
        results['database'] = self.test_database_stats()
        results['continuous'] = self.test_continuous_learning_control()
        
        # Print summary
        self.print_section("TEST SUMMARY")
        
        total_tests = 0
        passed_tests = 0
        
        print("📊 Test Results:")
        
        # Health check
        status = "✅ PASS" if results['health'] else "❌ FAIL"
        print(f"   Health Check: {status}")
        total_tests += 1
        if results['health']: passed_tests += 1
        
        # Registration
        status = "✅ PASS" if results['registration'] else "❌ FAIL"
        print(f"   Symbol Registration: {status}")
        total_tests += 1
        if results['registration']: passed_tests += 1
        
        # Training
        if isinstance(results['training'], dict):
            training_passed = sum(results['training'].values())
            training_total = len(results['training'])
            print(f"   Model Training: {training_passed}/{training_total} models trained")
            total_tests += training_total
            passed_tests += training_passed
        
        # Predictions
        if isinstance(results['predictions'], dict):
            pred_passed = sum(sum(model_results.values()) for model_results in results['predictions'].values())
            pred_total = sum(len(model_results) for model_results in results['predictions'].values())
            print(f"   Predictions: {pred_passed}/{pred_total} prediction tests passed")
            total_tests += pred_total
            passed_tests += pred_passed
        
        # Other tests
        for test_name, test_key in [
            ("System Status", "status"),
            ("Performance Monitoring", "performance"),
            ("Database Statistics", "database"),
            ("Continuous Learning", "continuous")
        ]:
            if isinstance(results[test_key], dict):
                test_passed = sum(results[test_key].values())
                test_total = len(results[test_key])
                print(f"   {test_name}: {test_passed}/{test_total} tests passed")
                total_tests += test_total
                passed_tests += test_passed
            else:
                status = "✅ PASS" if results[test_key] else "❌ FAIL"
                print(f"   {test_name}: {status}")
                total_tests += 1
                if results[test_key]: passed_tests += 1
        
        # Overall result
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n📈 Overall Results:")
        print(f"   Tests Passed: {passed_tests}/{total_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("\n🎉 EXCELLENT! The adaptive learning system is working perfectly!")
            print("✅ All major features are functional and ready for use.")
        elif success_rate >= 70:
            print("\n👍 GOOD! Most features are working correctly.")
            print("⚠️  Some minor issues detected - check failed tests above.")
        else:
            print("\n⚠️  ISSUES DETECTED! Several tests failed.")
            print("❌ Please review the errors above and check your setup.")
        
        print(f"\n🕒 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return success_rate >= 70


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Adaptive Learning API')
    parser.add_argument('--url', default='http://localhost:5000', 
                       help='Backend URL (default: http://localhost:5000)')
    parser.add_argument('--symbol', default='AAPL',
                       help='Symbol to test with (default: AAPL)')
    
    args = parser.parse_args()
    
    tester = AdaptiveLearningAPITester(base_url=args.url)
    tester.symbol = args.symbol
    
    try:
        success = tester.run_complete_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()