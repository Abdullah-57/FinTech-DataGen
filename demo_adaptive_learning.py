"""
Adaptive Learning Demo Script

This script demonstrates the complete adaptive learning and continuous evaluation
system for financial forecasting. It shows how to use the API endpoints and
showcases the key features of the system.
"""

import requests
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class AdaptiveLearningDemo:
    """Demo class for adaptive learning system"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.symbol = "AAPL"
        self.model_types = ["sgd", "lstm", "ensemble"]
    
    def print_section(self, title):
        """Print a formatted section header"""
        print("\n" + "="*60)
        print(f" {title}")
        print("="*60)
    
    def print_response(self, response, title="Response"):
        """Print formatted API response"""
        print(f"\n{title}:")
        if response.status_code == 200:
            data = response.json()
            print(json.dumps(data, indent=2, default=str))
        else:
            print(f"Error {response.status_code}: {response.text}")
    
    def check_backend_health(self):
        """Check if backend is running"""
        try:
            response = requests.get(f"{self.base_url}/api/health")
            if response.status_code == 200:
                print("✅ Backend is running and healthy")
                return True
            else:
                print("❌ Backend health check failed")
                return False
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to backend. Make sure it's running on port 5000")
            return False
    
    def demo_registration(self):
        """Demo: Register symbol for adaptive learning"""
        self.print_section("1. SYMBOL REGISTRATION")
        
        print(f"Registering {self.symbol} for adaptive learning with models: {self.model_types}")
        
        payload = {
            "symbol": self.symbol,
            "model_types": self.model_types
        }
        
        response = requests.post(
            f"{self.base_url}/api/adaptive/register",
            json=payload
        )
        
        self.print_response(response, "Registration Result")
        return response.status_code == 200
    
    def demo_initial_training(self):
        """Demo: Initial training for each model type"""
        self.print_section("2. INITIAL TRAINING")
        
        for model_type in self.model_types:
            print(f"\n🔄 Training {model_type.upper()} model for {self.symbol}...")
            
            payload = {
                "symbol": self.symbol,
                "model_type": model_type
            }
            
            response = requests.post(
                f"{self.base_url}/api/adaptive/train",
                json=payload
            )
            
            self.print_response(response, f"{model_type.upper()} Training Result")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    metrics = data.get('metrics', {})
                    print(f"✅ {model_type.upper()} trained successfully!")
                    print(f"   RMSE: {metrics.get('rmse', 'N/A'):.4f}")
                    print(f"   MAE: {metrics.get('mae', 'N/A'):.4f}")
                    print(f"   Version: {data.get('version', 'N/A')}")
            
            time.sleep(2)  # Brief pause between trainings
    
    def demo_predictions(self):
        """Demo: Make predictions with trained models"""
        self.print_section("3. MAKING PREDICTIONS")
        
        horizons = [1, 5, 10]
        
        for model_type in self.model_types:
            print(f"\n📈 Making predictions with {model_type.upper()} model...")
            
            for horizon in horizons:
                payload = {
                    "symbol": self.symbol,
                    "model_type": model_type,
                    "horizon": horizon
                }
                
                response = requests.post(
                    f"{self.base_url}/api/adaptive/predict",
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'success':
                        predictions = data.get('predictions', [])
                        print(f"   {horizon}-step forecast: {predictions[:3]}..." if len(predictions) > 3 else f"   {horizon}-step forecast: {predictions}")
                else:
                    print(f"   ❌ Prediction failed for horizon {horizon}")
    
    def demo_model_updates(self):
        """Demo: Manual model updates"""
        self.print_section("4. MODEL UPDATES")
        
        print("Triggering manual model updates to simulate new data arrival...")
        
        for model_type in self.model_types:
            print(f"\n🔄 Updating {model_type.upper()} model...")
            
            payload = {
                "symbol": self.symbol,
                "model_type": model_type
            }
            
            response = requests.post(
                f"{self.base_url}/api/adaptive/update",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    print(f"✅ {model_type.upper()} updated successfully!")
                    print(f"   Version: {data.get('version', 'N/A')}")
                    print(f"   New version created: {data.get('new_version_created', False)}")
                    
                    metrics = data.get('metrics', {})
                    if metrics:
                        print(f"   Updated RMSE: {metrics.get('rmse', 'N/A'):.4f}")
                else:
                    print(f"❌ Update failed: {data.get('message', 'Unknown error')}")
            else:
                print(f"❌ Update request failed")
    
    def demo_performance_monitoring(self):
        """Demo: Performance monitoring and version history"""
        self.print_section("5. PERFORMANCE MONITORING")
        
        for model_type in self.model_types:
            print(f"\n📊 Performance analysis for {model_type.upper()} model...")
            
            response = requests.get(
                f"{self.base_url}/api/adaptive/performance/{self.symbol}/{model_type}"
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    summary = data.get('performance_summary', {})
                    print(f"   Total versions: {summary.get('total_versions', 'N/A')}")
                    print(f"   Current version: {summary.get('current_version', 'N/A')}")
                    print(f"   Best RMSE: {summary.get('best_rmse', 'N/A'):.4f}")
                    print(f"   Latest RMSE: {summary.get('latest_rmse', 'N/A'):.4f}")
                    
                    # Show version history
                    history = data.get('version_history', [])
                    if history:
                        print(f"   Version history ({len(history)} versions):")
                        for version in history[-3:]:  # Show last 3 versions
                            metrics = version.get('metrics', {})
                            timestamp = version.get('timestamp', '')[:19]  # Remove microseconds
                            print(f"     v{version.get('version', 'N/A')}: RMSE={metrics.get('rmse', 'N/A'):.4f} ({timestamp})")
    
    def demo_system_status(self):
        """Demo: System status and statistics"""
        self.print_section("6. SYSTEM STATUS")
        
        print("Getting overall system status...")
        
        response = requests.get(f"{self.base_url}/api/adaptive/status")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Continuous learning running: {data.get('is_running', False)}")
            print(f"📊 Registered models: {data.get('registered_models', 0)}")
            
            models = data.get('models', {})
            if models:
                print("\n📈 Model Status:")
                for model_key, model_info in models.items():
                    print(f"   {model_key}:")
                    print(f"     Current version: {model_info.get('current_version', 'N/A')}")
                    print(f"     Total versions: {model_info.get('total_versions', 'N/A')}")
                    print(f"     Latest RMSE: {model_info.get('latest_rmse', 'N/A'):.4f}")
        
        # Get database statistics
        print("\n📊 Database Statistics:")
        response = requests.get(f"{self.base_url}/api/adaptive/stats")
        
        if response.status_code == 200:
            stats = response.json()
            print(f"   Model versions: {stats.get('total_model_versions', 0)}")
            print(f"   Training events: {stats.get('total_training_events', 0)}")
            print(f"   Active models: {stats.get('active_models', 0)}")
            print(f"   Recent versions (7 days): {stats.get('recent_versions', 0)}")
    
    def demo_continuous_learning_control(self):
        """Demo: Start/stop continuous learning"""
        self.print_section("7. CONTINUOUS LEARNING CONTROL")
        
        print("🚀 Starting continuous learning scheduler...")
        response = requests.post(f"{self.base_url}/api/adaptive/start")
        self.print_response(response, "Start Continuous Learning")
        
        if response.status_code == 200:
            print("✅ Continuous learning started!")
            print("   Models will now update automatically based on their schedules:")
            print("   - SGD: Every 6 hours")
            print("   - LSTM: Every 24 hours") 
            print("   - Ensemble: Every 12 hours")
            
            time.sleep(2)
            
            print("\n⏸️  Stopping continuous learning for demo...")
            response = requests.post(f"{self.base_url}/api/adaptive/stop")
            self.print_response(response, "Stop Continuous Learning")
    
    def demo_rollback(self):
        """Demo: Model rollback functionality"""
        self.print_section("8. MODEL ROLLBACK")
        
        # First, check if we have multiple versions to rollback to
        model_type = "sgd"  # Use SGD for rollback demo
        
        response = requests.get(
            f"{self.base_url}/api/adaptive/performance/{self.symbol}/{model_type}"
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                history = data.get('version_history', [])
                
                if len(history) >= 2:
                    # Rollback to previous version
                    target_version = history[-2]['version']  # Second to last version
                    
                    print(f"🔄 Rolling back {model_type.upper()} model to version {target_version}...")
                    
                    payload = {
                        "symbol": self.symbol,
                        "model_type": model_type,
                        "version": target_version
                    }
                    
                    response = requests.post(
                        f"{self.base_url}/api/adaptive/rollback",
                        json=payload
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('status') == 'success':
                            print(f"✅ Successfully rolled back to version {target_version}")
                            metrics = data.get('metrics', {})
                            print(f"   Restored RMSE: {metrics.get('rmse', 'N/A'):.4f}")
                        else:
                            print(f"❌ Rollback failed: {data.get('message', 'Unknown error')}")
                    else:
                        print("❌ Rollback request failed")
                else:
                    print("ℹ️  Only one version available, cannot demonstrate rollback")
        else:
            print("❌ Could not retrieve version history for rollback demo")
    
    def demo_api_endpoints(self):
        """Demo: Show available API endpoints"""
        self.print_section("9. AVAILABLE API ENDPOINTS")
        
        endpoints = [
            ("POST", "/api/adaptive/register", "Register symbol for adaptive learning"),
            ("POST", "/api/adaptive/train", "Perform initial model training"),
            ("POST", "/api/adaptive/update", "Manually trigger model update"),
            ("POST", "/api/adaptive/predict", "Make predictions with trained models"),
            ("GET", "/api/adaptive/status", "Get system status"),
            ("GET", "/api/adaptive/performance/<symbol>/<model_type>", "Get model performance"),
            ("POST", "/api/adaptive/rollback", "Rollback model to specific version"),
            ("POST", "/api/adaptive/start", "Start continuous learning"),
            ("POST", "/api/adaptive/stop", "Stop continuous learning"),
            ("GET", "/api/adaptive/stats", "Get database statistics"),
            ("GET", "/api/adaptive/versions", "Get model versions"),
            ("GET", "/api/adaptive/training-events", "Get training events"),
            ("POST", "/api/adaptive/cleanup", "Cleanup old data")
        ]
        
        print("📋 Adaptive Learning API Endpoints:")
        for method, endpoint, description in endpoints:
            print(f"   {method:4} {endpoint:45} - {description}")
    
    def run_complete_demo(self):
        """Run the complete adaptive learning demo"""
        print("🧠 ADAPTIVE LEARNING & CONTINUOUS EVALUATION DEMO")
        print("=" * 60)
        print("This demo showcases the complete adaptive learning system")
        print("for financial forecasting with automatic model updates,")
        print("version management, and performance tracking.")
        
        # Check backend health
        if not self.check_backend_health():
            print("\n❌ Demo cannot continue without backend connection.")
            print("Please start the backend server with: python backend/app.py")
            return
        
        try:
            # Run all demo sections
            self.demo_registration()
            self.demo_initial_training()
            self.demo_predictions()
            self.demo_model_updates()
            self.demo_performance_monitoring()
            self.demo_system_status()
            self.demo_continuous_learning_control()
            self.demo_rollback()
            self.demo_api_endpoints()
            
            # Final summary
            self.print_section("DEMO COMPLETE")
            print("🎉 Adaptive Learning Demo completed successfully!")
            print("\nKey features demonstrated:")
            print("✅ Online learning with SGD, LSTM, and Ensemble models")
            print("✅ Automatic model versioning and performance tracking")
            print("✅ Incremental updates with new data")
            print("✅ Model rollback capabilities")
            print("✅ Continuous learning scheduler")
            print("✅ Performance monitoring and analytics")
            print("✅ RESTful API for all operations")
            print("✅ MongoDB integration for persistence")
            
            print("\nNext steps:")
            print("1. Integrate with real-time data feeds")
            print("2. Set up automated scheduling")
            print("3. Configure performance alerts")
            print("4. Deploy to production environment")
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Demo interrupted by user")
        except Exception as e:
            print(f"\n❌ Demo error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to run the demo"""
    demo = AdaptiveLearningDemo()
    demo.run_complete_demo()


if __name__ == "__main__":
    main()