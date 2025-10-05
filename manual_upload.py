#!/usr/bin/env python3
"""
Manual upload of existing trained models
"""

from huggingface_hub import HfApi, create_repo, upload_file
import os

def upload_existing_models():
    api = HfApi()
    username = "abdullah-daoud"  # Replace with your username
    
    # Check if you have existing trained models
    model_files = [
        "backend/ml_models/trained_model.pkl",  # From predictor.py
        # Add other model files if they exist
    ]
    
    # Create repository
    repo_name = f"{username}/fintech-trained-models"
    try:
        create_repo(repo_name, exist_ok=True)
        print(f"✅ Created repository: {repo_name}")
    except Exception as e:
        print(f"Repository might already exist: {e}")
    
    # Upload existing model files
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                upload_file(
                    path_or_fileobj=model_file,
                    path_in_repo=os.path.basename(model_file),
                    repo_id=repo_name,
                    repo_type="model"
                )
                print(f"✅ Uploaded {model_file}")
            except Exception as e:
                print(f"❌ Error uploading {model_file}: {e}")
        else:
            print(f"⚠️ Model file not found: {model_file}")

if __name__ == "__main__":
    upload_existing_models()