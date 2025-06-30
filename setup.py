#!/usr/bin/env python3
"""
Setup script for IMDA - Intelligent Maintenance Decision Agent
Prepares the environment and creates necessary files for deployment.
"""
import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False
    return True

def create_model():
    """Create the PyTorch model if it doesn't exist"""
    print("🤖 Creating model...")
    if not os.path.exists("assets/fault_model.pt"):
        try:
            subprocess.check_call([sys.executable, "create_model.py"])
            print("✓ Model created successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error creating model: {e}")
            return False
    else:
        print("✓ Model already exists")
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    directories = ["feedback", "chroma_db", "assets"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✓ Directories created")

def setup_environment():
    """Set up environment variables"""
    print("🔧 Setting up environment...")
    
    # Create .env file template if it doesn't exist
    if not os.path.exists(".env"):
        env_content = """# Environment variables for IMDA
# Get your token from https://huggingface.co/settings/tokens
HUGGINGFACE_TOKEN=your_token_here

# Optional: Customize model
HUGGINGFACE_MODEL=microsoft/DialoGPT-medium
"""
        with open(".env", "w") as f:
            f.write(env_content)
        print("✓ Created .env template - please add your Hugging Face token")
    else:
        print("✓ Environment file exists")

def main():
    """Main setup function"""
    print("🚀 Setting up IMDA - Intelligent Maintenance Decision Agent")
    print("=" * 60)
    
    success = True
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        success = False
    
    # Create model
    if not create_model():
        success = False
    
    # Setup environment
    setup_environment()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Add your Hugging Face token to .env file (optional)")
        print("2. Run: python app.py")
        print("3. Open http://localhost:7860 in your browser")
    else:
        print("❌ Setup completed with errors")
        print("Please check the error messages above and try again")

if __name__ == "__main__":
    main()
