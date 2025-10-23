#!/usr/bin/env python3
"""
Script to run both the FastAPI server and Streamlit UI concurrently.
"""

import subprocess
import sys
import time
import threading
import signal
import os
from pathlib import Path

def run_command(command, name):
    """Run a command and handle its output."""
    try:
        print(f"🚀 Starting {name}...")
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output in real-time
        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"[{name}] {line.strip()}")
        
        process.wait()
    except Exception as e:
        print(f"❌ Error running {name}: {e}")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n🛑 Shutting down services...")
    sys.exit(0)

def main():
    """Main function to run both services."""
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🐾 Pet Classifier - Starting Both Services")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Error: pyproject.toml not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Define the commands
    serve_cmd = "task serve"
    ui_cmd = "task ui"
    
    print(f"📋 Commands to run:")
    print(f"   - FastAPI Server: {serve_cmd}")
    print(f"   - Streamlit UI: {ui_cmd}")
    print()
    
    # Start both services in separate threads
    serve_thread = threading.Thread(target=run_command, args=(serve_cmd, "FastAPI Server"))
    ui_thread = threading.Thread(target=run_command, args=(ui_cmd, "Streamlit UI"))
    
    try:
        # Start the FastAPI server first
        serve_thread.start()
        
        # Wait a moment for the server to start
        print("⏳ Waiting for FastAPI server to start...")
        time.sleep(3)
        
        # Start the Streamlit UI
        ui_thread.start()
        
        print()
        print("✅ Both services are starting!")
        print("🌐 FastAPI server will be available at: http://localhost:8000")
        print("🎨 Streamlit UI will be available at: http://localhost:8501")
        print()
        print("Press Ctrl+C to stop both services")
        print("=" * 50)
        
        # Wait for both threads to complete
        serve_thread.join()
        ui_thread.join()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
