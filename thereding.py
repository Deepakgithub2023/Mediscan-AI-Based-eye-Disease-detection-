import subprocess
import threading

def run_backend():
    """Run the backend script (main.py)."""
    subprocess.call(["python", "main.py"])

def run_frontend():
    """Run the Streamlit frontend (app_ui.py)."""
    subprocess.call(["streamlit", "run", "app_ui.py"])

if __name__ == "__main__":
    # Start the backend in a separate thread
    backend_thread = threading.Thread(target=run_backend)
    backend_thread.daemon = True  # Daemonize thread
    backend_thread.start()

    # Start the frontend
    run_frontend()