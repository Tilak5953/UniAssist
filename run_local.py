import subprocess
import sys
import time
import signal
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent
    print("==========================================================")
    print("Starting UniAssist Microservices Locally")
    print("==========================================================")

    # 1. Check Python Dependencies
    print("Checking core dependencies...")
    try:
        import fastapi
        import uvicorn
        import pydantic
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please run: pip install -r services/app_service/requirements.txt")
        sys.exit(1)

    print("Launching RAG Service on port 8001...")
    rag_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"],
        cwd=str(root / "services" / "rag_service")
    )

    print("Waiting 2 seconds for RAG service to initialize...")
    time.sleep(2)

    print("Launching App Gateway on port 8000...")
    app_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=str(root / "services" / "app_service")
    )

    print("\n==========================================================")
    print(" UniAssist Services are Running:")
    print(" -> Student Web Portal : http://localhost:8000")
    print(" -> RAG Microservice   : http://localhost:8001")
    print(" -> RAG API Docs       : http://localhost:8001/docs")
    print(" -> App API Docs       : http://localhost:8000/docs")
    print("==========================================================")
    print("Press Ctrl+C to stop all services.\n")

    def handle_sigint(sig, frame):
        print("\nShutting down UniAssist services...")
        app_proc.terminate()
        rag_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        while True:
            time.sleep(1)
            if rag_proc.poll() is not None:
                print("RAG service stopped unexpectedly.")
                break
            if app_proc.poll() is not None:
                print("App Gateway service stopped unexpectedly.")
                break
    except KeyboardInterrupt:
        handle_sigint(None, None)

if __name__ == "__main__":
    main()
