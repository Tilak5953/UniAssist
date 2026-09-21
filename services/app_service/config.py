import os

RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://127.0.0.1:8001")
OLLAMA_SERVICE_URL = os.getenv("OLLAMA_SERVICE_URL", "http://127.0.0.1:11434")

# Supported models optimized for low RAM AWS servers
AVAILABLE_MODELS = [
    {
        "id": "qwen2.5:0.5b",
        "name": "Qwen 2.5 (0.5B)",
        "badge": "Default (390 MB)",
        "description": "Ultra-lightweight 0.5B model. Operates seamlessly on low-RAM AWS instances (1GB RAM).",
        "recommended": True
    },
    {
        "id": "tinyllama:latest",
        "name": "TinyLlama (1.1B)",
        "badge": "Fast (630 MB)",
        "description": "Compact 1.1B parameter model with rapid token generation and low memory consumption.",
        "recommended": False
    },
    {
        "id": "qwen2.5:1.5b",
        "name": "Qwen 2.5 (1.5B)",
        "badge": "Balanced (980 MB)",
        "description": "Slightly larger 1.5B model providing higher reasoning fidelity under 1.5GB RAM.",
        "recommended": False
    },
    {
        "id": "codellama:7b",
        "name": "Code Llama (7B)",
        "badge": "Optional (3.8 GB)",
        "description": "Full Code Llama model (for higher-spec servers with 8GB+ RAM).",
        "recommended": False
    }
]

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen2.5:0.5b")

SYSTEM_PROMPT_RAG = """You are UniAssist, an authoritative and helpful AI University Assistant for BML Munjal University.
Your objective is to answer the student's question accurately using ONLY the provided university knowledge base context.

Guidelines:
1. Base your answer directly on the provided context. If the context does not contain the answer, politely state: "This information is not covered in the university guidelines available in my knowledge base. Please consult the Student Affairs or Academic Registrar's office."
2. Be concise, structured, and clear. Use bullet points or step-by-step numbers where appropriate.
3. Cite the relevant policy, section, or fee amounts mentioned in the context.
4. Do not invent rules, fees, dates, or regulations.
"""

SYSTEM_PROMPT_DIRECT = """You are an AI assistant answering a university student.
Answer the following question based on general knowledge to the best of your ability.
"""
