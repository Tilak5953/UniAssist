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

# Default models selected for live multi-model comparison
DEFAULT_COMPARE_MODELS = ["qwen2.5:0.5b", "tinyllama:latest", "qwen2.5:1.5b"]

# Priority presets for question-specific recommendation engine
EVALUATION_PRIORITIES = {
    "balanced": {
        "label": "Balanced Performance",
        "description": "Harmonizes accuracy, relevance, low hallucination, speed, and memory usage."
    },
    "fastest": {
        "label": "Fastest Response",
        "description": "Prioritizes lowest response latency and highest generation throughput."
    },
    "accuracy": {
        "label": "Most Accurate Answer",
        "description": "Emphasizes exact factual adherence to official university regulations."
    },
    "memory": {
        "label": "Lowest Memory Usage",
        "description": "Prefers lightweight models that minimize RAM consumption (ideal for 1GB EC2)."
    },
    "relevance": {
        "label": "Most Relevant & Grounded",
        "description": "Prioritizes context grounding, retrieval overlap, and hallucination resistance."
    }
}

SYSTEM_PROMPT_RAG = """You are UniAssist, an authoritative and helpful AI University Assistant for BML Munjal University.
Your objective is to answer the student's question accurately using ONLY the provided university knowledge base context.

Guidelines:
1. Base your answer directly on the provided context. If the context does not contain the answer, politely state: "This information is not covered in the university guidelines available in my knowledge base. Please consult the Student Affairs or Academic Registrar's office."
2. Be concise, structured, and clear. Use bullet points or step-by-step numbers where appropriate.
3. Cite the relevant policy, section, or fee amounts mentioned in the context.
4. Do not invent rules, fees, dates, or regulations.
5. Explicitly distinguish between different fee categories (e.g., regular backlog examination registration fee vs. summer supplementary exam fee vs. answer script re-evaluation fee). Never combine or present re-evaluation fees (₹800) as backlog registration fees (₹750). Always cite each fee with its exact policy document and section.
"""

SYSTEM_PROMPT_DIRECT = """You are an AI assistant answering a university student.
Answer the following question based on general knowledge to the best of your ability.
"""
