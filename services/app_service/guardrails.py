import re
from typing import Dict, Any, List, Optional, Tuple

# Configuration Limits
MAX_QUERY_LENGTH = 1000
MIN_QUERY_LENGTH = 3

# University Domain Concept Keywords (used for Scope Control)
UNIVERSITY_DOMAIN_KEYWORDS = {
    # Academic & Policies
    "attendance", "condonation", "debar", "debarred", "debarment", "medical", "leave",
    "exam", "exams", "examination", "semester", "end-semester", "mid-term", "cia", "ese",
    "admit", "card", "hall", "backlog", "supplementary", "re-evaluation", "answer", "script",
    "sgpa", "cgpa", "grade", "grading", "marks", "credit", "credits", "scale", "probation",
    "detention", "discipline", "umc", "unfair", "malpractice", "transfer", "branch",
    # Fees & Finance
    "fee", "fees", "tuition", "hostel", "mess", "fine", "late", "penalty", "scholarship",
    "waiver", "refund", "withdrawal", "admission", "installment",
    # Campus & Facilities
    "library", "borrowing", "books", "curfew", "gate", "biometric", "hostel", "warden",
    "room", "allotment", "grievance", "redressal", "anti-ragging", "complaint", "dean", "registrar",
    # UniAssist System & Code Architecture (In-Scope for System Architecture Queries)
    "uniassist", "rag", "retrieval", "vector", "embedding", "chunk", "chunking", "ollama",
    "docker", "architecture", "microservice", "gateway", "orchestrator", "bml", "munjal"
}

# Explicit Out-of-Domain Indicators
OUT_OF_DOMAIN_PATTERNS = [
    r"\b(bake|recipe|cook|ingredients|cake|pizza|pasta|soup|curry)\b",
    r"\b(stock|stocks|nasdaq|crypto|bitcoin|ethereum|forex|trading|share price)\b",
    r"\b(movie|celebrity|hollywood|bollywood|actor|actress|album|song|lyrics)\b",
    r"\b(cricket|football|nba|fifa|world cup|olympics|scorecard|match|tennis)\b",
    r"\b(horoscope|astrology|zodiac|fortune|tarot)\b",
    r"\b(hack|exploit|malware|keylogger|ddos|sql injection|payload)\b",
    r"\b(write a poem|write a song|write a story about|tell me a joke)\b"
]

# Security: Prompt Injection & Adversarial Jailbreak Patterns
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
    r"(reveal|show|print|display|dump)\s+(your\s+)?(system\s+prompt|initial\s+prompt|instructions)",
    r"you\s+are\s+now\s+(in\s+)?(dan|developer\s+mode|unrestricted|jailbreak)",
    r"bypass\s+(all\s+)?(safety|security|content|guardrail)\s+(filters|rules|policies)",
    r"act\s+as\s+(an?\s+)?unrestricted\s+ai",
    r"disregard\s+(the\s+)?(system|safety)\s+message",
    r"forget\s+(all\s+)?prior\s+commands"
]

# Security: Secret & Credential Leakage Probing Patterns
SECRET_PROBING_PATTERNS = [
    r"\b(\.env|aws_access_key_id|aws_secret_access_key|secret_key|private_key)\b",
    r"\b(id_rsa|uniassist-key\.pem|passwd|shadow|id_ed25519)\b",
    r"\b(database_url|db_password|api_key|bearer\s+token)\b"
]

# Inappropriate Content / Toxic Patterns
INAPPROPRIATE_PATTERNS = [
    r"\b(kill|murder|bomb|weapon|terrorist|suicide)\b",
    r"\b(abuse|slur|vulgarity)\b"
]


class GuardrailResult:
    def __init__(self, passed: bool, reason: str = "", guardrail_name: str = "", sanitized_query: str = ""):
        self.passed = passed
        self.reason = reason
        self.guardrail_name = guardrail_name
        self.sanitized_query = sanitized_query

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "guardrail_name": self.guardrail_name,
            "reason": self.reason
        }


class GuardrailsEngine:
    def __init__(self, retrieval_threshold: float = 0.25):
        # Empirically justified threshold: valid university queries score 0.38 - 0.58;
        # out-of-scope/insufficient score < 0.20
        self.retrieval_threshold = retrieval_threshold

    def validate_input(self, query: str) -> GuardrailResult:
        """
        Input Validation Guardrail:
        Restricts excessively long, empty, or malformed inputs.
        """
        if not query or not query.strip():
            return GuardrailResult(
                passed=False,
                guardrail_name="input_validation",
                reason="Query is empty or contains only whitespace. Please provide a valid question."
            )

        cleaned = query.strip()
        if len(cleaned) < MIN_QUERY_LENGTH:
            return GuardrailResult(
                passed=False,
                guardrail_name="input_validation",
                reason=f"Query is too short ({len(cleaned)} chars). Minimum length is {MIN_QUERY_LENGTH} characters."
            )

        if len(cleaned) > MAX_QUERY_LENGTH:
            return GuardrailResult(
                passed=False,
                guardrail_name="input_validation",
                reason=f"Input exceeds maximum allowed length of {MAX_QUERY_LENGTH} characters (received {len(cleaned)} chars). Please shorten your question."
            )

        # Inappropriate content check
        for pat in INAPPROPRIATE_PATTERNS:
            if re.search(pat, cleaned, re.IGNORECASE):
                return GuardrailResult(
                    passed=False,
                    guardrail_name="input_validation",
                    reason="Input contains prohibited or inappropriate content. Please adhere to university code of conduct."
                )

        return GuardrailResult(passed=True, guardrail_name="input_validation", sanitized_query=cleaned)

    def validate_security(self, query: str) -> GuardrailResult:
        """
        Security & System Override Guardrail:
        Blocks adversarial prompt injections, system instruction overrides, and credential probing.
        """
        cleaned = query.strip()

        # Check prompt injection patterns
        for pat in PROMPT_INJECTION_PATTERNS:
            if re.search(pat, cleaned, re.IGNORECASE):
                return GuardrailResult(
                    passed=False,
                    guardrail_name="security",
                    reason="Security Violation: Adversarial prompt injection or system override detected. Requests to alter system instructions are strictly rejected."
                )

        # Check secret probing
        for pat in SECRET_PROBING_PATTERNS:
            if re.search(pat, cleaned, re.IGNORECASE):
                return GuardrailResult(
                    passed=False,
                    guardrail_name="security",
                    reason="Security Violation: Probing for internal environment variables, credentials, or system secrets is prohibited."
                )

        return GuardrailResult(passed=True, guardrail_name="security", sanitized_query=cleaned)

    def validate_scope(self, query: str) -> GuardrailResult:
        """
        Scope Control Guardrail:
        Ensures queries are relevant to university domain (academics, grading, attendance, exams, fees, campus, architecture).
        Politely rejects out-of-domain requests.
        """
        cleaned = query.strip()
        q_lower = cleaned.lower()

        # 1. Check explicit out-of-domain patterns
        for pat in OUT_OF_DOMAIN_PATTERNS:
            if re.search(pat, q_lower):
                return GuardrailResult(
                    passed=False,
                    guardrail_name="scope",
                    reason="UniAssist is an academic knowledge assistant for BML Munjal University regulations, policies, and campus procedures. I cannot assist with topics outside this academic domain."
                )

        # 2. Tokenize query words
        words = set(re.findall(r"\b[a-zA-Z0-9_%₹\-]{3,}\b", q_lower))
        stop_words = {
            "what", "where", "when", "which", "who", "whom", "whose", "why", "how",
            "the", "and", "for", "with", "about", "can", "could", "would", "should",
            "tell", "explain", "give", "help", "please", "know", "information", "details"
        }
        content_words = words - stop_words

        # If query has domain keywords or mentions university context, pass
        has_domain_term = bool(content_words.intersection(UNIVERSITY_DOMAIN_KEYWORDS))
        is_general_university_inquiry = any(term in q_lower for term in ["university", "college", "campus", "student", "bml", "munjal", "professor", "course", "subject", "policy", "rule"])

        if not has_domain_term and not is_general_university_inquiry and len(content_words) >= 3:
            # Check if all content words are foreign to university vocabulary
            return GuardrailResult(
                passed=False,
                guardrail_name="scope",
                reason="UniAssist is specialized in university academic regulations, attendance, examinations, grading, and campus policies. Your query does not appear to relate to university matters."
            )

        return GuardrailResult(passed=True, guardrail_name="scope", sanitized_query=cleaned)

    def validate_retrieval_context(self, query: str, context_chunks: List[Dict[str, Any]]) -> GuardrailResult:
        """
        Insufficient Information Guardrail:
        Empirically grounded check: Evaluates whether retrieved context provides adequate evidence.
        Prevents LLM hallucination and guessing when required policy facts are missing.
        """
        if not context_chunks:
            return GuardrailResult(
                passed=False,
                guardrail_name="insufficient_context",
                reason="I cannot find sufficient official university policy information in our knowledge base to answer this question accurately. Please refer to the relevant academic office or Student Handbook."
            )

        top_score = max((c.get("similarity_score", 0.0) for c in context_chunks), default=0.0)

        # Extract core informative nouns from query
        q_tokens = set(re.findall(r"\b[a-zA-Z0-9_%₹\-]{4,}\b", query.lower()))
        common_stop = {
            "what", "which", "where", "when", "does", "have", "with", "from", "that", "this",
            "about", "would", "should", "could", "there", "their", "university", "policy",
            "student", "students", "campus", "located", "location", "rules", "guidelines",
            "regulations", "official", "information", "details"
        }
        core_query_tokens = q_tokens - common_stop

        # Combine text from all retrieved chunks
        all_context_text = " ".join((c.get("text", "") for c in context_chunks)).lower()
        overlap_tokens = {t for t in core_query_tokens if t in all_context_text}

        # Insufficiency trigger condition:
        # 1. Similarity score is below empirical threshold (0.25), OR
        # 2. Query has specific topic nouns (e.g. "astronaut", "swimming", "pool") and <40% exist in retrieved context
        is_low_similarity = (top_score < self.retrieval_threshold)
        topic_coverage = (len(overlap_tokens) / max(1, len(core_query_tokens))) if core_query_tokens else 1.0
        has_insufficient_topical_overlap = (len(core_query_tokens) >= 1 and topic_coverage < 0.40)

        if is_low_similarity or has_insufficient_topical_overlap:
            return GuardrailResult(
                passed=False,
                guardrail_name="insufficient_context",
                reason="I cannot find sufficient official university policy information in our knowledge base to answer this question accurately. Please refer to the Dean of Academic Affairs or the Student Handbook."
            )

        return GuardrailResult(passed=True, guardrail_name="insufficient_context")

    def validate_output(self, answer: str, context_chunks: List[Dict[str, Any]], use_rag: bool) -> Tuple[bool, str]:
        """
        Output Control Guardrail:
        Ensures response adheres to structure, avoids prompt leakage, and masks any accidental secrets.
        """
        if not answer:
            return False, "Generation failed to produce an output."

        # Mask synthetic credentials or accidental environment leaks
        sanitized = re.sub(r"(AKIA[0-9A-Z]{16})", "[REDACTED_AWS_KEY]", answer)
        sanitized = re.sub(r"(?i)(password\s*[:=]\s*)[^\s,]+", r"\1[REDACTED]", sanitized)

        # Check for raw system prompt leakage in output
        if "Context Information from University Knowledge Base:" in sanitized or "SYSTEM_PROMPT" in sanitized:
            sanitized = re.sub(r"Context Information from University Knowledge Base:.*?(?=Student Question:|$)", "", sanitized, flags=re.DOTALL)
            sanitized = sanitized.replace("Student Question:", "").strip()

        return True, sanitized


# Singleton instance
guardrails = GuardrailsEngine(retrieval_threshold=0.25)
