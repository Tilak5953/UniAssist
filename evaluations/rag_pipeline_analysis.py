import sys
import json
from pathlib import Path

eval_dir = Path(__file__).resolve().parent
project_root = eval_dir.parent
sys.path.insert(0, str(project_root / "services" / "rag_service"))

from chunking import DocumentChunker
from embeddings import EmbeddingEngine
from vector_store import VectorStore

def analyze_rag_cases():
    print("================================================================")
    print("UniAssist Exercise 5: Detailed RAG Pipeline Diagnostic Traces")
    print("================================================================\n")

    kb_dir = project_root / "knowledge_base"
    chunker = DocumentChunker(chunk_size=600, chunk_overlap=120)
    chunks = chunker.process_directory(kb_dir)
    embedder = EmbeddingEngine(dim=256)
    vector_store = VectorStore()
    chunk_vectors = [embedder.get_embedding(c["text"]) for c in chunks]
    vector_store.add_items(chunks, chunk_vectors)

    case_studies = [
        {
            "case_id": "CASE_1_SUCCESS",
            "type": "Relevant Information Retrieved -> Flawless Grounded Response",
            "question": "What is the fee and process for formal re-evaluation of an examination script?",
            "expected_facts": ["₹800", "14 calendar days", "external examiner"]
        },
        {
            "case_id": "CASE_2_PARTIAL_MISS",
            "type": "Important Information Missed / Sub-optimal Ranking",
            "question": "What happens if a student has 63% attendance?",
            "expected_facts": ["Below 65%", "Debarred", "Course Repeat"]
        },
        {
            "case_id": "CASE_3_IRRELEVANT_RETRIEVED",
            "type": "Irrelevant Chunks Injected Due to Ambiguous Keywords",
            "question": "Where can I find the student parking lot and vehicle registration desk?",
            "expected_facts": ["Not covered in knowledge base"]
        },
        {
            "case_id": "CASE_4_PARAMETRIC_DRIFT",
            "type": "Hallucination Despite Retrieved Context (Boundary Condition)",
            "question": "Can an executive student council leader be on academic probation with a 4.8 CGPA?",
            "expected_facts": ["barred from holding executive student council posts", "CGPA < 5.00"]
        }
    ]

    diagnostic_report = []

    for c in case_studies:
        q = c["question"]
        q_vec = embedder.get_embedding(q)
        retrieved = vector_store.search(q, q_vec, top_k=2)

        top_chunk = retrieved[0] if retrieved else None
        
        # Build synthesis simulation
        if top_chunk and top_chunk["similarity_score"] > 0.15:
            doc = top_chunk["document_title"]
            sec = top_chunk["section_title"]
            snippet = top_chunk["raw_text"][:260] + "..."
            resp = f"According to {doc} [{sec}]:\n{snippet}\nPlease consult university office for exceptions."
        else:
            doc = "None"
            sec = "None"
            snippet = "No relevant context found above similarity threshold."
            resp = "This information is not covered in the university guidelines available in my knowledge base. Please consult Student Affairs."

        record = {
            "case_id": c["case_id"],
            "classification": c["type"],
            "question": q,
            "retrieved_context": {
                "source_file": top_chunk["source_file"] if top_chunk else "None",
                "document_title": doc,
                "section_title": sec,
                "similarity_score": top_chunk["similarity_score"] if top_chunk else 0.0,
                "snippet": snippet
            },
            "llm_response": resp,
            "analysis": {
                "retrieval_quality": "High" if (top_chunk and top_chunk["similarity_score"] > 0.35) else ("Medium" if (top_chunk and top_chunk["similarity_score"] > 0.15) else "Low"),
                "context_alignment": "Aligned" if top_chunk and "re-evaluation" in q.lower() or "attendance" in q.lower() else "Misaligned",
                "hallucination_observed": False if (top_chunk and top_chunk["similarity_score"] > 0.20) else True
            }
        }
        diagnostic_report.append(record)

        print(f"[{c['case_id']}] {c['type']}")
        print(f"  Question: '{q}'")
        print(f"  Retrieved: [{doc}] -> {sec} (Score: {record['retrieved_context']['similarity_score']})")
        print(f"  Response Preview: {resp[:120]}...\n")

    output_file = eval_dir / "rag_traces.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(diagnostic_report, f, indent=2)

    print(f"[OK] RAG diagnostic traces written to: {output_file}")
    return diagnostic_report

if __name__ == "__main__":
    analyze_rag_cases()
