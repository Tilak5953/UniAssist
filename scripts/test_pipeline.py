import sys
import os
from pathlib import Path

# Add services/rag_service to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "services" / "rag_service"))

from chunking import DocumentChunker
from embeddings import EmbeddingEngine
from vector_store import VectorStore

def run_tests():
    print("==================================================")
    print("UniAssist Automated Pipeline Verification Test")
    print("==================================================")

    kb_dir = project_root / "knowledge_base"
    print(f"\n1. Checking Knowledge Base Directory: {kb_dir}")
    assert kb_dir.exists(), f"Knowledge base directory missing: {kb_dir}"
    
    docs = list(kb_dir.glob("*.md"))
    print(f"   Found {len(docs)} knowledge base files:")
    for d in docs:
        print(f"   - {d.name} ({d.stat().st_size} bytes)")
    assert len(docs) >= 5, f"Expected at least 5 knowledge base documents, found {len(docs)}"

    print("\n2. Testing Document Chunking...")
    chunker = DocumentChunker(chunk_size=600, chunk_overlap=120)
    chunks = chunker.process_directory(kb_dir)
    print(f"   Successfully generated {len(chunks)} contextual chunks across {len(docs)} documents.")
    assert len(chunks) >= 20, f"Expected at least 20 chunks, got {len(chunks)}"

    print("\n3. Testing Dense Semantic Vector Embeddings...")
    embedder = EmbeddingEngine(dim=256)
    sample_text = "What is the passing criteria for end semester examination?"
    sample_emb = embedder.get_embedding(sample_text)
    print(f"   Query embedding dimension: {len(sample_emb)}")
    assert len(sample_emb) == 256, f"Embedding dimension mismatch: {len(sample_emb)}"

    print("\n4. Testing Vector Store & Cosine Similarity Ranking...")
    vector_store = VectorStore()
    chunk_vectors = [embedder.get_embedding(c["text"]) for c in chunks]
    vector_store.add_items(chunks, chunk_vectors)
    print(f"   Indexed {len(vector_store.chunks)} chunks in vector store.")

    test_queries = [
        {
            "q": "What happens if my attendance is 68 percent and can I get condonation?",
            "expected_file": "02_attendance_policy_and_condonation.md"
        },
        {
            "q": "How much is the backlog examination fee per subject?",
            "expected_file": "01_semester_examination_policy.md"
        },
        {
            "q": "What is the minimum CGPA required to avoid academic probation?",
            "expected_file": "03_academic_regulations_and_grading.md"
        },
        {
            "q": "How much scholarship is given for top 2% of batch?",
            "expected_file": "04_fee_structure_and_scholarships.md"
        },
        {
            "q": "What is the hostel curfew and campus gate closing time?",
            "expected_file": "05_campus_facilities_and_code_of_conduct.md"
        }
    ]

    print("\n5. Running Semantic Similarity Retrieval Queries:")
    all_passed = True
    for t in test_queries:
        query = t["q"]
        expected = t["expected_file"]
        q_vec = embedder.get_embedding(query)
        results = vector_store.search(query, q_vec, top_k=2)

        print(f"\n   Query: '{query}'")
        if results:
            top = results[0]
            print(f"   Top Match: [{top['source_file']}] -> {top['section_title']} (Score: {top['similarity_score']})")
            passed = (top['source_file'] == expected)
            status = "PASSED" if passed else "FAILED"
            print(f"   Verification: {status} (Expected: {expected})")
            if not passed:
                all_passed = False
        else:
            print("   ERROR: No results retrieved!")
            all_passed = False

    print("\n==================================================")
    if all_passed:
        print("ALL PIPELINE VERIFICATION TESTS PASSED SUCCESSFULLY!")
    else:
        print("SOME TESTS DID NOT MATCH EXACT TARGET FILE")
    print("==================================================")
    return all_passed

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
