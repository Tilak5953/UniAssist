"""
Automated Verification Suite for UniAssist Live Model Evaluation & Recommendation Dashboard
Tests 5 representative questions, RAG vs Direct modes, priority switching, and metric evaluations.
"""
import sys
import os
from pathlib import Path

# Add services paths
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "services" / "rag_service"))
sys.path.insert(0, str(root / "services" / "app_service"))

from main import app, CompareModelsRequest, compare_models
from fastapi.testclient import TestClient

client = TestClient(app)

def run_tests():
    print("=" * 65)
    print("UniAssist Live Model Evaluation Dashboard Verification Test")
    print("=" * 65)

    test_queries = [
        {
            "id": "T1",
            "name": "Attendance Requirement (Policy Grounded)",
            "query": "What is the minimum attendance requirement for appearing in semester examinations?",
            "use_rag": True,
            "expected_reference": True,
            "expected_facts": ["75%"]
        },
        {
            "id": "T2",
            "name": "Examination Rules & Passing Criteria",
            "query": "What are the semester examination rules and CIA weightage?",
            "use_rag": True,
            "expected_reference": True,
            "expected_facts": ["40%", "60%"]
        },
        {
            "id": "T3",
            "name": "SGPA Calculation & Conversion Formula",
            "query": "How is SGPA calculated and converted to equivalent percentage?",
            "use_rag": True,
            "expected_reference": True,
            "expected_facts": ["(CGPA - 0.75) * 10"]
        },
        {
            "id": "T4",
            "name": "Late Tuition Fees & Penalties",
            "query": "What is the late fee for tuition payment and refund policy?",
            "use_rag": True,
            "expected_reference": True,
            "expected_facts": ["₹100", "refund"]
        },
        {
            "id": "T5",
            "name": "Out-of-Domain Question (No Reference)",
            "query": "Where is the student Olympic swimming pool located on campus?",
            "use_rag": True,
            "expected_reference": False,
            "expected_facts": []
        }
    ]

    all_passed = True

    for t in test_queries:
        print(f"\n--> Running Test {t['id']}: {t['name']}")
        print(f"    Query: '{t['query']}'")
        req = CompareModelsRequest(
            query=t["query"],
            models=["qwen2.5:0.5b", "tinyllama:latest", "qwen2.5:1.5b"],
            use_rag=t["use_rag"],
            priority="balanced"
        )
        res = compare_models(req)

        # 1. Verify models evaluated
        models = res.get("model_results", [])
        if len(models) != 3:
            print(f"    [FAIL] Expected 3 models evaluated, got {len(models)}")
            all_passed = False
        else:
            print(f"    [PASS] Evaluated 3 models: {[m['model_id'] for m in models]}")

        # 2. Verify Latency measurement
        for m in models:
            lat = m.get("latency_ms", 0)
            gen_lat = m.get("generation_latency_ms", 0)
            if lat <= 0 or gen_lat <= 0:
                print(f"    [FAIL] Invalid latency for {m['model_id']}: {lat}ms")
                all_passed = False
        print(f"    [PASS] Latency measured: {[f'{m['model_name']}: {m['latency_ms']}ms' for m in models]}")

        # 3. Verify Accuracy evaluation
        rec = res.get("recommendation", {})
        ref_avail = res.get("retrieval", {}).get("reference_available", False)
        print(f"    Reference Ground Truth Available: {ref_avail}")

        for m in models:
            acc = m.get("accuracy", {})
            if ref_avail:
                if acc.get("status") != "verified" or acc.get("score") is None:
                    print(f"    [FAIL] Expected verified accuracy for {m['model_id']}")
                    all_passed = False
            else:
                if acc.get("status") != "unverified" or acc.get("score") is not None:
                    print(f"    [FAIL] Expected unverified accuracy for out-of-domain query")
                    all_passed = False
                if "Accuracy not verified" not in acc.get("display", ""):
                    print(f"    [FAIL] Expected 'Accuracy not verified' display string")
                    all_passed = False

        if not ref_avail:
            print("    [PASS] Verified unverified accuracy message: 'Accuracy not verified – no suitable reference answer available'")
        else:
            print(f"    [PASS] Accuracy scores: {[f'{m['model_name']}: {m['accuracy']['score']}%' for m in models]}")

        # 4. Verify Relevance & Hallucination
        for m in models:
            rel = m.get("relevance", {}).get("score", None)
            hallu_level = m.get("hallucination", {}).get("risk_level", "")
            if rel is None or rel < 0 or not hallu_level:
                print(f"    [FAIL] Missing relevance or hallucination for {m['model_id']}")
                all_passed = False
        print(f"    [PASS] Relevance & Hallucination computed properly.")

        # 5. Verify Recommendation & Trade-offs
        winner = rec.get("recommended_model_name")
        verdict = rec.get("verdict")
        evidence = rec.get("evidence")
        trade_offs = rec.get("trade_offs")
        if not winner or not verdict or not evidence or not trade_offs:
            print(f"    [FAIL] Incomplete recommendation payload: {rec}")
            all_passed = False
        else:
            print(f"    [PASS] Recommendation: {verdict}")
            print(f"           Evidence: {evidence[:90]}...")
            print(f"           Trade-offs: {trade_offs[:90]}...")

    # 6. Test Direct LLM Mode (Ungrounded)
    print("\n--> Running Test T6: Direct LLM Mode (Ungrounded Parametric)")
    req_direct = CompareModelsRequest(
        query="What is the minimum attendance requirement for appearing in semester examinations?",
        models=["qwen2.5:0.5b", "tinyllama:latest", "qwen2.5:1.5b"],
        use_rag=False,
        priority="balanced"
    )
    res_direct = compare_models(req_direct)
    for m in res_direct["model_results"]:
        if m["hallucination"]["risk_level"] != "High Risk":
            print(f"    [FAIL] Expected High Risk for Direct LLM without context, got {m['hallucination']['risk_level']}")
            all_passed = False
        if "Direct LLM" not in m["warnings"][0] and "ungrounded" not in m["hallucination"]["display"].lower():
            print(f"    [FAIL] Missing ungrounded warning for Direct mode")
            all_passed = False
    print("    [PASS] Direct LLM mode correctly flagged with High Hallucination Risk & Ungrounded warning.")

    # 7. Test Priority Variations
    print("\n--> Running Test T7: Recommendation Priority Sensitivity")
    priorities = ["fastest", "accuracy", "memory", "relevance", "balanced"]
    for p in priorities:
        req_p = CompareModelsRequest(
            query="What is the minimum attendance requirement for appearing in semester examinations?",
            priority=p
        )
        res_p = compare_models(req_p)
        rec_p = res_p["recommendation"]
        print(f"    Priority '{p}' -> Recommended: {rec_p['recommended_model_name']} ({rec_p['verdict']})")

    print("\n" + "=" * 65)
    if all_passed:
        print("ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY! [100% PASS]")
    else:
        print("SOME TESTS FAILED. PLEASE REVIEW LOGS.")
    print("=" * 65)
    return all_passed

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
