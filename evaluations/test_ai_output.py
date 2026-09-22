import sys
import os
import re
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Ensure app_service is first on sys.path to avoid collision with rag_service/main.py
project_root = Path(__file__).resolve().parent.parent
app_service_dir = str(project_root / "services" / "app_service")
if app_service_dir not in sys.path:
    sys.path.insert(0, app_service_dir)

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

class AIOutputTestSuite:
    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    def log_test(self, test_id: str, name: str, expected: str, actual: str, passed: bool, failure_detail: str = ""):
        self.results.append({
            "test_id": test_id,
            "test_name": name,
            "expected_behavior": expected,
            "actual_output": actual,
            "status": "PASS" if passed else "FAIL",
            "failure_detail": failure_detail if not passed else None
        })
        status_str = "[PASS]" if passed else "[FAIL]"
        safe_name = name.encode("ascii", "replace").decode("ascii")
        print(f" {status_str} {test_id}: {safe_name}")
        if not passed:
            safe_detail = failure_detail.encode("ascii", "replace").decode("ascii")
            print(f"        Failure Detail: {safe_detail}")

    # Test 1: Relevance Check
    def test_relevance(self):
        query = "What is the mandatory attendance requirement for end-semester examinations?"
        resp = client.post("/api/ask", json={"query": query, "use_rag": True})
        data = resp.json()
        ans = data.get("answer", "").lower()
        
        has_core = "75" in ans and ("attendance" in ans or "mandatory" in ans or "debar" in ans)
        self.log_test(
            test_id="TEST-01",
            name="Answer Relevance to Question",
            expected="Answer directly addresses the 75% statutory attendance requirement.",
            actual=ans[:120] + "...",
            passed=has_core,
            failure_detail=f"Expected '75%' and 'attendance' in answer, but received: {ans[:100]}"
        )

    # Test 2: Supported by Retrieved Context
    def test_context_support(self):
        query = "How much is the regular backlog examination fee per subject?"
        resp = client.post("/api/ask", json={"query": query, "use_rag": True})
        data = resp.json()
        ans = data.get("answer", "")
        citations = data.get("citations", [])
        
        has_fee = "750" in ans
        has_citation = any("semester_examination_policy" in c.get("source_file", "").lower() for c in citations)
        passed = has_fee and has_citation
        self.log_test(
            test_id="TEST-02",
            name="Answer Grounding in Retrieved Context",
            expected="Answer asserts 750 regular backlog fee and cites official examination policy document.",
            actual=f"Answer: {ans[:80]}... | Citations: {len(citations)} chunks",
            passed=passed,
            failure_detail=f"Missing 750 fee or examination policy citation (fee_found={has_fee}, cit_found={has_citation})"
        )

    # Test 3: Absence of Unsupported Claims
    def test_no_unsupported_claims(self):
        query = "Can a student with 68% attendance get condonation, and what is the fine?"
        resp = client.post("/api/ask", json={"query": query, "use_rag": True})
        data = resp.json()
        ans = data.get("answer", "")
        
        has_correct_fine = "1200" in ans or "1,200" in ans
        no_fake_high_fine = "5000" not in ans and "10000" not in ans
        passed = has_correct_fine and no_fake_high_fine
        self.log_test(
            test_id="TEST-03",
            name="Absence of Unsupported or Hallucinated Claims",
            expected="Answer identifies the 1,200 condonation fine without hallucinating unsupported amounts.",
            actual=ans[:100] + "...",
            passed=passed,
            failure_detail="Failed to identify 1,200 fine or fabricated unsupported numbers."
        )

    # Test 4: Expected Response Format
    def test_format_adherence(self):
        query = "What are the rules regarding late entry and early exit from the examination hall?"
        resp = client.post("/api/ask", json={"query": query, "use_rag": True})
        data = resp.json()
        ans = data.get("answer", "")
        
        no_prompt_leak = "Context Information from University Knowledge Base:" not in ans and "SYSTEM_PROMPT" not in ans
        is_structured = len(ans) > 30 and ("15" in ans or "30" in ans or "minute" in ans.lower())
        passed = no_prompt_leak and is_structured
        self.log_test(
            test_id="TEST-04",
            name="Expected Response Format & Structure Adherence",
            expected="Structured response without raw prompt leakage, mentioning examination hall timing rules.",
            actual=ans[:110] + "...",
            passed=passed,
            failure_detail="Answer leaked internal system prompts or lacked expected response structure."
        )

    # Test 5: Answers When Sufficient Information is Available
    def test_answers_when_sufficient(self):
        query = "What is the fee and process for formal re-evaluation of an examination answer script?"
        resp = client.post("/api/ask", json={"query": query, "use_rag": True})
        data = resp.json()
        ans = data.get("answer", "")
        guard_status = data.get("guardrail_status", {})
        
        not_refused = "insufficient" not in ans.lower() and ("800" in ans or "re-evaluation" in ans.lower())
        passed = not_refused and guard_status.get("insufficient_context") == "PASS"
        self.log_test(
            test_id="TEST-05",
            name="Answers When Sufficient Information is Available",
            expected="Provides complete answer regarding 800 re-evaluation fee when policy context is present.",
            actual=ans[:100] + "...",
            passed=passed,
            failure_detail="Erroneously refused or triggered insufficient-context guardrail on legitimate indexed policy."
        )

    # Test 6: Appropriately Refuses When Information is Unavailable
    def test_refuses_when_insufficient(self):
        query = "Where is the student Olympic swimming pool located on campus?"
        resp = client.post("/api/ask", json={"query": query, "use_rag": True})
        data = resp.json()
        ans = data.get("answer", "")
        guard_status = data.get("guardrail_status", {})
        
        is_refusal = "[Insufficient Information]" in ans or "sufficient" in ans.lower()
        blocked_by_guardrail = guard_status.get("insufficient_context") == "BLOCKED"
        passed = is_refusal and blocked_by_guardrail
        self.log_test(
            test_id="TEST-06",
            name="Appropriately Refuses When Information is Unavailable",
            expected="Politely states sufficient official policy information is unavailable in the knowledge base.",
            actual=ans[:120] + "...",
            passed=passed,
            failure_detail=f"Did not refuse unindexed entity: answer was {ans[:100]}"
        )

    # Test 7: Rejects Out-of-Scope Questions
    def test_rejects_out_of_scope(self):
        query = "How to bake a chocolate cake at home?"
        resp = client.post("/api/ask", json={"query": query, "use_rag": True})
        data = resp.json()
        ans = data.get("answer", "")
        guard_status = data.get("guardrail_status", {})
        
        is_scope_alert = "[Scope Alert]" in ans or "academic domain" in ans.lower()
        blocked_by_scope = guard_status.get("scope") == "BLOCKED"
        passed = is_scope_alert and blocked_by_scope
        self.log_test(
            test_id="TEST-07",
            name="Rejects Out-of-Scope Non-University Questions",
            expected="Rejects non-academic cooking query with standard university assistant referral.",
            actual=ans[:110] + "...",
            passed=passed,
            failure_detail=f"Failed to reject out-of-scope query: answer was {ans[:100]}"
        )

    # Test 8: Handles Excessively Long or Invalid Inputs Correctly
    def test_input_validation(self):
        long_query = "Please tell me the university policy " + ("on attendance and grading " * 80)
        resp = client.post("/api/ask", json={"query": long_query})
        data = resp.json()
        ans = data.get("answer", "")
        guard_status = data.get("guardrail_status", {})
        
        is_input_error = "[Input Error]" in ans or "exceeds maximum allowed length" in ans.lower()
        blocked_by_input = guard_status.get("input_validation") == "BLOCKED"
        passed = is_input_error and blocked_by_input
        self.log_test(
            test_id="TEST-08",
            name="Handles Excessively Long / Invalid Inputs Correctly",
            expected="Blocks input exceeding 1,000 characters with an informative [Input Error] notice.",
            actual=ans[:110] + "...",
            passed=passed,
            failure_detail=f"Allowed input exceeding length limit: answer was {ans[:100]}"
        )

    def run_all(self) -> Dict[str, Any]:
        print("================================================================")
        print("UniAssist AI Output Systematic Testing Suite (Task 3)")
        print("Deterministic PASS/FAIL Output Checks")
        print("================================================================\n")
        
        self.test_relevance()
        self.test_context_support()
        self.test_no_unsupported_claims()
        self.test_format_adherence()
        self.test_answers_when_sufficient()
        self.test_refuses_when_insufficient()
        self.test_rejects_out_of_scope()
        self.test_input_validation()
        
        total = len(self.results)
        passed_count = sum(1 for r in self.results if r["status"] == "PASS")
        pass_rate = round((passed_count / total) * 100, 1)
        
        print("\n----------------------------------------------------------------")
        print(f"TOTAL TESTS: {total} | PASSED: {passed_count} | FAILED: {total - passed_count} | PASS RATE: {pass_rate}%")
        print("----------------------------------------------------------------\n")
        
        report_file = project_root / "evaluations" / "ai_output_test_report.json"
        summary = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_tests": total,
            "passed_tests": passed_count,
            "failed_tests": total - passed_count,
            "pass_rate_pct": pass_rate,
            "test_cases": self.results
        }
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Detailed output test report saved to: {report_file}")
        return summary

if __name__ == "__main__":
    suite = AIOutputTestSuite()
    summary = suite.run_all()
    if summary["failed_tests"] > 0:
        sys.exit(1)
    sys.exit(0)
