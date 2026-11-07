import json
import unittest

from scripts.codespaces.report_failure import issue_payload, publish


class FailureReportingTests(unittest.TestCase):
    def env(self):
        return {
            "GITHUB_REPOSITORY": "VictorWong123/tokenomics",
            "GITHUB_SHA": "a" * 40,
            "GITHUB_RUN_ID": "123",
            "GITHUB_RUN_ATTEMPT": "1",
            "INITIALIZE_RESULT": "success",
            "WORKLOAD_RESULT": "failure",
            "FINALIZE_RESULT": "success",
            "SHUTDOWN_CONFIRMED": "true",
            "MODEL_REPORT": json.dumps(
                {
                    "stage": "correctness",
                    "first_mismatch": {
                        "prompt_id": "development-qa-00",
                        "configuration_id": "retain-fixed4",
                        "mismatch": {
                            "field": "token_ids",
                            "position": 29,
                            "reference": 16417,
                            "actual": 2415,
                            "raw_prompt": "DO-NOT-PUBLISH",
                        },
                    },
                }
            ),
        }

    def test_real_failure_case_is_bounded_and_has_source_run_and_shutdown(self):
        payload = issue_payload(self.env())
        for value in (
            "16417",
            "2415",
            "29",
            "retain-fixed4",
            "development-qa-00",
            "a" * 40,
            "/runs/123/attempts/1",
        ):
            self.assertIn(value, payload["body"])
        self.assertNotIn("DO-NOT-PUBLISH", payload["body"])
        env = self.env()
        env["WORKLOAD_RESULT"] = "success"
        self.assertIsNone(issue_payload(env))

    def test_infrastructure_failure_without_report_and_wrong_repository(self):
        env = self.env()
        env.update(
            INITIALIZE_RESULT="failure",
            WORKLOAD_RESULT="skipped",
            FINALIZE_RESULT="skipped",
            MODEL_REPORT="",
            SHUTDOWN_CONFIRMED="",
        )
        self.assertIn("initialize", issue_payload(env)["body"])
        env["GITHUB_REPOSITORY"] = "untrusted/fork"
        with self.assertRaises(ValueError):
            issue_payload(env)

    def test_retry_deduplicates_and_never_edits_manual_issues(self):
        env = self.env()
        payload = issue_payload(env)
        calls = []

        def request(path, data=None):
            calls.append((path, data))
            if data is None:
                return [
                    {"number": 7, "body": "manual issue", "html_url": "manual"},
                    {"number": 8, "body": payload["body"], "html_url": "automatic"},
                ]
            self.fail("The same run must not write another issue")

        self.assertEqual(publish(env, request), "automatic")
        self.assertEqual(len(calls), 1)
        created = []

        def empty_repository(path, data=None):
            if data is None:
                return [{"number": 7, "body": "manual issue", "html_url": "manual"}]
            created.append((path, data))
            return {"html_url": "new-issue"}

        self.assertEqual(publish(env, empty_repository), "new-issue")
        self.assertEqual(created[0][0], "repos/VictorWong123/tokenomics/issues")
