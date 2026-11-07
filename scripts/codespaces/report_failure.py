#!/usr/bin/env python3
"""File a bounded test-failure issue after the Linux lifecycle has finished."""
import hashlib
import json
import os
import re
import subprocess

REPOSITORY = "VictorWong123/tokenomics"


def slug(value):
    return (
        value
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value)
        else "unknown"
    )


def issue_payload(env):
    if env["GITHUB_REPOSITORY"] != REPOSITORY:
        raise ValueError("Unexpected repository")
    sha, run, attempt = (
        env[key] for key in ("GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT")
    )
    if not re.fullmatch(r"[a-f0-9]{40}", sha) or not all(
        re.fullmatch(r"[0-9]+", value) for value in (run, attempt)
    ):
        raise ValueError("Invalid source or run identity")
    phases = {
        phase: env.get(phase.upper() + "_RESULT", "unknown")
        for phase in ("initialize", "workload", "finalize")
    }
    failed = [phase for phase, result in phases.items() if result == "failure"]
    if not failed:
        return None
    raw = env.get("MODEL_REPORT", "")
    if len(raw) > 8192:
        raise ValueError("Oversized model report")
    try:
        report = json.loads(raw) if raw else {}
    except ValueError:
        report = {}
    if not isinstance(report, dict):
        report = {}
    first = report.get("first_mismatch") or {}
    if not isinstance(first, dict):
        first = {}
    mismatch = first.get("mismatch") or {}
    if not isinstance(mismatch, dict):
        mismatch = {}
    case = {key: slug(first.get(key)) for key in ("prompt_id", "configuration_id")}
    case["mismatch"] = {
        key: value
        for key, value in mismatch.items()
        if key in {"field", "position", "reference", "actual", "expected"}
        and (type(value) is int or slug(value) != "unknown")
    }
    stage = slug(report.get("stage"))
    signature = json.dumps(
        [
            failed,
            stage,
            case["prompt_id"],
            case["configuration_id"],
            mismatch.get("field"),
        ],
        sort_keys=True,
    )
    marker = (
        "<!-- tokenomics-linux-failure:"
        + hashlib.sha256(signature.encode()).hexdigest()[:20]
        + " -->"
    )
    url = f"https://github.com/{REPOSITORY}/actions/runs/{run}/attempts/{attempt}"
    error_type = slug(report.get("error_type"))
    checked = report.get("checked_comparisons")
    if type(checked) is not int or not 0 <= checked <= 100000:
        checked = "unknown"
    body = (
        f"{marker}\nLinux verification failed in {', '.join(failed)} (stage `{stage}`).\n\n"
        f"Source: `{sha}`\nRun: {url}\n"
        f"Failure type: `{error_type}`\nCompleted comparisons: {checked}\n"
        f"Codespace shutdown confirmed: **{env.get('SHUTDOWN_CONFIRMED') == 'true'}**\n\n"
        f"First failing test case:\n```json\n{json.dumps(case, indent=2, sort_keys=True)}\n```\n\n"
        "See the run's logs and bounded artifact for the complete evidence. "
        "A successful cleanup does not mean the tests passed."
    )
    return {
        "title": f"Linux verification failed: {stage} / {case['configuration_id']}",
        "body": body,
        "marker": marker,
        "run_url": url,
    }


def api(path, payload=None):
    command = ["gh", "api", path]
    if payload is not None:
        command += [
            "--method",
            "PATCH" if re.search(r"/issues/[0-9]+$", path) else "POST",
            "--input",
            "-",
        ]
    result = subprocess.run(
        command,
        input=None if payload is None else json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    return json.loads(result.stdout)


def publish(env, request=api):
    payload = issue_payload(env)
    if payload is None:
        return None
    base = f"repos/{REPOSITORY}/issues"
    # Match only our own marker; never modify a hand-written issue or PR.
    page = 1
    while True:
        issues = request(f"{base}?state=open&per_page=100&page={page}")
        for issue in issues:
            if "pull_request" not in issue and payload["marker"] in (
                issue.get("body") or ""
            ):
                if payload["run_url"] in issue["body"]:
                    return issue["html_url"]
                body = (issue["body"] + "\n\n---\n\n" + payload["body"])[-60000:]
                request(f"{base}/{int(issue['number'])}", {"body": body})
                return issue["html_url"]
        if len(issues) < 100:
            break
        page += 1
    created = request(base, {key: payload[key] for key in ("title", "body")})
    return created["html_url"]


if __name__ == "__main__":
    result = publish(os.environ)
    if result:
        print("TOKENOMICS_FAILURE_ISSUE " + result)
