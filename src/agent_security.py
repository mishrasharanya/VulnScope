"""Transparent prototype controls for VulnScope's AI-agent security demo.

The prompt detector in this module is deliberately heuristic. It demonstrates
the v2 architecture but is not a substitute for a separately trained and
validated prompt-injection classifier. Tool authorization is deterministic by
design and must remain independent from the detector's confidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionSignal:
    name: str
    explanation: str
    pattern: str
    weight: float


DETECTION_SIGNALS = (
    DetectionSignal(
        "instruction_override",
        "Attempts to override or disregard existing instructions.",
        r"\b(ignore|disregard|forget|override)\b.{0,40}\b(instructions?|rules?|policy|policies|system)\b",
        0.45,
    ),
    DetectionSignal(
        "secret_extraction",
        "Requests secrets, credentials, tokens, or hidden instructions.",
        r"\b(reveal|show|print|extract|send|exfiltrate|leak)\b.{0,50}\b(secret|password|credential|token|api[ _-]?key|system prompt)\b",
        0.55,
    ),
    DetectionSignal(
        "authority_impersonation",
        "Claims a privileged role or special authorization.",
        r"\b(i am|act as|pretend to be)\b.{0,35}\b(admin|administrator|developer|root|security team)\b",
        0.25,
    ),
    DetectionSignal(
        "tool_coercion",
        "Pressures the agent to execute a sensitive action without checks.",
        r"\b(bypass|disable|skip|without)\b.{0,35}\b(approval|permission|authorization|safety|validation|checks?)\b",
        0.45,
    ),
    DetectionSignal(
        "encoded_payload",
        "Asks the agent to decode or execute an encoded payload.",
        r"\b(base64|hex|encoded|decode)\b.{0,40}\b(execute|run|command|payload|instructions?)\b",
        0.30,
    ),
)


TOOL_POLICIES = {
    "read_public_cve": {
        "risk": "low",
        "safe_decision": "ALLOW",
        "description": "Read public CVE information.",
    },
    "read_local_file": {
        "risk": "medium",
        "safe_decision": "REQUIRE_APPROVAL",
        "description": "Read a local file that may contain private data.",
    },
    "send_external_message": {
        "risk": "high",
        "safe_decision": "REQUIRE_APPROVAL",
        "description": "Send information outside the current application.",
    },
    "delete_file": {
        "risk": "high",
        "safe_decision": "REQUIRE_APPROVAL",
        "description": "Delete a local file.",
    },
    "access_credentials": {
        "risk": "prohibited",
        "safe_decision": "BLOCK",
        "description": "Access secrets or stored credentials.",
    },
}


def detect_prompt_injection(prompt: str) -> dict:
    """Return an explainable heuristic assessment of untrusted prompt text."""
    normalized = " ".join(str(prompt).split())
    matches = []
    score = 0.0
    for signal in DETECTION_SIGNALS:
        if re.search(signal.pattern, normalized, flags=re.IGNORECASE):
            matches.append(
                {
                    "signal": signal.name,
                    "explanation": signal.explanation,
                    "weight": signal.weight,
                }
            )
            score += signal.weight
    score = min(score, 1.0)
    suspicious = score >= 0.45
    return {
        "prompt": normalized,
        "risk_score": score,
        "classification": "SUSPICIOUS" if suspicious else "NO SIGNAL DETECTED",
        "suspicious": suspicious,
        "signals": matches,
        "detector_type": "prototype_heuristic_v1",
        "warning": (
            "Heuristic demonstration only. A clean result does not prove that "
            "a prompt is safe."
        ),
    }


def authorize_tool_call(
    tool_name: str,
    arguments: str,
    detection: dict,
) -> dict:
    """Apply deterministic policy to a simulated tool request."""
    policy = TOOL_POLICIES.get(tool_name)
    if policy is None:
        return {
            "decision": "BLOCK",
            "reason": "Unknown tools are denied by default.",
            "rule": "default_deny_unknown_tool",
        }

    argument_text = str(arguments).lower()
    sensitive_argument = bool(
        re.search(
            r"\b(secret|credential|password|token|api[ _-]?key|\.env|private key)\b",
            argument_text,
        )
    )
    if policy["risk"] == "prohibited" or sensitive_argument:
        return {
            "decision": "BLOCK",
            "reason": "The requested capability or arguments involve secrets.",
            "rule": "block_secret_access",
        }

    if detection.get("suspicious") and policy["risk"] in {"medium", "high"}:
        return {
            "decision": "BLOCK",
            "reason": (
                "Suspicious instructions cannot invoke a medium- or high-risk tool."
            ),
            "rule": "block_suspicious_privileged_action",
        }

    return {
        "decision": policy["safe_decision"],
        "reason": policy["description"],
        "rule": f"tool_policy_{tool_name}",
    }


def evaluate_agent_request(prompt: str, tool_name: str, arguments: str = "") -> dict:
    """Evaluate detection and authorization without executing any tool."""
    detection = detect_prompt_injection(prompt)
    authorization = authorize_tool_call(tool_name, arguments, detection)
    return {
        "detection": detection,
        "authorization": authorization,
        "tool_name": tool_name,
        "arguments": arguments,
        "executed": False,
    }
