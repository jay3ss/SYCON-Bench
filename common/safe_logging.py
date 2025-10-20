import logging
import re
from typing import Any, Mapping

# Patterns to identify common API key formats (e.g., OpenAI keys start with sk-...)
# Patterns to identify common API key formats (e.g., OpenAI keys start with sk-...)
# Use simple patterns for fixed-width matches and group-capturing patterns for key/value pairs.
REDACT_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9\-_]{20,}"),  # OpenAI-style keys (no groups)
    # Capture the "api_key: 'VALUE'" or 'api_key="VALUE"' pattern so we can preserve the key and quotes
    re.compile(r"(?i)(api_key\s*[:=]\s*['\"])([^'\"]+)(['\"])"),
    # Capture "Authorization: Bearer TOKEN" preserving the prefix
    re.compile(r"(?i)(authorization\s*:\s*bearer\s)([A-Za-z0-9\._\-]+)"),
]


def redact_text(s: str) -> str:
    """Redact sensitive substrings in s according to REDACT_PATTERNS."""
    if not isinstance(s, str):
        return s
    out = s
    for pat in REDACT_PATTERNS:
        try:
            # If pattern has capturing groups, preserve surrounding context and only redact the sensitive group(s)
            if pat.groups == 0:
                out = pat.sub("***REDACTED***", out)
            else:
                def _repl(m):
                    # If there are 3 groups (prefix, secret, suffix/quote), keep prefix and suffix
                    if m.lastindex >= 3:
                        return m.group(1) + "***REDACTED***" + (m.group(3) or "")
                    # If only prefix+secret, keep prefix
                    return m.group(1) + "***REDACTED***"
                out = pat.sub(_repl, out)
        except Exception:
            # If anything goes wrong, fall back to a best-effort replacement
            out = pat.sub("***REDACTED***", out)
    return out


def _redact_mapping(mapping: Mapping[str, Any]) -> dict:
    """Return a shallow copy of mapping with common sensitive keys redacted."""
    redacted = {}
    for k, v in mapping.items():
        if isinstance(k, str) and k.lower() in ("api_key", "apikey", "openai_api_key", "secret", "authorization"):
            redacted[k] = "***REDACTED***"
        else:
            if isinstance(v, str):
                redacted[k] = redact_text(v)
            else:
                redacted[k] = v
    return redacted


class RedactingFilter(logging.Filter):
    """
    Logging filter that redacts common API keys and sensitive values from messages
    and mapping-style arguments.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Sanitize record.msg if it's a string
        try:
            if isinstance(record.msg, str):
                record.msg = redact_text(record.msg)
        except Exception:
            # Be conservative: don't block logging on filter failure
            pass

        # Sanitize mapping-style arguments (e.g., logging.info("Arguments: %s", args_dict))
        try:
            if isinstance(record.args, dict):
                record.args = _redact_mapping(record.args)
            elif isinstance(record.args, tuple):
                # Convert tuple args to tuple where strings are redacted
                new_args = []
                for a in record.args:
                    if isinstance(a, dict):
                        new_args.append(_redact_mapping(a))
                    elif isinstance(a, str):
                        new_args.append(redact_text(a))
                    else:
                        new_args.append(a)
                record.args = tuple(new_args)
        except Exception:
            pass

        # Additionally, ensure the formatted message (if present) does not leak keys.
        try:
            if hasattr(record, "message"):
                if isinstance(record.message, str):
                    record.message = redact_text(record.message)
        except Exception:
            pass

        return True


def setup_logging(verbose: bool = False) -> None:
    """
    Configure root logger with safe defaults and attach a RedactingFilter to all handlers.

    Usage: call this in place of a direct logging.basicConfig(...) call.
    """
    level = logging.DEBUG if verbose else logging.INFO
    # Configure basic logging; handlers will be present on the root logger
    logging.basicConfig(level=level, handlers=[logging.StreamHandler()], format="%(levelname)s: %(message)s")

    root = logging.getLogger()
    filt = RedactingFilter()
    # Attach filter to existing handlers
    for h in list(root.handlers):
        h.addFilter(filt)
    # Also attach filter to the root logger itself (catches child loggers)
    root.addFilter(filt)
