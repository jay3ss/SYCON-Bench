import logging
from common import safe_logging

def test_redact_openai_key(caplog):
    safe_logging.setup_logging(verbose=True)
    logger = logging.getLogger("test_logger")
    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
    with caplog.at_level(logging.INFO):
        logger.info("Using key: %s", secret)
    logs = "\n".join([r.getMessage() for r in caplog.records])
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in logs
    assert "***REDACTED***" in logs

def test_redact_mapping_api_key(caplog):
    safe_logging.setup_logging(verbose=False)
    logger = logging.getLogger("test2")
    args = {"api_key": "sk-0123456789ABCDEFGHIJKLMNOP"}
    with caplog.at_level(logging.INFO):
        logger.info("Arguments: %s", args)
    logs = "\n".join([r.getMessage() for r in caplog.records])
    assert "sk-0123456789ABCDEFGHIJKLMNOP" not in logs
    assert "***REDACTED***" in logs
