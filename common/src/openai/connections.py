"""Provider selection and local validation. Never persist resolved credentials."""

import ipaddress
import os
import re
from urllib.parse import urlsplit, urlunsplit

import httpx
from openai import OpenAI
from common.src.configs.schema import connection_section, values


class RequestFailure(RuntimeError):
    def __init__(self, status_code, exhausted=False):
        self.status_code = status_code if type(status_code) is int else None
        reason = "exhausted retries" if exhausted else "failed"
        super().__init__(f"Model request {reason} (HTTP {self.status_code or 'unknown'}). Check authentication and request settings.")


def endpoint(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Set an explicit endpoint.")
    parsed = urlsplit(value.strip())
    if (parsed.scheme not in {"https", "http"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Endpoint must be an HTTP(S) URL without credentials, query, or fragment.")
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("Invalid endpoint port.") from None
    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    if port and port != (443 if parsed.scheme == "https" else 80):
        host += f":{port}"
    return urlunsplit((parsed.scheme, host, parsed.path.rstrip("/"), "", ""))


def validate(values, live=False):
    provider = values.get("provider")
    auth = values.get("auth_mode")
    if values.get("openai_key"):
        raise ValueError("Plaintext keys require migration with python -m src.migrate_credentials.")
    if provider not in {"openai", "compatible", "local"}:
        raise ValueError("Select provider openai, compatible, or local.")
    url = endpoint(values.get("openai_endpoint"))
    if not isinstance(values.get("openai_model"), str) or values["openai_model"] in {"", "<TBD>"}:
        raise ValueError("Set a model identifier.")
    if values.get("message_role", "system") not in {"system", "developer"}:
        raise ValueError("message_role must be system or developer.")
    if values.get("stream") or values.get("with_raw_response"):
        raise ValueError("Streaming and raw responses are not supported by this pipeline.")
    if provider == "local":
        try:
            local = ipaddress.ip_address(urlsplit(url).hostname).is_loopback
        except ValueError:
            local = False
        if not local or auth != "none":
            raise ValueError("Local mode requires a literal loopback endpoint and auth_mode none.")
    else:
        if not url.startswith("https://"):
            raise ValueError("Hosted providers require HTTPS.")
        if auth != "api-key":
            raise ValueError("Hosted providers require auth_mode api-key.")
        if endpoint(values.get("credential_endpoint")) != url:
            raise ValueError("Explicitly rebind credential_endpoint before using a changed endpoint.")
        if values.get("credential_provider") != provider:
            raise ValueError("Explicitly rebind credential_provider before using a changed provider.")
        ref = values.get("api_key_env", "")
        if not isinstance(ref, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", ref):
            raise ValueError("api_key_env must name an environment variable.")
        if live and not os.environ.get(ref, "").strip():
            raise ValueError(f"Required credential environment variable {ref} is empty.")
    timeout = values.get("request_timeout", 120.0)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout < float("inf"):
        raise ValueError("request_timeout must be positive and finite.")
    retries = values.get("max_retries", 5)
    if type(retries) is not int or retries < 0:
        raise ValueError("max_retries must be a nonnegative integer.")
    for field in ("min_retry_delay", "max_retry_delay"):
        value = values.get(field, 1.0 if field == "min_retry_delay" else 30.0)
        if type(value) not in (int, float) or not 0 <= value < float("inf"):
            raise ValueError(f"{field} must be finite and nonnegative.")
    for field in ("max_tokens", "max_completion_tokens", "top_logprobs"):
        if field in values and type(values[field]) is not int:
            raise ValueError(f"{field} must be an integer.")
    if values.get("max_tokens", -1) > 0 and values.get("max_completion_tokens", -1) > 0:
        raise ValueError("Choose only one token limit parameter.")
    for field, upper in (("temperature", 2), ("top_p", 1), ("frequency_penalty", 2), ("presence_penalty", 2)):
        if field in values:
            value = values[field]
            if type(value) not in (int, float) or not -1 <= value <= upper:
                raise ValueError(f"{field} is outside the supported range.")
    return url


class LocalClient(OpenAI):
    @property
    def auth_headers(self):
        return {}


def client(values):
    url = validate(values, live=True)
    options = dict(
        max_retries=0, timeout=values.get("request_timeout", 120.0),
        http_client=httpx.Client(follow_redirects=False, trust_env=False),
    )
    if values["provider"] == "local":
        return LocalClient(base_url=url, api_key="local-no-auth", **options)
    return OpenAI(base_url=url, api_key=os.environ[values["api_key_env"]], **options)


def normalized(section):
    return values(connection_section(section))
