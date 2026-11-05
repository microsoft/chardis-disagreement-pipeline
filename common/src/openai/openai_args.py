from common.src.configs.config_value import ConfigValue
from common.src.configs.configurable_base import ConfigurableBase
from common.src.configs.schema import (
    AliasAction, FIELD_ALIASES, connection_section, runtime_config, section_name,
)
from openai import OpenAI, APIConnectionError, APITimeoutError
from common.src.openai import connections
import argparse
import numpy as np
from pathlib import Path
import random
import time


class OpenAIArgs(ConfigurableBase):
    """
    Configuration parameters for connecting to the selected model service.
    Includes model selection, endpoint details, and generation settings.
    """
    DEFAULT_FREQUENCY_PENALTY = 0.0
    DEFAULT_MAX_TOKENS = -1
    DEFAULT_PRESENCE_PENALTY = 0.0
    DEFAULT_TEMPERATURE = 1.0
    DEFAULT_TOP_P = 1.0
    DEFAULT_LOGPROBS = False
    DEFAULT_TOP_LOGPROBS = 1
    DEFAULT_SCOPE = ""
    DEFAULT_MIN_RETRY_DELAY = 1.0
    DEFAULT_MAX_RETRY_DELAY = 30.0
    DEFAULT_MAX_RETRIES = 5
    DEFAULT_STREAM = False

    def __init__(self):
        """Initialize an OpenAIArgs instance with API connection details and generation settings."""
        self.openai_model = ConfigValue(
            None,
            True,
            "The model identifier or deployment name to use.")

        self.openai_endpoint = ConfigValue(
            None,
            True,
            "The API endpoint to use.")

        self.openai_api_version = ConfigValue(
            "",
            False,
            "Legacy compatibility field. Not sent by current adapters.")

        self.temperature = ConfigValue(
            OpenAIArgs.DEFAULT_TEMPERATURE,
            False,
            f"Controls randomness. Lowering the temperature means that the model will produce more repetitive and deterministic responses (default = {OpenAIArgs.DEFAULT_TEMPERATURE}).")

        self.max_tokens = ConfigValue(
            OpenAIArgs.DEFAULT_MAX_TOKENS,
            False,
            f"Set a limit on the number of tokens per model response. Any non-positive value will not be passed to the API (default = {OpenAIArgs.DEFAULT_MAX_TOKENS}).")

        self.max_completion_tokens = ConfigValue(
            OpenAIArgs.DEFAULT_MAX_TOKENS,
            False,
            f"Set a limit on the number of tokens per model response. Any non-positive value will not be passed to the API (default = {OpenAIArgs.DEFAULT_MAX_TOKENS}).")

        self.top_p = ConfigValue(
            OpenAIArgs.DEFAULT_TOP_P,
            False,
            f"Similar to temperature, this controls randomness but uses a different method (default = {OpenAIArgs.DEFAULT_TOP_P}).")

        self.frequency_penalty = ConfigValue(
            OpenAIArgs.DEFAULT_FREQUENCY_PENALTY,
            False,
            f"Reduce the chance of repeating a token proportionally based on how often it has appeared in the text so far (default = {OpenAIArgs.DEFAULT_FREQUENCY_PENALTY}).")

        self.presence_penalty = ConfigValue(
            OpenAIArgs.DEFAULT_PRESENCE_PENALTY,
            False,
            f"Reduce the chance of repeating any token that has appeared in the text at all so far (default = {OpenAIArgs.DEFAULT_PRESENCE_PENALTY}).")

        self.openai_key = ConfigValue(
            "",
            False,
            "Legacy plaintext credential. Migrate to api_key_env before use.")

        self.logprobs = ConfigValue(
            OpenAIArgs.DEFAULT_LOGPROBS,
            False,
            f"Whether to return logprobs for the generated tokens (default = {OpenAIArgs.DEFAULT_LOGPROBS}).")

        self.top_logprobs = ConfigValue(
            OpenAIArgs.DEFAULT_TOP_LOGPROBS,
            False,
            f"The number of top logprobs to return if logprobs is enabled (default = {OpenAIArgs.DEFAULT_TOP_LOGPROBS}).")

        self.scope = ConfigValue(
            OpenAIArgs.DEFAULT_SCOPE,
            False,
            "Legacy compatibility field. Current adapters do not acquire OAuth tokens.")

        self.min_retry_delay = ConfigValue(
            OpenAIArgs.DEFAULT_MIN_RETRY_DELAY,
            False,
            f"The minimum delay between retries in seconds (default = {OpenAIArgs.DEFAULT_MIN_RETRY_DELAY}).")

        self.max_retry_delay = ConfigValue(
            OpenAIArgs.DEFAULT_MAX_RETRY_DELAY,
            False,
            f"The maximum delay between retries in seconds (default = {OpenAIArgs.DEFAULT_MAX_RETRY_DELAY}).")

        self.max_retries = ConfigValue(
            OpenAIArgs.DEFAULT_MAX_RETRIES,
            False,
            f"The maximum number of retries for transient errors. Any non-positive value means no retries (default = {OpenAIArgs.DEFAULT_MAX_RETRIES}).")

        self.stream = ConfigValue(
            OpenAIArgs.DEFAULT_STREAM,
            False,
            f"Whether to use streaming responses from the API (default = {OpenAIArgs.DEFAULT_STREAM}).")

        self.with_raw_response = ConfigValue(
            False,
            False,
            f"Whether to return the raw response from the API (default = False).")

        self.provider = ConfigValue(
            "compatible",
            False,
            "Provider adapter: openai, compatible, or local.")
        self.auth_mode = ConfigValue("api-key", False, "api-key, or none for literal loopback endpoints.")
        self.api_key_env = ConfigValue("MODEL_API_KEY", False, "Environment variable holding the key.")
        self.credential_endpoint = ConfigValue("", False, "Explicit approved credential destination.")
        self.credential_provider = ConfigValue("compatible", False, "Provider explicitly bound to this credential.")
        self.message_role = ConfigValue("system", False, "Trusted instruction role: system or developer.")
        self.request_timeout = ConfigValue(120.0, False, "Per-request timeout in seconds.")

    _KEY_ALIASES = FIELD_ALIASES

    def load_config(self, config: dict, value_name: str, required: bool = True):
        config = config if config is not None else {}
        converted = runtime_config(config)
        value_name = section_name(value_name)
        section = converted.get(value_name)
        if isinstance(section, dict):
            if required and "auth_mode" not in section:
                raise ValueError(f"{section_name(value_name, public=True)} requires explicit auth_mode and credential destination bindings.")
            converted[value_name] = connection_section(section)
        error = super().load_config(converted, value_name, required)
        config.clear()
        config.update(converted)
        return error

    @classmethod
    def from_model_catalog(cls, model_name: str, models_path: Path = None, connection=None):
        """Combine catalog generation settings with an explicitly supplied connection."""
        from common.src.openai.model_catalog import load_model_config

        config = load_model_config(model_name, models_path)
        instance = cls()
        settings = {key: config[key] for key in vars(instance) if key in config}
        settings["openai_model"] = config["model_name"]
        settings.update(connections.normalized(connection or {}))
        instance.load_config({"openai": {key: {"value": value} for key, value in settings.items()}},
                             "openai", required=False)
        return instance

    def to_args(parser: argparse.ArgumentParser, prefix="", required=True):
        """Register model connection CLI arguments on the provided parser.

        Adds model, endpoint, and tuning parameters to an ArgumentParser,
        optionally namespaced with a prefix for modular configurations.
        """
        parser.add_argument(
            f"--{prefix}model", f"--{prefix}openai_model",
            dest=f"{prefix}openai_model", action=AliasAction, metavar="MODEL",
            type=str,
            required=required,
            help="The model identifier or deployment name to use.")

        parser.add_argument(
            f"--{prefix}endpoint", f"--{prefix}openai_endpoint", f"--{prefix}base_url",
            dest=f"{prefix}openai_endpoint", action=AliasAction, metavar="ENDPOINT",
            type=str,
            required=required,
            help="The API endpoint to use.")

        parser.add_argument(
            f"--{prefix}api_version", f"--{prefix}openai_api_version",
            dest=f"{prefix}openai_api_version", action=AliasAction, metavar="API_VERSION",
            type=str,
            default="",
            help="Legacy compatibility field. Not sent by current adapters.")
        for field in ("provider", "auth_mode", "api_key_env", "credential_endpoint",
                      "credential_provider", "message_role"):
            parser.add_argument(f"--{prefix}{field}", type=str, default=None)

        parser.add_argument(
            f"--{prefix}temperature",
            type=float,
            default=OpenAIArgs.DEFAULT_TEMPERATURE,
            help=f"Controls randomness. Lowering the temperature means that the model will produce more repetitive and deterministic responses (default = {OpenAIArgs.DEFAULT_TEMPERATURE}).")

        parser.add_argument(
            f"--{prefix}max_tokens",
            type=int,
            default=OpenAIArgs.DEFAULT_MAX_TOKENS,
            help=f"Legacy parameter for the maximum number of tokens to generate in completion. For newer models (e.g. o1 series) this may be ignored or unsupported. Any non-positive value will not be passed to the API (default = {OpenAIArgs.DEFAULT_MAX_TOKENS}).")

        parser.add_argument(
            f"--{prefix}max_completion_tokens",
            type=int,
            default=OpenAIArgs.DEFAULT_MAX_TOKENS,
            help=f"The upper bound for number of tokens to generate in the completion, including reasoning tokens. Any non-positive value will not be passed to the API (default = {OpenAIArgs.DEFAULT_MAX_TOKENS}).")

        parser.add_argument(
            f"--{prefix}top_p",
            type=float,
            default=OpenAIArgs.DEFAULT_TOP_P,
            help=f"Similar to temperature, this controls randomness but uses a different method (default = {OpenAIArgs.DEFAULT_TOP_P}).")

        parser.add_argument(
            f"--{prefix}frequency_penalty",
            type=float,
            default=OpenAIArgs.DEFAULT_FREQUENCY_PENALTY,
            help=f"Reduce the chance of repeating a token proportionally based on how often it has appeared in the text so far (default = {OpenAIArgs.DEFAULT_FREQUENCY_PENALTY}).")

        parser.add_argument(
            f"--{prefix}presence_penalty",
            type=float,
            default=OpenAIArgs.DEFAULT_PRESENCE_PENALTY,
            help=f"Reduce the chance of repeating any token that has appeared in the text at all so far (default = {OpenAIArgs.DEFAULT_PRESENCE_PENALTY}).")

        parser.add_argument(
            f"--{prefix}key", f"--{prefix}openai_key",
            dest=f"{prefix}openai_key", action=AliasAction, metavar="KEY",
            type=str,
            default=None,
            help="Legacy plaintext key option. Use api_key_env instead; plaintext keys require migration.")

        parser.add_argument(
            f"--{prefix}logprobs",
            action="store_true",
            default=OpenAIArgs.DEFAULT_LOGPROBS,
            help=f"Whether to return logprobs for the generated tokens (default = {OpenAIArgs.DEFAULT_LOGPROBS}).")

        parser.add_argument(
            f"--{prefix}top_logprobs",
            type=int,
            default=OpenAIArgs.DEFAULT_TOP_LOGPROBS,
            help=f"The number of top logprobs to return if logprobs is enabled (default = {OpenAIArgs.DEFAULT_TOP_LOGPROBS}).")

        parser.add_argument(
            f"--{prefix}scope",
            type=str,
            default=OpenAIArgs.DEFAULT_SCOPE,
            help="Legacy compatibility field. Current adapters do not acquire OAuth tokens.")

        parser.add_argument(
            f"--{prefix}min_retry_delay",
            type=float,
            default=OpenAIArgs.DEFAULT_MIN_RETRY_DELAY,
            help=f"The minimum delay between retries in seconds (default = {OpenAIArgs.DEFAULT_MIN_RETRY_DELAY}).")

        parser.add_argument(
            f"--{prefix}max_retry_delay",
            type=float,
            default=OpenAIArgs.DEFAULT_MAX_RETRY_DELAY,
            help=f"The maximum delay between retries in seconds (default = {OpenAIArgs.DEFAULT_MAX_RETRY_DELAY}).")

        parser.add_argument(
            f"--{prefix}max_retries",
            type=int,
            default=OpenAIArgs.DEFAULT_MAX_RETRIES,
            help=f"The maximum number of retries for transient errors. Any non-positive value means no retries (default = {OpenAIArgs.DEFAULT_MAX_RETRIES}).")

        parser.add_argument(
            f"--{prefix}stream",
            action="store_true",
            default=OpenAIArgs.DEFAULT_STREAM,
            help=f"Whether to use streaming responses from the API (default = {OpenAIArgs.DEFAULT_STREAM}).")

        parser.add_argument(
            f"--{prefix}with_raw_response",
            action="store_true",
            default=False,
            help=f"Whether to return the raw response from the API (default = False).")

    def from_args(args: argparse.Namespace, prefix=""):
        """Construct an OpenAIArgs instance from CLI arguments, using the given prefix to map namespaced parameters."""
        instance = OpenAIArgs()
        settings = {name: getattr(args, prefix + name) for name in vars(instance)
                    if getattr(args, prefix + name, None) is not None}
        if instance.load_config({"openai": {key: {"value": value} for key, value in settings.items()}}, "openai"):
            raise ValueError("Incomplete model connection arguments.")
        return instance

    def get_client(self):
        return connections.client(vars(self))

    def get_model_messages(self, system_prompt: str, user_content: str) -> list:
        """Separate trusted instructions from user content using the configured role."""

        role = self.message_role
        if role not in {"system", "developer"}:
            raise ValueError("Select message_role system or developer explicitly.")

        return [
            {"role": role,
             "content": [{
                 "type": "text",
                 "text": system_prompt
             }]},
            {"role": "user",
             "content": [{
                     "type": "text",
                         "text": user_content
                         }]}
        ]

    def get_logprobs(chat_completion) -> list:
        """
            Extracts and returns log-probabilities for specific tokens ("0" or "1") 
            from a LLM API response, conditional on token context.

            The function iterates over a list of token-level responses, and for each token 
            that is either "0" or "1", it checks whether one of the five preceding tokens is "Value". 
            If this condition is met, the token's log-probability is exponentiated, converted to a 
            percentage, rounded to three decimal places, and added to the result list.

        Args:
            chat_completion (ChatCompletion): A chat completion response from the OpenAI API.

        Returns:
            extracted_logprobs (list): A list of percentage values representing the exponentiated 
                            log-probabilities of selected tokens.
        """
        if chat_completion.choices[0].logprobs is None or chat_completion.choices[0].logprobs.content is None:
            return None

        content = chat_completion.choices[0].logprobs.content
        extracted_logprobs = []

        for i, tok_lp in enumerate(content):
            # Strip whitespace once so we can reuse the cleaned token
            tok = tok_lp.token.strip()
            # A) token must be "0" or "1"
            if tok not in {"0", "1"}:
                continue

            # B) One of the previous ≤5 tokens must be "Value"
            window_start = max(0, i - 5)
            prev_tokens = (t.token.strip() for t in content[window_start:i])

            if any(t == "Value" for t in prev_tokens):
                logprobs = tok_lp.logprob
                logprobs = np.round(np.exp(logprobs)*100, 3)
                extracted_logprobs.append(logprobs)

        # `extracted_logprobs` now holds all matching log-probabilities
        return extracted_logprobs

    def get_api_chat_completion(self, client: OpenAI, messages: list, **extra_params):
        """
        Send a chat completion request using the configured model parameters.

        Builds the API payload dynamically from the instance attributes (e.g., token limits,
        temperature, penalties, and logprob options) and sends it via the provided client.

        Args:
            client (OpenAI): The initialized SDK client used to send the request.
            messages (list): The chat message list structured according to the model’s expected format.

        Returns:
            openai.types.chat.chat_completion.ChatCompletion: The API response containing the model output.
        """

        api_params = {
            "model": self.openai_model,
            "messages": messages,
            "stop": None,
            "stream": self.stream
        }

        # Merge any extra parameters (e.g. n for multiple completions)
        for k, v in extra_params.items():
            if v is not None:
                api_params[k] = v

        if self.max_tokens > 0:
            api_params["max_tokens"] = self.max_tokens

        if self.max_completion_tokens > 0:
            api_params["max_completion_tokens"] = self.max_completion_tokens

        if "max_tokens" in api_params and "max_completion_tokens" in api_params:
            raise ValueError("Choose only one token limit parameter.")

        if self.logprobs and self.top_logprobs > 0:
            api_params["logprobs"] = True
            api_params["top_logprobs"] = self.top_logprobs

        if self.temperature >= 0.0:
            api_params["temperature"] = self.temperature

        if self.top_p >= 0.0:
            api_params["top_p"] = self.top_p

        if self.frequency_penalty >= 0.0:
            api_params["frequency_penalty"] = self.frequency_penalty

        if self.presence_penalty >= 0.0:
            api_params["presence_penalty"] = self.presence_penalty

        retries = 0
        delay = self.min_retry_delay
        max_retries = max(0, self.max_retries)

        while retries <= max_retries:
            try:
                if self.with_raw_response:
                    return client.chat.completions.with_raw_response.create(**api_params)
                else:
                    return client.chat.completions.create(**api_params)
            except Exception as e:
                response_code = getattr(e, "status_code", "Unknown")

                if response_code in (408, 409, 429, 500, 502, 503, 504) or isinstance(e, (APIConnectionError, APITimeoutError)):
                    # Rate limit or transient error, wait and try again
                    retries += 1

                    if retries > self.max_retries:
                        # Reached maximum retries, raise error
                        raise connections.RequestFailure(response_code, exhausted=True) from None

                    # Respect server-provided Retry-After if present (seconds).
                    retry_after = None
                    resp = getattr(e, "response", None)
                    if resp is not None:
                        headers = getattr(resp, "headers", None)
                        if headers is not None:
                            for h in ("retry-after", "Retry-After",
                                      "x-ratelimit-reset-requests",
                                      "x-ratelimit-reset-tokens"):
                                v = headers.get(h) if hasattr(headers, "get") else None
                                if v:
                                    try:
                                        retry_after = float(v)
                                        break
                                    except (TypeError, ValueError):
                                        pass

                    sleep_for = retry_after if retry_after is not None and 0 <= retry_after < float("inf") else delay
                    sleep_for = max(0, min(sleep_for, self.max_retry_delay))
                    import sys as _sys
                    print(
                        f"[openai] HTTP {response_code} — retry "
                        f"{retries}/{self.max_retries} in {sleep_for:.1f}s",
                        file=_sys.stderr, flush=True,
                    )
                    time.sleep(sleep_for)
                    delay = min(delay * random.uniform(2, 3),
                                self.max_retry_delay)
                else:
                    raise connections.RequestFailure(response_code) from None
