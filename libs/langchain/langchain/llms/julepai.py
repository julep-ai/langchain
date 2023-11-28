import json
import logging
from typing import Any, Callable, Dict, List, Mapping, Optional

import requests
from requests import ConnectTimeout, ReadTimeout, RequestException
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.pydantic_v1 import Extra, root_validator
from langchain.utils.env import get_from_dict_or_env

DEFAULT_JULEPAI_SERVICE_URL = "https://api-alpha.julep.ai"
DEFAULT_JULEPAI_SERVICE_PATH = "/v1/completions"

logger = logging.getLogger(__name__)


class JulepAI(LLM):
    """JulepAI Service models.

    To use, you should have the environment variable ``JULEPAI_SERVICE_URL``,
    ``JULEPAI_SERVICE_PATH`` and ``JULEPAI_API_KEY`` set with your JulepAI
    Service, or pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain.llms import JulepAI

            julep_ai = JulepAI(
                julepai_service_url="JULEPAI_SERVICE_URL",
                julepai_service_path="JULEPAI_SERVICE_PATH",
                julepai_api_key="JULEPAI_API_KEY",
            )
    """  # noqa: E501

    """Key/value arguments to pass to the model. Reserved for future use"""
    model_kwargs: Optional[dict] = None

    """Optional"""

    julepai_service_url: Optional[str] = None
    julepai_service_path: Optional[str] = None
    julepai_api_key: Optional[str] = None
    model: str = "julep-ai/samantha-1-turbo"
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.95
    repetition_penalty: Optional[float] = 1.0
    top_k: Optional[int] = 1
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[List[str]] = None
    max_retries: Optional[int] = 10

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = {field.alias for field in cls.__fields__.values()}

        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name not in all_required_field_names:
                if field_name in extra:
                    raise ValueError(f"Found {field_name} supplied twice.")
                extra[field_name] = values.pop(field_name)
        values["model_kwargs"] = extra
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        julepai_service_url = get_from_dict_or_env(
            values,
            "julepai_service_url",
            "JULEPAI_SERVICE_URL",
            DEFAULT_JULEPAI_SERVICE_URL,
        )
        julepai_service_path = get_from_dict_or_env(
            values,
            "julepai_service_path",
            "JULEPAI_SERVICE_PATH",
            DEFAULT_JULEPAI_SERVICE_PATH,
        )
        julepai_api_key = get_from_dict_or_env(
            values, 
            "julepai_api_key", 
            "JULEPAI_API_KEY", 
            None,
        )

        if julepai_service_url.endswith("/"):
            julepai_service_url = julepai_service_url[:-1]
        if not julepai_service_path.startswith("/"):
            julepai_service_path = "/" + julepai_service_path

        values["julepai_service_url"] = julepai_service_url
        values["julepai_service_path"] = julepai_service_path
        values["julepai_api_key"] = julepai_api_key

        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Cohere API."""
        return {
            "model": "julep-ai/samantha-1-turbo",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop": ["<", "<|"],
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            "julepai_service_url": self.julepai_service_url,
            "julepai_service_path": self.julepai_service_path,
            **{"model_kwargs": _model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "julepai"

    def _invocation_params(
        self, stop_sequences: Optional[List[str]], **kwargs: Any
    ) -> dict:
        params = self._default_params
        if self.stop is not None and stop_sequences is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop is not None:
            params["stop"] = self.stop
        elif not params.get("stop"):
            params["stop"] = stop_sequences
        return {**params, **kwargs}

    @staticmethod
    def _process_response(response: Any, stop: Optional[List[str]]) -> str:
        return response["choices"][0]["text"]

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to JulepAI Service endpoint.
        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
        Returns:
            The string generated by the model.
        Example:
            .. code-block:: python
                response = julepai("Tell me a joke.")
        """
        params = self._invocation_params(stop, **kwargs)
        prompt = prompt.strip()

        response = completion_with_retry(
            self,
            prompt=prompt,
            params=params,
            url=f"{self.julepai_service_url}{self.julepai_service_path}",
        )
        _stop = params.get("stop_sequences")
        return self._process_response(response, _stop)


def make_request(
    self: JulepAI,
    prompt: str,
    url: str = f"{DEFAULT_JULEPAI_SERVICE_URL}{DEFAULT_JULEPAI_SERVICE_PATH}",
    params: Optional[Dict] = None,
) -> Any:
    """Generate text from the model."""
    params = params or {}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.julepai_api_key}",
    }

    body = {"prompt": prompt}

    # add params to body
    for key, value in params.items():
        body[key] = value

    # make request
    response = requests.post(url, headers=headers, json=body)

    if response.status_code != 200:
        raise Exception(
            f"Request failed with status code {response.status_code}"
            f" and message {response.text}"
        )

    return json.loads(response.text)


def _create_retry_decorator(llm: JulepAI) -> Callable[[Any], Any]:
    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterward
    max_retries = llm.max_retries if llm.max_retries is not None else 3
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type((RequestException, ConnectTimeout, ReadTimeout))
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def completion_with_retry(llm: JulepAI, **kwargs: Any) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm)

    @retry_decorator
    def _completion_with_retry(**_kwargs: Any) -> Any:
        return make_request(llm, **_kwargs)

    return _completion_with_retry(**kwargs)
