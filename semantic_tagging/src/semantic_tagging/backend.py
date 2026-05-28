import abc
from concurrent.futures import ThreadPoolExecutor
import json
import os
import threading
import time
from typing import Any

from .config import BackendConfig
from .types import BackendResponse, PromptRequest


class InferenceBackend(abc.ABC):
    @abc.abstractmethod
    def generate_batch(self, requests_batch: list[PromptRequest]) -> list[BackendResponse]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def backend_name(self) -> str:
        raise NotImplementedError

    def debug_status(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "model_name": self.model_name,
        }


class MockBackend(InferenceBackend):
    def __init__(self, *, model_name: str = "mock-model", canned_by_text: dict[str, dict[str, Any]] | None = None) -> None:
        self._model_name = model_name
        self._canned_by_text = canned_by_text or {}

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def backend_name(self) -> str:
        return "mock"

    def generate_batch(self, requests_batch: list[PromptRequest]) -> list[BackendResponse]:
        responses: list[BackendResponse] = []
        for request in requests_batch:
            payload = self._canned_by_text.get(request.raw_text, _default_mock_payload(request.organ))
            responses.append(
                BackendResponse(
                    request_id=request.request_id,
                    raw_output=json.dumps(payload, ensure_ascii=False),
                    model_name=self.model_name,
                    backend_name=self.backend_name,
                    prompt_text=request.prompt_text,
                    finish_reason="stop",
                )
            )
        return responses


class VLLMServerBackend(InferenceBackend):
    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("vLLM server backend requires the 'requests' package.") from exc
        self._requests = requests
        api_key = os.environ.get(config.api_key_env, "EMPTY")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._thread_local = threading.local()

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def backend_name(self) -> str:
        return "vllm_server"

    def generate_batch(self, requests_batch: list[PromptRequest]) -> list[BackendResponse]:
        if not requests_batch:
            return []
        concurrency = max(1, min(self.config.request_concurrency, len(requests_batch)))
        if concurrency == 1:
            return [self._generate_one(request) for request in requests_batch]
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            return list(executor.map(self._generate_one, requests_batch))

    def _generate_one(self, request: PromptRequest) -> BackendResponse:
        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": [
                {"role": "user", "content": request.prompt_text},
            ],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.use_response_format_json:
            payload["response_format"] = {"type": "json_object"}
        if self.config.use_guided_json:
            payload["extra_body"] = {"guided_json": True}
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        last_error: Exception | None = None
        for attempt in range(self.config.request_retries + 1):
            try:
                response = self._get_session().post(
                    url,
                    headers=self._headers,
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )
                response.raise_for_status()
                response_payload = response.json()
                choice = response_payload["choices"][0]
                content = choice["message"]["content"]
                usage = response_payload.get("usage") or {}
                return BackendResponse(
                    request_id=request.request_id,
                    raw_output=str(content),
                    model_name=self.model_name,
                    backend_name=self.backend_name,
                    prompt_text=request.prompt_text,
                    finish_reason=choice.get("finish_reason"),
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                )
            except self._requests.exceptions.RequestException as exc:
                last_error = exc
                self._reset_session()
                if attempt >= self.config.request_retries:
                    break
                sleep_seconds = self.config.retry_backoff_seconds * (attempt + 1)
                print(
                    f"[semantic_tagging backend] retrying request_id={request.request_id} "
                    f"attempt={attempt + 1}/{self.config.request_retries} "
                    f"after={sleep_seconds:.1f}s error={type(exc).__name__}: {exc}"
                )
                time.sleep(sleep_seconds)
        raise RuntimeError(
            f"vLLM request failed after {self.config.request_retries + 1} attempts "
            f"for request_id={request.request_id}: {last_error}"
        )

    def _build_session(self):
        session = self._requests.Session()
        adapter = self._requests.adapters.HTTPAdapter(
            pool_connections=max(4, self.config.request_concurrency),
            pool_maxsize=max(4, self.config.request_concurrency),
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _get_session(self):
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = self._build_session()
            self._thread_local.session = session
        return session

    def _reset_session(self) -> None:
        session = getattr(self._thread_local, "session", None)
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
        self._thread_local.session = self._build_session()

    def debug_status(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "backend_name": self.backend_name,
            "model_name": self.model_name,
            "base_url": self.config.base_url,
            "request_concurrency": self.config.request_concurrency,
            "request_retries": self.config.request_retries,
        }
        try:
            response = self._get_session().get(
                self.config.base_url.rstrip("/") + "/models",
                headers=self._headers,
                timeout=min(self.config.timeout_seconds, 10),
            )
            payload["models_status_code"] = response.status_code
            payload["models_ok"] = response.ok
            if response.ok:
                body = response.json()
                payload["model_ids"] = [item.get("id") for item in body.get("data", [])[:8]]
        except self._requests.exceptions.RequestException as exc:
            payload["models_ok"] = False
            payload["models_error"] = f"{type(exc).__name__}: {exc}"
        return payload


class TransformersLocalBackend(InferenceBackend):
    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Transformers backend requires torch and transformers.") from exc
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        if getattr(self._tokenizer, "pad_token", None) is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            device_map="auto",
        )

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def backend_name(self) -> str:
        return "transformers_local"

    def generate_batch(self, requests_batch: list[PromptRequest]) -> list[BackendResponse]:
        prompts = [request.prompt_text for request in requests_batch]
        tokenized = self._tokenizer(prompts, return_tensors="pt", padding=True)
        tokenized = {key: value.to(self._model.device) for key, value in tokenized.items()}
        outputs = self._model.generate(
            **tokenized,
            max_new_tokens=self.config.max_tokens,
            do_sample=False,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            pad_token_id=self._tokenizer.pad_token_id,
        )
        responses: list[BackendResponse] = []
        input_lengths = tokenized["input_ids"].shape[1]
        for request, output_ids in zip(requests_batch, outputs):
            generated_ids = output_ids[input_lengths:]
            text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
            responses.append(
                BackendResponse(
                    request_id=request.request_id,
                    raw_output=text.strip(),
                    model_name=self.model_name,
                    backend_name=self.backend_name,
                    prompt_text=request.prompt_text,
                    finish_reason="stop",
                )
            )
        return responses


def build_backend(config: BackendConfig) -> InferenceBackend:
    kind = config.kind.lower().strip()
    if kind == "mock":
        return MockBackend(model_name=config.model_name)
    if kind == "vllm":
        return VLLMServerBackend(config)
    if kind == "transformers":
        return TransformersLocalBackend(config)
    raise ValueError(f"Unsupported backend kind: {config.kind}")


def _default_mock_payload(organ: str) -> dict[str, Any]:
    return {
        "organ": organ,
        "normality": "normal",
        "polarity": "negative",
        "certainty": "definite",
        "primary_subtype": None,
        "secondary_subtypes": [],
        "modifiers": [],
        "evidence_spans": [],
        "confidence": 0.75,
        "decision_status": "accepted",
        "decision_source": "mock_backend",
        "ontology_version": "v1",
        "proposed_new_subtype": None,
        "proposed_new_family": None,
        "validation_flags": [],
    }
