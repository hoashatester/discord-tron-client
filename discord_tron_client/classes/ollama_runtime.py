import json
import logging
from typing import Any

import requests

from discord_tron_client.classes.app_config import AppConfig

logger = logging.getLogger(__name__)


class OllamaRuntime:
    def __init__(self, config: AppConfig):
        self.config = config

    def _base_url(self) -> str:
        return self.config.get_ollama_base_url()

    def _timeout(self) -> int:
        return self.config.get_ollama_timeout_seconds()

    def _post(self, path: str, *, json_body: dict[str, Any], stream: bool = False):
        response = requests.post(
            f"{self._base_url()}{path}",
            json=json_body,
            timeout=self._timeout(),
            stream=stream,
        )
        response.raise_for_status()
        return response

    def available_models(self) -> set[str]:
        try:
            response = requests.get(
                f"{self._base_url()}/api/tags",
                timeout=min(self._timeout(), 30),
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning(f"Failed to query local Ollama tags: {exc}")
            return set()
        models = set()
        for row in payload.get("models", []) or []:
            name = str(row.get("name") or "").strip()
            if name:
                models.add(name)
        return models

    def ensure_model_present(self, model: str):
        model_name = str(model or "").strip() or self.config.get_ollama_model_default()
        if model_name in self.available_models():
            return
        logger.info(f"Ollama model {model_name} missing locally. Pulling it now.")
        response = self._post(
            "/api/pull",
            json_body={"name": model_name, "stream": True},
            stream=True,
        )
        try:
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                try:
                    payload = json.loads(raw_line.decode("utf-8", errors="ignore"))
                except Exception:
                    continue
                status = str(payload.get("status") or "").strip()
                if status:
                    logger.info(f"Ollama pull [{model_name}]: {status}")
        finally:
            response.close()

    def unload_all_models(self):
        models = self.available_models()
        if not models:
            return
        try:
            response = requests.get(
                f"{self._base_url()}/api/ps",
                timeout=min(self._timeout(), 30),
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning(f"Failed to list active Ollama models for unload: {exc}")
            return
        for row in payload.get("models", []) or []:
            model_name = str(row.get("name") or "").strip()
            if not model_name:
                continue
            try:
                self._post(
                    "/api/generate",
                    json_body={
                        "model": model_name,
                        "prompt": "",
                        "stream": False,
                        "keep_alive": 0,
                    },
                ).close()
                logger.info(f"Unloaded Ollama model {model_name} before diffusion work.")
            except Exception as exc:
                logger.warning(f"Failed unloading Ollama model {model_name}: {exc}")

    def prepare_for_diffusion(self):
        self.unload_all_models()

    def prepare_for_ollama(self):
        pipeline_manager = AppConfig.get_pipeline_manager()
        if pipeline_manager is None:
            return
        try:
            pipeline_manager.delete_pipes()
        except Exception as exc:
            logger.warning(f"Failed deleting diffusers pipelines before Ollama call: {exc}")

    def complete(
        self,
        *,
        role: str,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        keep_alive: str | None = None,
    ) -> str:
        self.prepare_for_ollama()
        model_name = str(model or "").strip() or self.config.get_ollama_model_default()
        self.ensure_model_present(model_name)
        keep_alive_value = str(keep_alive or self.config.get_ollama_keep_alive() or "30m")
        messages = []
        system_text = str(role or "").strip()
        user_text = str(prompt or "").strip()
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user_text})
        response = self._post(
            "/api/chat",
            json_body={
                "model": model_name,
                "stream": False,
                "keep_alive": keep_alive_value,
                "messages": messages,
                "options": {
                    "temperature": float(temperature),
                    "num_predict": int(max_tokens),
                },
            },
        )
        payload = response.json()
        text = str(
            payload.get("message", {}).get("content")
            or payload.get("response")
            or ""
        ).strip()
        if not text:
            raise RuntimeError("Local Ollama returned empty content.")
        return text
