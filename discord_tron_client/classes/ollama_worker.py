import asyncio
import logging

from discord_tron_client.classes.app_config import AppConfig
from discord_tron_client.classes.hardware import HardwareInfo
from discord_tron_client.classes.message import WebsocketMessage

logger = logging.getLogger(__name__)


class OllamaWorker:
    def __init__(self):
        self.config = AppConfig()

    def complete(self, payload):
        runtime = AppConfig.get_ollama_runtime()
        return runtime.complete(
            role=str(payload.get("role") or ""),
            prompt=str(payload.get("prompt") or ""),
            model=str(payload.get("model") or "").strip() or None,
            temperature=float(payload.get("temperature") or 0.7),
            max_tokens=int(payload.get("max_tokens") or 2048),
            keep_alive=str(payload.get("keep_alive") or "").strip() or None,
        )

    async def complete_handler(self, payload, websocket):
        request_id = str(payload.get("request_id") or "").strip()
        job_id = str(payload.get("job_id") or "").strip()
        worker_id = HardwareInfo.get_identifier()
        result_payload = {
            "request_id": request_id,
            "ok": False,
            "text": "",
            "detail": "",
            "worker_id": worker_id,
        }
        try:
            loop = asyncio.get_event_loop()
            result_payload["text"] = await loop.run_in_executor(
                AppConfig.get_image_worker_thread(),
                self.complete,
                payload,
            )
            result_payload["ok"] = True
        except Exception as exc:
            logger.error(f"Ollama worker completion failed: {exc}")
            result_payload["detail"] = str(exc)
        message = WebsocketMessage(
            message_type="ollama_result",
            module_name="ollama",
            module_command="complete_result",
            data=result_payload,
            arguments={
                "worker_id": worker_id,
                "job_id": job_id,
                "request_id": request_id,
            },
        )
        websocket = AppConfig.get_websocket()
        await websocket.send(message.to_json())
