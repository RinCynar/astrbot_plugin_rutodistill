import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional
from astrbot.api import logger

try:
    from astrbot.core.utils.astrbot_path import get_astrbot_data_path
except ImportError:
    def get_astrbot_data_path() -> str:
        return "data"


class JSONStore:
    def __init__(self, plugin_name: str = "astrbot_plugin_rutodistill"):
        self.plugin_name = plugin_name
        self.lock = asyncio.Lock()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.base_dir = self._get_storage_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_storage_dir(self) -> Path:
        try:
            root = Path(get_astrbot_data_path())
        except Exception:
            root = Path("data")
        return root / "plugin_data" / self.plugin_name

    def _get_file_path(self, session_id: str) -> Path:
        # Safe filename sanitization
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in session_id)
        return self.base_dir / f"{safe_id}.json"

    async def get_session(self, session_id: str, default_factory: Optional[callable] = None) -> Dict[str, Any]:
        async with self.lock:
            if session_id in self._cache:
                return self._cache[session_id]

            file_path = self._get_file_path(session_id)
            if file_path.exists():
                try:
                    def _read():
                        with open(file_path, "r", encoding="utf-8") as f:
                            return json.load(f)
                    data = await asyncio.to_thread(_read)
                    self._cache[session_id] = data
                    return data
                except Exception as e:
                    logger.error(f"[rutodistill] Failed to read storage file {file_path}: {e}")

            default_data = default_factory() if default_factory else {}
            self._cache[session_id] = default_data
            return default_data

    async def save_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        async with self.lock:
            self._cache[session_id] = data
            file_path = self._get_file_path(session_id)
            tmp_path = file_path.with_suffix(".tmp")

            def _atomic_write():
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                tmp_path.replace(file_path)

            try:
                await asyncio.to_thread(_atomic_write)
                return True
            except Exception as e:
                logger.error(f"[rutodistill] Failed to save session {session_id} to {file_path}: {e}")
                return False

    async def flush_all(self):
        async with self.lock:
            for session_id, data in list(self._cache.items()):
                file_path = self._get_file_path(session_id)
                tmp_path = file_path.with_suffix(".tmp")
                try:
                    def _write(d=data, tmp=tmp_path, final=file_path):
                        with open(tmp, "w", encoding="utf-8") as f:
                            json.dump(d, f, ensure_ascii=False, indent=2)
                        tmp.replace(final)
                    await asyncio.to_thread(_write)
                except Exception as e:
                    logger.error(f"[rutodistill] Flush error for {session_id}: {e}")
