"""Local Whisper ASR streaming client, ALSA capture, and optional Edge Impulse KWS wake word."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import queue
import subprocess
import threading
import time
from collections import deque
from typing import Any, Callable, Iterator

import httpx
import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK

logger = logging.getLogger(__name__)


def build_asr_create_payload(
    model: str = "whisper-small-quantized",
    sample_rate: int = 16000,
    channels: int = 1,
    vad_ms: int = 700,
    language: str | None = None,
    translate: bool = False,
) -> dict[str, Any]:
    """Build the HTTP POST body for audio-analytics `/transcriptions/create`."""
    payload: dict[str, Any] = {
        "model": model,
        "stream": True,
        "translate": translate,
        "parameters": json.dumps([
            {"key": "sampling_rate", "value": str(sample_rate)},
            {"key": "channels", "value": str(channels)},
            {"key": "format", "value": "pcm_s16le"},
            {"key": "vad", "value": str(vad_ms)},
        ]),
    }
    if language and language.strip():
        payload["language"] = language.strip()
    return payload


def build_asr_audio_message(session_id: str, pcm_bytes: bytes) -> dict[str, Any]:
    """Build one PCM audio frame message for `ws://audio-analytics-runner:8085/stream`."""
    return {
        "message_type": "transcriptions_session_audio",
        "message_source": "audio_analytics_api",
        "session_id": session_id,
        "type": "input_audio",
        "data": base64.b64encode(pcm_bytes).decode("utf-8"),
    }


def parse_asr_ws_event(raw_message: str | dict[str, Any], expected_session_id: str) -> tuple[str, str] | None:
    """Parse one WebSocket JSON event from audio-analytics-runner into `(kind, payload)`."""
    if isinstance(raw_message, str):
        try:
            data = json.loads(raw_message)
        except json.JSONDecodeError:
            logger.warning("Failed to parse ASR WebSocket JSON: %s", raw_message)
            return None
    else:
        data = raw_message

    message_session_id = data.get("session_id")
    if message_session_id is not None and message_session_id != expected_session_id:
        return None

    evt_type = data.get("type") or data.get("message_type")
    evt_state = data.get("state")
    evt_text = str(data.get("text", "")).strip()

    if evt_state == "connection_established":
        return ("established", "")
    if evt_type == "transcript.text.delta":
        return ("partial", evt_text)
    if evt_type == "transcript.text.done":
        return ("done", evt_text)
    if evt_type == "transcript.event":
        return (str(evt_state or "event"), evt_text)
    if evt_type == "error":
        return ("error", str(data.get("message") or data.get("error") or data))
    return None


def extract_kws_match(
    result: dict[str, Any] | None,
    target_label: str = "hey_arduino",
    confidence: float = 0.80,
) -> tuple[str, float] | None:
    """Check Edge Impulse `/api/features` JSON for a keyword match above `confidence`."""
    if not result:
        return None
    classification = result.get("result", {}).get("classification")
    if not isinstance(classification, dict):
        return None

    norm_target = target_label.strip().lower().replace(" ", "_")
    best_label: str | None = None
    best_score = 0.0
    for label, raw_score in classification.items():
        score = float(raw_score)
        norm_label = str(label).strip().lower().replace(" ", "_")
        if norm_target != "*" and norm_label != norm_target:
            continue
        if score >= confidence and score > best_score:
            best_label = str(label)
            best_score = score
    return (best_label, best_score) if best_label is not None else None


class AlsaMicCapture:
    """Continuous `arecord` raw `S16_LE` PCM capture subprocess."""

    def __init__(
        self,
        device: str = "plughw:CARD=Array,DEV=0",
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_ms: int = 100,
    ) -> None:
        self.device = device.strip()
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_bytes = int(sample_rate * chunk_ms / 1000) * channels * 2
        self._proc: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        cmd = ["arecord", "-q"]
        if self.device:
            cmd.extend(["-D", self.device])
        cmd.extend([
            "-f", "S16_LE",
            "-r", str(self.sample_rate),
            "-c", str(self.channels),
            "-t", "raw",
        ])
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def read_chunk(self) -> bytes:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("AlsaMicCapture has not been started.")
        data = self._proc.stdout.read(self.chunk_bytes)
        if not data:
            stderr = b""
            if self._proc.stderr is not None:
                stderr = self._proc.stderr.read()
            raise RuntimeError(f"arecord stream ended unexpectedly: {stderr.decode(errors='replace').strip()}")
        return data

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        if proc.stdout:
            proc.stdout.close()
        if proc.stderr:
            proc.stderr.close()

    def __enter__(self) -> AlsaMicCapture:
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop()


class WakeWordDetector:
    """Optional Edge Impulse HTTP KWS client (`ei-keyword-spot-runner:1337`)."""

    def __init__(self, voice_cfg: dict[str, Any]) -> None:
        self.kws_api_url = str(voice_cfg.get("kws_api_url", "http://ei-keyword-spot-runner:1337")).rstrip("/")
        self.target_label = str(voice_cfg.get("kws_label", "hey_arduino"))
        self.confidence = float(voice_cfg.get("kws_confidence", 0.80))
        self.capture_device = str(voice_cfg.get("capture_device", "plughw:CARD=Array,DEV=0"))
        self.sample_rate = int(voice_cfg.get("sample_rate", 16000))
        self.channels = int(voice_cfg.get("channels", 1))
        self.debounce_s = float(voice_cfg.get("kws_debounce_s", 2.0))

    def wait_for_trigger(
        self,
        is_muted: Callable[[], bool],
        stop_event: threading.Event,
    ) -> bool:
        """Listen on ALSA mic until the wake word is spotted or `stop_event` is set."""
        with httpx.Client(timeout=5.0) as client:
            info_resp = client.get(f"{self.kws_api_url}/api/info")
            info_resp.raise_for_status()
            params = info_resp.json().get("modelParameters", {})
            window_samples = int(params.get("input_features_count", self.sample_rate))
            slide_samples = max(1600, int(window_samples * 0.4))
            ring: deque[int] = deque(maxlen=window_samples)
            new_samples = 0

            with AlsaMicCapture(
                device=self.capture_device,
                sample_rate=self.sample_rate,
                channels=self.channels,
                chunk_ms=100,
            ) as mic:
                last_trigger = 0.0
                while not stop_event.is_set():
                    pcm = mic.read_chunk()
                    if is_muted():
                        ring.clear()
                        new_samples = 0
                        continue
                    samples = np.frombuffer(pcm, dtype=np.int16).tolist()
                    ring.extend(samples)
                    new_samples += len(samples)
                    if len(ring) < window_samples or new_samples < slide_samples:
                        continue
                    new_samples = 0

                    resp = client.post(f"{self.kws_api_url}/api/features", json={"features": list(ring)})
                    if resp.status_code != 200:
                        continue
                    match = extract_kws_match(resp.json(), self.target_label, self.confidence)
                    now = time.monotonic()
                    if match and (now - last_trigger) >= self.debounce_s:
                        logger.info("Wake word detected: %s (%.2f)", match[0], match[1])
                        return True
        return False


class WhisperASRClient:
    """Streaming client for `audio-analytics-runner` (`whisper-small-quantized`)."""

    def __init__(self, voice_cfg: dict[str, Any]) -> None:
        self.asr_api_url = str(
            voice_cfg.get("asr_api_url", "http://audio-analytics-runner:8085/audio-analytics/v1/api")
        ).rstrip("/")
        self.asr_ws_url = str(voice_cfg.get("asr_ws_url", "ws://audio-analytics-runner:8085/stream"))
        self.model = str(voice_cfg.get("asr_model", "whisper-small-quantized"))
        self.language = str(voice_cfg.get("asr_language", "")).strip() or None
        self.vad_ms = int(voice_cfg.get("asr_vad_ms", 700))
        self.capture_device = str(voice_cfg.get("capture_device", "plughw:CARD=Array,DEV=0"))
        self.sample_rate = int(voice_cfg.get("sample_rate", 16000))
        self.channels = int(voice_cfg.get("channels", 1))

    def _create_session(self) -> str:
        payload = build_asr_create_payload(
            model=self.model,
            sample_rate=self.sample_rate,
            channels=self.channels,
            vad_ms=self.vad_ms,
            language=self.language,
        )
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(f"{self.asr_api_url}/transcriptions/create", json=payload)
            resp.raise_for_status()
            data = resp.json()
        session_id = data.get("session_id")
        if not session_id or data.get("state") != "asr_initialized":
            raise RuntimeError(f"Unexpected ASR session response: {data}")
        return str(session_id)

    def _flush_session(self, session_id: str) -> None:
        try:
            with httpx.Client(timeout=3.0) as client:
                client.post(f"{self.asr_api_url}/transcriptions/flush", json={"session_id": session_id})
        except Exception as exc:
            logger.debug("ASR flush warning for %s: %s", session_id, exc)

    def _close_session(self, session_id: str) -> None:
        try:
            with httpx.Client(timeout=10.0) as client:
                client.post(f"{self.asr_api_url}/transcriptions/close", json={"session_id": session_id})
        except Exception as exc:
            logger.debug("ASR close warning for %s: %s", session_id, exc)

    async def _run_session(
        self,
        is_muted: Callable[[], bool],
        stop_event: threading.Event,
        out_queue: queue.Queue[str],
        max_duration_s: float | None = None,
    ) -> None:
        session_id = await asyncio.to_thread(self._create_session)
        pcm_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=40)
        muted_state = {"speech_while_muted": False, "cooldown_until": 0.0}
        session_stop = threading.Event()
        deadline = (time.monotonic() + max_duration_s) if max_duration_s else None

        def _mic_reader() -> None:
            try:
                with AlsaMicCapture(
                    device=self.capture_device,
                    sample_rate=self.sample_rate,
                    channels=self.channels,
                    chunk_ms=100,
                ) as mic:
                    while not stop_event.is_set() and not session_stop.is_set():
                        chunk = mic.read_chunk()
                        if is_muted():
                            muted_state["speech_while_muted"] = True
                            muted_state["cooldown_until"] = time.monotonic() + 0.4
                            chunk = b"\x00" * len(chunk)
                        try:
                            pcm_queue.put_nowait(chunk)
                        except queue.Full:
                            pass
            except Exception as exc:
                logger.warning("Microphone reader exited: %s", exc)
            finally:
                try:
                    pcm_queue.put_nowait(None)
                except queue.Full:
                    pass

        reader_thread = threading.Thread(target=_mic_reader, daemon=True, name=f"ASRMic-{session_id}")
        reader_thread.start()

        async def _await_established(ws: Any) -> None:
            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            evt = parse_asr_ws_event(raw, session_id)
            if not evt or evt[0] != "established":
                raise RuntimeError(f"Expected connection_established, got: {raw}")

        async def _send_loop(ws: Any) -> None:
            while not stop_event.is_set() and not session_stop.is_set():
                if deadline is not None and time.monotonic() >= deadline:
                    return
                try:
                    chunk = await asyncio.to_thread(pcm_queue.get, True, 0.2)
                except queue.Empty:
                    continue
                if chunk is None:
                    return
                msg = build_asr_audio_message(session_id, chunk)
                await ws.send(json.dumps(msg))

        async def _recv_loop(ws: Any) -> None:
            while not stop_event.is_set() and not session_stop.is_set():
                if deadline is not None and time.monotonic() >= deadline:
                    return
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                except TimeoutError:
                    continue
                evt = parse_asr_ws_event(raw, session_id)
                if not evt:
                    continue
                kind, payload = evt
                if kind == "speech_start":
                    muted_state["speech_while_muted"] = is_muted()
                elif kind == "done":
                    drop = (
                        is_muted()
                        or muted_state["speech_while_muted"]
                        or time.monotonic() < muted_state["cooldown_until"]
                    )
                    muted_state["speech_while_muted"] = False
                    if payload and not drop:
                        out_queue.put(payload)
                elif kind == "error":
                    raise RuntimeError(f"ASR service error: {payload}")

        async def _drain_loop(ws: Any) -> None:
            while not stop_event.is_set() and not session_stop.is_set():
                try:
                    await asyncio.wait_for(ws.recv(), timeout=0.5)
                except TimeoutError:
                    continue

        async def _flush_loop() -> None:
            while not stop_event.is_set() and not session_stop.is_set():
                await asyncio.sleep(10.0)
                if not stop_event.is_set() and not session_stop.is_set():
                    await asyncio.to_thread(self._flush_session, session_id)

        try:
            async with (
                websockets.connect(self.asr_ws_url, ping_interval=10, ping_timeout=5, close_timeout=5) as write_ws,
                websockets.connect(self.asr_ws_url, ping_interval=10, ping_timeout=5, close_timeout=5) as read_ws,
            ):
                await _await_established(write_ws)
                await _await_established(read_ws)
                tasks = [
                    asyncio.create_task(_send_loop(write_ws)),
                    asyncio.create_task(_recv_loop(read_ws)),
                    asyncio.create_task(_drain_loop(write_ws)),
                ]
                flush_task = asyncio.create_task(_flush_loop())
                try:
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    for task in done:
                        exc = task.exception()
                        if exc and not isinstance(exc, (ConnectionClosedOK, asyncio.CancelledError)):
                            raise exc
                finally:
                    flush_task.cancel()
                    await asyncio.gather(flush_task, return_exceptions=True)
        except (ConnectionClosedOK, ConnectionClosed) as exc:
            logger.info("ASR WebSocket closed (%s); will reopen if active.", exc)
        finally:
            session_stop.set()
            reader_thread.join(timeout=2.0)
            await asyncio.to_thread(self._close_session, session_id)

    def listen_once(
        self,
        is_muted: Callable[[], bool],
        stop_event: threading.Event,
        timeout_s: float = 12.0,
    ) -> str | None:
        """Capture one spoken utterance after a wake-word trigger."""
        out_queue: queue.Queue[str] = queue.Queue()
        local_stop = threading.Event()

        def _worker() -> None:
            try:
                asyncio.run(self._run_session(is_muted, local_stop, out_queue, max_duration_s=timeout_s))
            except Exception as exc:
                logger.warning("ASR single-utterance session failed: %s", exc)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        deadline = time.monotonic() + timeout_s
        try:
            while not stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                try:
                    return out_queue.get(timeout=min(0.25, remaining))
                except queue.Empty:
                    if not t.is_alive():
                        return out_queue.get_nowait() if not out_queue.empty() else None
        finally:
            local_stop.set()
            t.join(timeout=3.0)
        return None

    def iter_utterances(
        self,
        is_muted: Callable[[], bool],
        stop_event: threading.Event | None = None,
    ) -> Iterator[str]:
        """Continuously yield completed ASR utterances, reconnecting cleanly on timeout."""
        stop = stop_event or threading.Event()
        out_queue: queue.Queue[str] = queue.Queue()

        def _runner() -> None:
            while not stop.is_set():
                try:
                    asyncio.run(self._run_session(is_muted, stop, out_queue))
                except Exception as exc:
                    logger.warning("ASR session error (%s); retrying in 1.0s...", exc)
                if not stop.is_set():
                    time.sleep(0.5)

        thread = threading.Thread(target=_runner, daemon=True, name="WhisperASRRunner")
        thread.start()
        try:
            while not stop.is_set():
                try:
                    yield out_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
        finally:
            stop.set()
            thread.join(timeout=3.0)
