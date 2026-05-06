from __future__ import annotations

import asyncio
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

from xoxxox.irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    default_runtime_device,
    resolve_cfg_scales,
    save_wav,
)


FIXED_SECONDS = 5.0


def _parse_optional_float(value: str | float | None) -> float | None:
    if value is None:
        return None

    raw = str(value).strip().lower()
    if raw in {"none", "null", "off", "disable", "disabled"}:
        return None

    out = float(raw)
    if not math.isfinite(out):
        raise ValueError(f"Expected finite float for value={value!r}.")
    return out


def _resolve_checkpoint_path(
    *,
    checkpoint: str | None,
    hf_checkpoint: str | None,
) -> str:
    if checkpoint is not None:
        checkpoint_path = Path(str(checkpoint)).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"[checkpoint] using local file: {checkpoint_path}", flush=True)
        return str(checkpoint_path)

    if hf_checkpoint is None:
        raise ValueError("Either checkpoint or hf_checkpoint is required.")

    repo_id = str(hf_checkpoint).strip()
    if repo_id == "":
        raise ValueError("hf_checkpoint must be non-empty.")

    checkpoint_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.safetensors",
    )
    print(
        f"[checkpoint] downloaded model.safetensors from hf://{repo_id} -> {checkpoint_path}",
        flush=True,
    )
    return str(checkpoint_path)


@dataclass
class IrodoriTTS:
    runtime: InferenceRuntime
    caption: str | None
    ref_wav: str | None
    ref_latent: str | None
    no_ref: bool
    ref_normalize_db: float | None
    ref_ensure_max: bool
    max_ref_seconds: float | None
    max_text_len: int | None
    max_caption_len: int | None
    num_steps: int
    num_candidates: int
    decode_mode: str
    cfg_scale_text: float
    cfg_scale_caption: float
    cfg_scale_speaker: float
    cfg_guidance_mode: str
    cfg_min_t: float
    cfg_max_t: float
    truncation_factor: float | None
    rescale_k: float | None
    rescale_sigma: float | None
    context_kv_cache: bool
    speaker_kv_scale: float | None
    speaker_kv_min_t: float | None
    speaker_kv_max_layers: int | None
    seed: int | None
    trim_tail: bool
    tail_window_size: int
    tail_std_threshold: float
    tail_mean_threshold: float
    tmp_dir: Path
    lock: asyncio.Lock


def init_tts(
    *,
    checkpoint: str | None = None,
    hf_checkpoint: str | None = "Aratako/Irodori-TTS-500M-v2",
    ref_wav: str | None = None,
    ref_latent: str | None = None,
    no_ref: bool = False,
    caption: str | None = None,
    model_device: str = default_runtime_device(),
    model_precision: str = "fp32",
    codec_device: str = default_runtime_device(),
    codec_precision: str = "fp32",
    codec_repo: str = "Aratako/Semantic-DACVAE-Japanese-32dim",
    codec_deterministic_encode: bool = True,
    codec_deterministic_decode: bool = True,
    enable_watermark: bool = False,
    compile_model: bool = False,
    compile_dynamic: bool = False,
    max_ref_seconds: float | None = 30.0,
    ref_normalize_db: str | float | None = -16.0,
    ref_ensure_max: bool = True,
    max_text_len: int | None = None,
    max_caption_len: int | None = None,
    num_steps: int = 40,
    num_candidates: int = 1,
    decode_mode: str = "sequential",
    cfg_scale_text: float = 3.0,
    cfg_scale_caption: float = 3.0,
    cfg_scale_speaker: float = 5.0,
    cfg_guidance_mode: str = "independent",
    cfg_scale: float | None = None,
    cfg_min_t: float = 0.5,
    cfg_max_t: float = 1.0,
    truncation_factor: float | None = None,
    rescale_k: float | None = None,
    rescale_sigma: float | None = None,
    context_kv_cache: bool = True,
    speaker_kv_scale: float | None = None,
    speaker_kv_min_t: float = 0.9,
    speaker_kv_max_layers: int | None = None,
    seed: int | None = None,
    trim_tail: bool = True,
    tail_window_size: int = 20,
    tail_std_threshold: float = 0.05,
    tail_mean_threshold: float = 0.1,
    tmp_dir: str = "/dev/shm",
) -> IrodoriTTS:
    if isinstance(ref_wav, str) and ref_wav.strip() == "":
        ref_wav = None
        no_ref = True

    checkpoint_path = _resolve_checkpoint_path(
        checkpoint=checkpoint,
        hf_checkpoint=hf_checkpoint,
    )

    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=checkpoint_path,
            model_device=str(model_device),
            codec_repo=str(codec_repo),
            model_precision=str(model_precision),
            codec_device=str(codec_device),
            codec_precision=str(codec_precision),
            codec_deterministic_encode=bool(codec_deterministic_encode),
            codec_deterministic_decode=bool(codec_deterministic_decode),
            enable_watermark=bool(enable_watermark),
            compile_model=bool(compile_model),
            compile_dynamic=bool(compile_dynamic),
        )
    )

    if runtime.model_cfg.use_speaker_condition and not (
        no_ref or ref_wav is not None or ref_latent is not None
    ):
        raise ValueError(
            "speaker-conditioned checkpoints require one of ref_wav, ref_latent, or no_ref."
        )

    cfg_scale_text, cfg_scale_caption, cfg_scale_speaker, scale_messages = (
        resolve_cfg_scales(
            cfg_guidance_mode=str(cfg_guidance_mode),
            cfg_scale_text=float(cfg_scale_text),
            cfg_scale_caption=float(cfg_scale_caption),
            cfg_scale_speaker=float(cfg_scale_speaker),
            cfg_scale=float(cfg_scale) if cfg_scale is not None else None,
            use_caption_condition=bool(
                runtime.model_cfg.use_caption_condition
                and caption is not None
                and str(caption).strip() != ""
            ),
            use_speaker_condition=bool(runtime.model_cfg.use_speaker_condition),
        )
    )
    for msg in scale_messages:
        print(msg)

    return IrodoriTTS(
        runtime=runtime,
        caption=None if caption is None else str(caption),
        ref_wav=None if ref_wav is None else str(ref_wav),
        ref_latent=None if ref_latent is None else str(ref_latent),
        no_ref=bool(no_ref),
        ref_normalize_db=_parse_optional_float(ref_normalize_db),
        ref_ensure_max=bool(ref_ensure_max),
        max_ref_seconds=max_ref_seconds,
        max_text_len=max_text_len,
        max_caption_len=max_caption_len,
        num_steps=int(num_steps),
        num_candidates=int(num_candidates),
        decode_mode=str(decode_mode),
        cfg_scale_text=cfg_scale_text,
        cfg_scale_caption=cfg_scale_caption,
        cfg_scale_speaker=cfg_scale_speaker,
        cfg_guidance_mode=str(cfg_guidance_mode),
        cfg_min_t=float(cfg_min_t),
        cfg_max_t=float(cfg_max_t),
        truncation_factor=None
        if truncation_factor is None
        else float(truncation_factor),
        rescale_k=None if rescale_k is None else float(rescale_k),
        rescale_sigma=None if rescale_sigma is None else float(rescale_sigma),
        context_kv_cache=bool(context_kv_cache),
        speaker_kv_scale=None if speaker_kv_scale is None else float(speaker_kv_scale),
        speaker_kv_min_t=None
        if speaker_kv_scale is None
        else float(speaker_kv_min_t),
        speaker_kv_max_layers=None
        if speaker_kv_max_layers is None
        else int(speaker_kv_max_layers),
        seed=None if seed is None else int(seed),
        trim_tail=bool(trim_tail),
        tail_window_size=int(tail_window_size),
        tail_std_threshold=float(tail_std_threshold),
        tail_mean_threshold=float(tail_mean_threshold),
        tmp_dir=Path(tmp_dir),
        lock=asyncio.Lock(),
    )


async def infer_tts(tts: IrodoriTTS, text: str) -> bytes:
    if not text or not text.strip():
        raise ValueError("text must not be empty.")

    print("ref_wav[" + str(tts.ref_wav) + "]", flush=True) # DBG
    print("tts.caption[" + str(tts.caption) + "]", flush=True) # DBG
    print("text[" + str(text) + "]", flush=True) # DBG
    print("tts.cfg_scale_speaker[" + str(tts.cfg_scale_speaker) + "]", flush=True) # DBG
    print("tts.cfg_scale_caption[" + str(tts.cfg_scale_caption) + "]", flush=True) # DBG
    print("tts.cfg_scale_text[" + str(tts.cfg_scale_text) + "]", flush=True) # DBG

    async with tts.lock:
        result = await asyncio.to_thread(
            tts.runtime.synthesize,
            SamplingRequest(
                text=str(text),
                caption=tts.caption,
                ref_wav=tts.ref_wav,
                ref_latent=tts.ref_latent,
                no_ref=tts.no_ref,
                ref_normalize_db=tts.ref_normalize_db,
                ref_ensure_max=tts.ref_ensure_max,
                num_candidates=tts.num_candidates,
                decode_mode=tts.decode_mode,
                seconds=FIXED_SECONDS,
                max_ref_seconds=tts.max_ref_seconds,
                max_text_len=tts.max_text_len,
                max_caption_len=tts.max_caption_len,
                num_steps=tts.num_steps,
                cfg_scale_text=tts.cfg_scale_text,
                cfg_scale_caption=tts.cfg_scale_caption,
                cfg_scale_speaker=tts.cfg_scale_speaker,
                cfg_guidance_mode=tts.cfg_guidance_mode,
                cfg_scale=None,
                cfg_min_t=tts.cfg_min_t,
                cfg_max_t=tts.cfg_max_t,
                truncation_factor=tts.truncation_factor,
                rescale_k=tts.rescale_k,
                rescale_sigma=tts.rescale_sigma,
                context_kv_cache=tts.context_kv_cache,
                speaker_kv_scale=tts.speaker_kv_scale,
                speaker_kv_min_t=tts.speaker_kv_min_t,
                speaker_kv_max_layers=tts.speaker_kv_max_layers,
                seed=tts.seed,
                trim_tail=tts.trim_tail,
                tail_window_size=tts.tail_window_size,
                tail_std_threshold=tts.tail_std_threshold,
                tail_mean_threshold=tts.tail_mean_threshold,
            ),
            log_fn=None,
        )

    if tts.num_candidates != 1:
        raise ValueError("infer_tts currently returns bytes for num_candidates=1 only.")

    fd, output_path = tempfile.mkstemp(
        suffix=".wav",
        prefix="irodori_tts_",
        dir=str(tts.tmp_dir),
    )
    os.close(fd)

    output = Path(output_path)
    try:
        save_wav(output, result.audio, result.sample_rate)
        return output.read_bytes()
    finally:
        output.unlink(missing_ok=True)
