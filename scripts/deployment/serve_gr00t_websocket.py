#!/usr/bin/env python3
"""
GR00T N1.6 WebSocket inference service (msgpack).

Usage:
  uv run --extra websocket python scripts/deployment/serve_gr00t_websocket.py \\
    --model-path /path/to/checkpoint --embodiment-tag AGIBOT_GENIE1 --port 8000

Dependencies: `uv sync --extra websocket` (or `pip install websockets`); the project already includes msgpack / msgpack-numpy.

**Payload format (Agibot three cameras + flattened state)**

- ``images``: ``top_head`` / ``hand_left`` / ``hand_right`` — scale with aspect ratio preserved + pad to 256×256.
- ``state``: flattened list / ndarray, layout:
  left arm 7 | right arm 7 | left gripper 1 | right gripper 1 | ... | waist (index 20) 1 (when using AGIBOT_GENIE1_WAIST).
- ``prompt`` / ``task_name``: language instruction.

Response:
  - ``actions``: one command vector per timestep. ``agibot_genie1`` is **16-D**
    (0–7 left arm, 7–14 right arm, 14 left gripper, 15 right gripper);
    ``agibot_genie1_waist`` is **17-D** (16-D + 1 waist dimension).
  - ``actions_by_key``: actions split by joint key names.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import socket
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro
from PIL import Image

root_dir = Path(__file__).parent.parent.parent
import sys
sys.path.append(str(root_dir))

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.launch_finetune import load_modality_config
from gr00t.policy.gr00t_policy import Gr00tPolicy

logger = logging.getLogger(__name__)

TARGET_VIDEO_SIZE = (256, 256)

FLAT_STATE_SLICES: dict[str, list[tuple[str, slice]]] = {
    EmbodimentTag.AGIBOT_GENIE1.value: [
        ("left_arm_joint_position", slice(0, 7)),
        ("right_arm_joint_position", slice(7, 14)),
        ("left_effector_position", slice(14, 15)),
        ("right_effector_position", slice(15, 16)),
    ],
    EmbodimentTag.AGIBOT_GENIE1_WAIST.value: [
        ("left_arm_joint_position", slice(0, 7)),
        ("right_arm_joint_position", slice(7, 14)),
        ("left_effector_position", slice(14, 15)),
        ("right_effector_position", slice(15, 16)),
        ("waist_position", slice(20, 21)),
    ],
}

IMAGE_KEYS = ("top_head", "hand_left", "hand_right")

WAIST_TASK_NAMES = {"sorting_packages"}


def _normalize_embodiment_tag(tag: EmbodimentTag | str) -> EmbodimentTag:
    if isinstance(tag, EmbodimentTag):
        return tag
    try:
        return EmbodimentTag[tag]
    except KeyError:
        return EmbodimentTag(tag)


def resize_keep_aspect(frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Scale with aspect ratio preserved and center-pad to (target_h, target_w)."""
    h, w = frame.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    pil = Image.fromarray(frame)
    pil = pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
    arr = np.asarray(pil, dtype=np.uint8)
    pad_h = target_h - arr.shape[0]
    pad_w = target_w - arr.shape[1]
    pad_top = pad_h // 2
    pad_left = pad_w // 2
    pad_bottom = pad_h - pad_top
    pad_right = pad_w - pad_left
    out = np.pad(
        arr,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    return out.astype(np.uint8)


def _ensure_ndarray(x: Any, dtype: np.dtype | None = None) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x
    else:
        arr = np.array(x)
    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype)
    return arr


def _decode_maybe_b64_ndarray(img: Any) -> np.ndarray:
    if isinstance(img, str):
        buf = base64.b64decode(img)
        return np.load(io.BytesIO(buf), allow_pickle=False)
    return _ensure_ndarray(img)


def _prepare_frame_hwc_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    elif img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _video_ntc_from_array(arr: np.ndarray) -> np.ndarray:
    """Return (T,H,W,C) uint8."""
    if arr.ndim == 3:
        return arr[np.newaxis, ...]
    if arr.ndim == 4:
        return arr
    raise ValueError(f"Video array must be (H,W,3) or (T,H,W,3), got {arr.shape}")


def _ensure_video_thwc_u8(arr: Any) -> np.ndarray:
    """Normalize incoming video array to (T,H,W,C) uint8 (channels-last)."""
    x = _ensure_ndarray(arr)
    if x.ndim == 2:
        # (H,W) grayscale -> (H,W,3)
        x = _prepare_frame_hwc_u8(x)
        return x[np.newaxis, ...]
    if x.ndim == 3:
        # (H,W,3) or (3,H,W)
        frame = _prepare_frame_hwc_u8(x)
        return frame[np.newaxis, ...]
    if x.ndim != 4:
        raise ValueError(f"Video must be (H,W,3) or (T,H,W,3) or (T,3,H,W), got {x.shape}")

    # (T,H,W,C) or (T,C,H,W)
    if x.shape[-1] in (3, 4):
        frame = x
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
    elif x.shape[1] == 3:
        # T,C,H,W -> T,H,W,C
        frame = np.transpose(x, (0, 2, 3, 1))
    else:
        raise ValueError(f"Unrecognized video layout {x.shape}, expected channels-last or channels-first")

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def resize_video_keep_aspect(seq_thwc_u8: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize each frame with keep-aspect + centered padding."""
    out_frames = []
    for t in range(seq_thwc_u8.shape[0]):
        out_frames.append(resize_keep_aspect(seq_thwc_u8[t], target_h, target_w))
    return np.stack(out_frames, axis=0).astype(np.uint8)


def images_to_video_payload(images: dict[str, Any]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key in IMAGE_KEYS:
        if key not in images:
            raise ValueError(f"Payload missing images.{key}")
        img = _decode_maybe_b64_ndarray(images[key])
        if isinstance(img, list):
            img = np.array(img, dtype=np.uint8)
        img = _prepare_frame_hwc_u8(img)
        img = resize_keep_aspect(img, TARGET_VIDEO_SIZE[0], TARGET_VIDEO_SIZE[1])
        out[key] = _video_ntc_from_array(img)
    return out


def flat_state_to_dict(
    flat: list | np.ndarray,
    embodiment_tag: EmbodimentTag,
) -> dict[str, np.ndarray]:
    slices = FLAT_STATE_SLICES[embodiment_tag.value]
    max_idx = max(sl.stop for _, sl in slices)
    arr = _ensure_ndarray(flat, dtype=np.float32).reshape(-1)
    if arr.size < max_idx:
        raise ValueError(f"Flat state needs at least {max_idx} dims, got {arr.size}")
    return {key: arr[sl].astype(np.float32).reshape(1, -1) for key, sl in slices}


def payload_to_observation(policy: Gr00tPolicy, payload: dict[str, Any]) -> dict[str, Any]:
    modality_configs = policy.get_modality_config()
    lang_keys = modality_configs["language"].modality_keys
    lang_key = lang_keys[0]
    prompt = payload.get("prompt") or payload.get("task_name") or ""
    if isinstance(prompt, (bytes, bytearray)):
        prompt = prompt.decode("utf-8", errors="replace")

    if "images" not in payload:
        raise ValueError("Payload must contain 'images' key")
    if policy.embodiment_tag not in (EmbodimentTag.AGIBOT_GENIE1, EmbodimentTag.AGIBOT_GENIE1_WAIST):
        raise ValueError(f"Only supports AGIBOT_GENIE1 / AGIBOT_GENIE1_WAIST, got {policy.embodiment_tag}")

    videos = images_to_video_payload(payload["images"])
    st = payload.get("state")
    if st is None:
        raise ValueError("Payload requires 'state'")
    states = flat_state_to_dict(st, policy.embodiment_tag)

    t_lang = len(modality_configs["language"].delta_indices)
    return {
        "video": {k: videos[k][np.newaxis, ...] for k in videos},
        "state": {k: states[k][np.newaxis, ...] for k in states},
        "language": {lang_key: [[str(prompt)] * t_lang]},
    }


def action_to_wire(
    policy: Gr00tPolicy, action: dict[str, np.ndarray],
    horizon: int | None = None, task_name: str = "",
) -> dict[str, Any]:
    """
    Convert policy output actions to the client's expected result format (same as N1.5 websocket).

    If ``horizon`` is set, only the first ``horizon`` timesteps are kept (applies to agibot and other embodiments).

    agibot_genie1:
        result = {
            "actions": [cmd0, cmd1, ...],  # each cmd is a 16-D list:
            # 0–7 left arm, 7–14 right arm, 14 left gripper, 15 right gripper
        }
    Other embodiments: still concatenate per-step vectors in ``action`` ``modality_keys`` order.
    """

    return _action_to_result_format_agibot(action, horizon=horizon, task_name=task_name)


def _action_part_2d(action: dict[str, np.ndarray], key: str, alt_key: str, default_cols: int) -> np.ndarray:
    """Extract (T, D) from action; Gr00tPolicy uses (B,T,D), use batch 0. Supports legacy keys ``action.*``."""
    arr = None
    for k in (key, alt_key):
        if k in action:
            arr = action[k]
            break
    if arr is None:
        return np.zeros((1, default_cols), dtype=np.float32)
    x = _ensure_ndarray(arr, dtype=np.float32)
    if x.ndim == 3:
        x = x[0]
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError(f"action[{key}] expected (B,T,D) or (T,D), got {arr.shape}")
    return x


def _action_to_result_format_agibot(
    action: dict[str, np.ndarray], horizon: int | None = None, task_name: str = "",
) -> dict[str, Any]:
    left_key = "left_arm_joint_position"
    right_key = "right_arm_joint_position"
    left_grip_key = "left_effector_position"
    right_grip_key = "right_effector_position"
    waist_key = "waist_position"

    left = _action_part_2d(action, left_key, f"action.{left_key}", 7)
    right = _action_part_2d(action, right_key, f"action.{right_key}", 7)
    left_grip = _action_part_2d(action, left_grip_key, f"action.{left_grip_key}", 1)
    right_grip = _action_part_2d(action, right_grip_key, f"action.{right_grip_key}", 1)

    has_waist = (
        (waist_key in action or f"action.{waist_key}" in action)
        and any(t in task_name for t in WAIST_TASK_NAMES)
    )
    if has_waist:
        waist = _action_part_2d(action, waist_key, f"action.{waist_key}", 1)
    else:
        waist = None

    parts = [left, right, left_grip, right_grip]
    if waist is not None:
        parts.append(waist)

    T = max(p.shape[0] for p in parts)
    parts = [
        np.pad(p, ((0, T - p.shape[0]), (0, 0)), constant_values=0) if p.shape[0] < T else p
        for p in parts
    ]

    if horizon is not None:
        T = min(T, max(0, int(horizon)))
        parts = [p[:T] for p in parts]

    if has_waist:
        left, right, left_grip, right_grip, waist = parts
        pad_zeros = np.zeros((left.shape[0], 4), dtype=np.float32)
        cmds = np.concatenate([left, right, left_grip, right_grip, pad_zeros, waist], axis=-1)
    else:
        left, right, left_grip, right_grip = parts
        cmds = np.concatenate(parts, axis=-1)
    
    by_key = {
        left_key: left.tolist(),
        right_key: right.tolist(),
        left_grip_key: left_grip.tolist(),
        right_grip_key: right_grip.tolist(),
    }
    if has_waist:
        by_key[waist_key] = waist.tolist()

    return {
        "actions": [cmd.tolist() for cmd in cmds],
        "actions_by_key": by_key,
    }


def _make_pack_unpack():
    import msgpack

    def unpack_array(obj: Any) -> Any:
        if isinstance(obj, dict) and b"__ndarray__" in obj:
            return np.ndarray(
                buffer=obj[b"data"],
                dtype=np.dtype(obj[b"dtype"]),
                shape=tuple(obj[b"shape"]),
            )
        if isinstance(obj, dict) and b"__npgeneric__" in obj:
            return np.dtype(obj[b"dtype"]).type(obj[b"data"])
        return obj

    try:
        import msgpack_numpy as m

        m.patch()
    except ImportError:
        pass

    def _bytes_keys_to_str(x: Any) -> Any:
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                if isinstance(k, (bytes, bytearray)):
                    try:
                        k = k.decode("utf-8")
                    except Exception:
                        k = str(k)
                out[k] = _bytes_keys_to_str(v)
            return out
        if isinstance(x, list):
            return [_bytes_keys_to_str(v) for v in x]
        return x

    def _pack(obj: Any) -> bytes:
        try:
            import msgpack_numpy as m

            return msgpack.packb(obj, default=m.encode)
        except ImportError:

            def _enc(o: Any) -> Any:
                if isinstance(o, np.ndarray):
                    return o.tolist()
                raise TypeError(type(o))

            return msgpack.packb(obj, default=_enc)

    def _unpack_raw(data: bytes) -> Any:
        try:
            import msgpack_numpy as m

            return msgpack.unpackb(
                data,
                raw=True,
                strict_map_key=False,
                object_hook=lambda o: unpack_array(m.decode(o)),
            )
        except ImportError:
            return msgpack.unpackb(
                data,
                raw=True,
                strict_map_key=False,
                object_hook=unpack_array,
            )

    def unpack(data: bytes) -> Any:
        return _bytes_keys_to_str(_unpack_raw(data))

    return _pack, unpack


async def _handler(
    ws: Any,
    policy: Gr00tPolicy,
    pack: Any,
    unpack: Any,
    action_horizon: int | None = None,
) -> None:
    logger.info("Connection from %s opened", ws.remote_address)
    modality_configs = policy.get_modality_config()
    meta = {
        "embodiment": policy.embodiment_tag.value,
        "video_keys": modality_configs["video"].modality_keys,
        "state_keys": modality_configs["state"].modality_keys,
        "action_keys": modality_configs["action"].modality_keys,
        "language_keys": modality_configs["language"].modality_keys,
    }
    await ws.send(pack(meta))

    req_idx = 0
    dump_dir = root_dir / "outputs" / "ws_obs_video_first_frames"
    dump_dir.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            raw = await ws.recv()
            payload = unpack(raw)
            if not isinstance(payload, dict):
                raise TypeError(f"Expected dict payload, got {type(payload)}")
            obs = payload_to_observation(policy, payload)
            task_name = str(payload.get("task_name") or payload.get("prompt") or "")
            action, _info = policy.get_action(obs)
            resp = action_to_wire(policy, action, horizon=action_horizon, task_name=task_name)
            await ws.send(pack(resp))
            req_idx += 1
        except Exception:
            logger.exception("Inference error")
            await ws.send(pack({"error": "inference_failed", "traceback": traceback.format_exc()}))
            await ws.close(1011, "Internal error")


@dataclass
class ServeArgs:
    model_path: str | Path
    """HF Hub ID or local checkpoint directory."""
    device: str = "cuda:0"
    host: str = "0.0.0.0"
    port: int = 8000
    modality_config_path: str | None = None
    """When using the same training-time `.py` modality extension, load/register before inference (same as launch_finetune)."""
    action_horizon: int | None = None
    """If set, return only the first N timesteps of actions (agibot and other embodiments)."""


def main(args: ServeArgs) -> None:
    try:
        import websockets.asyncio.server as ws_server  # type: ignore
    except ImportError as e:
        raise ImportError(
            "websockets is required; run: uv sync --extra websocket or pip install websockets"
        ) from e

    if args.modality_config_path:
        load_modality_config(args.modality_config_path)
    else:
        root_dir = Path(__file__).parent.parent.parent
        modality_config_path = root_dir / "gr00t" / "configs" / "data" / "embodiment_configs.py"
        load_modality_config(str(modality_config_path))
    data_statistics_file = Path(args.model_path) /"experiment_cfg" /"dataset_statistics.json"
    import json
    with open(data_statistics_file, "r") as f:
        data_statistics = json.load(f)
        for key in data_statistics:
            if "agibot_genie1_waist" in key:
                embodiment_tag = EmbodimentTag.AGIBOT_GENIE1_WAIST
            else:
                embodiment_tag = EmbodimentTag.AGIBOT_GENIE1

    tag = _normalize_embodiment_tag(embodiment_tag)
    policy = Gr00tPolicy(
        embodiment_tag=tag,
        model_path=str(args.model_path),
        device=args.device,
    )

    pack, unpack = _make_pack_unpack()

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError:
        local_ip = "127.0.0.1"
    logger.info(
        "GR00T WebSocket server: host=%s ip=%s port=%s embodiment=%s",
        hostname,
        local_ip,
        args.port,
        tag.value,
    )

    async def run() -> None:
        async with ws_server.serve(
            lambda ws: _handler(ws, policy, pack, unpack, args.action_horizon),
            args.host,
            args.port,
            compression=None,
            max_size=2**28,
        ) as server:
            await server.serve_forever()

    asyncio.run(run())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(ServeArgs))
