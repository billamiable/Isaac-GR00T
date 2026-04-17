#!/usr/bin/env python3
"""Compare raw GenieSim samples against preprocessed training inputs.

This script aligns one episode/step between:
1. raw instruction dataset
2. instruction_preprocessed dataset

It reports differences for:
- trim alignment
- language/task text
- selected parquet values (state/action groups when available)
- decoded video frames
- frames after the GR00T eval-time image transform
- collated VLM inputs produced by the GR00T processor

It also saves a few image artifacts to help inspect visual differences.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.model.gr00t_n1d6.image_augmentations import apply_with_replay
from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
from gr00t.utils.video_utils import get_frames_by_indices


DEFAULT_RAW_ROOT = Path(
    "/home/yujie/workspace/yujie/iDataset/simulation/genie_sim/dataset/task_suite/instruction"
)
DEFAULT_PREPROCESSED_ROOT = Path(
    "/home/yujie/workspace/yujie/iDataset/simulation/genie_sim/dataset/task_suite/instruction_preprocessed"
)
DEFAULT_OUTPUT_ROOT = Path("debug_outputs/preprocess_compare")
DEFAULT_TASK = "pick_block_color"
DEFAULT_EMBODIMENT = "agibot_genie1"


@dataclass
class DatasetMeta:
    root: Path
    info: dict[str, Any]
    episodes_map: dict[int, dict[str, Any]]
    modality: dict[str, Any]
    include_waist: bool


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _get_hl_text(entry: Any) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        value = entry.get("high_level_instruction", "")
        return value if isinstance(value, str) else str(value)
    return str(entry)


def _build_modality_from_info(info: dict[str, Any], *, include_waist: bool = True) -> dict[str, Any]:
    features = info.get("features", {})

    def _extract_indices(feature_key: str, field_suffix: str) -> list[int] | None:
        feat = features.get(feature_key, {})
        descs = feat.get("field_descriptions", {})
        for name, desc in descs.items():
            if name.endswith(field_suffix):
                return desc.get("indices", [])
        return None

    def _build_arm_and_effector(feature_key: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        joint_indices = _extract_indices(feature_key, "/joint/position")
        if joint_indices and len(joint_indices) >= 14:
            result["left_arm_joint_position"] = {
                "start": joint_indices[0],
                "end": joint_indices[6] + 1,
            }
            result["right_arm_joint_position"] = {
                "start": joint_indices[7],
                "end": joint_indices[13] + 1,
            }
        for side in ("left", "right"):
            eff_indices = _extract_indices(feature_key, f"/{side}_effector/position")
            if eff_indices:
                result[f"{side}_effector_position"] = {
                    "start": eff_indices[0],
                    "end": eff_indices[-1] + 1,
                }
        if include_waist:
            waist_indices = _extract_indices(feature_key, "/waist/position")
            if waist_indices and len(waist_indices) >= 5:
                result["waist_position"] = {
                    "start": waist_indices[4],
                    "end": waist_indices[4] + 1,
                }
        return result

    video_section = {}
    video_key_map = {
        "observation.images.top_head": "top_head",
        "observation.images.hand_left": "hand_left",
        "observation.images.hand_right": "hand_right",
    }
    for original_key, short_name in video_key_map.items():
        if original_key in features and features[original_key].get("dtype") == "video":
            video_section[short_name] = {"original_key": original_key}

    return {
        "state": _build_arm_and_effector("observation.state"),
        "action": _build_arm_and_effector("action"),
        "video": video_section,
        "annotation": {"language.action_text": {"original_key": "task_index"}},
    }


def resize_keep_aspect(frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w, _ = frame.shape
    scale = min(target_h / h, target_w / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    # Match preprocess_dataset.py exactly: use OpenCV default interpolation.
    resized = cv2.resize(frame, (new_w, new_h))
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return np.pad(
        resized,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=0,
    )


def to_uint8_image(array: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        arr = array.detach().cpu().numpy()
    else:
        arr = np.asarray(array)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    return arr


def save_image(path: Path, image: np.ndarray | torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8_image(image)).save(path)


def make_contact_sheet(images: list[tuple[str, np.ndarray | torch.Tensor]], out_path: Path) -> None:
    rendered = []
    for label, image in images:
        arr = to_uint8_image(image)
        title = np.full((24, arr.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(
            title,
            label,
            (6, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        rendered.append(np.concatenate([title, arr], axis=0))
    max_h = max(img.shape[0] for img in rendered)
    padded = []
    for img in rendered:
        if img.shape[0] < max_h:
            pad = np.full((max_h - img.shape[0], img.shape[1], 3), 255, dtype=np.uint8)
            img = np.concatenate([img, pad], axis=0)
        padded.append(img)
    canvas = np.concatenate(padded, axis=1)
    save_image(out_path, canvas)


def compute_trim_range(info: dict[str, Any], episode: int) -> tuple[int, int]:
    segs = info["instruction_segments"][str(episode)]
    starts = [seg["start_frame_index"] for seg in segs]
    ends = [seg["end_frame_index"] for seg in segs]
    return min(starts), max(ends)


def format_episode_path(info: dict[str, Any], pattern_key: str, episode: int, **kwargs: Any) -> str:
    chunk = episode // info["chunks_size"]
    pattern = info[pattern_key]
    return pattern.format(episode_chunk=chunk, episode_index=episode, **kwargs)


def load_dataset_meta(task_root: Path, include_waist: bool | None = None) -> DatasetMeta:
    meta_root = task_root / "meta"
    info = load_json(meta_root / "info.json")
    episodes_map = {row["episode_index"]: row for row in load_jsonl(meta_root / "episodes.jsonl")}
    modality_path = meta_root / "modality.json"
    if modality_path.exists():
        modality = load_json(modality_path)
        inferred_include_waist = "waist_position" in modality.get("state", {})
    else:
        inferred_include_waist = True if include_waist is None else include_waist
        modality = _build_modality_from_info(info, include_waist=inferred_include_waist)
    return DatasetMeta(
        root=task_root,
        info=info,
        episodes_map=episodes_map,
        modality=modality,
        include_waist=inferred_include_waist,
    )


def extract_instruction_for_frame(info: dict[str, Any], episode: int, frame_index: int) -> dict[str, Any] | None:
    for seg in info["instruction_segments"][str(episode)]:
        if seg["start_frame_index"] <= frame_index <= seg["end_frame_index"]:
            return seg
    return None


def derive_high_level_instruction(info: dict[str, Any], episode: int) -> str:
    segments = info["instruction_segments"][str(episode)]
    instructions = [seg.get("instruction", "") for seg in segments if seg.get("instruction")]
    if len(instructions) == 1:
        return instructions[0]
    punctuation = ".,!?;:，。！？；：、"
    cleaned = [inst.rstrip(punctuation) for inst in instructions]
    return ", ".join(cleaned)


def get_effective_episode_language(meta: DatasetMeta, episode: int) -> str:
    high_level_map = meta.info.get("high_level_instruction", {})
    if str(episode) in high_level_map:
        return _get_hl_text(high_level_map[str(episode)])
    return derive_high_level_instruction(meta.info, episode)


def read_frame(video_path: Path, frame_index: int, backend: str) -> np.ndarray:
    frame = get_frames_by_indices(str(video_path), np.array([frame_index]), video_backend=backend)[0]
    return np.asarray(frame)


def simulate_preprocessed_video(
    raw_video_path: Path,
    output_video_path: Path,
    fps: float,
    target_size: tuple[int, int] = (256, 256),
) -> Path:
    """Recreate preprocess_dataset.py video path: decode, resize/pad, write mp4v."""
    if output_video_path.exists():
        return output_video_path

    try:
        from decord import VideoReader
    except ImportError as exc:
        raise ImportError("decord required for simulated video preprocessing") from exc

    vr = VideoReader(str(raw_video_path))
    frames = vr.get_batch(range(len(vr))).asnumpy()
    target_h, target_w = target_size
    processed = np.array([resize_keep_aspect(frame, target_h, target_w) for frame in frames])

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (target_w, target_h))
    for frame in processed:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return output_video_path


def trim_video_like_pipeline(
    input_video_path: Path,
    output_video_path: Path,
    min_frame: int,
    max_frame: int,
    fps: float,
) -> Path:
    """Recreate trim_static_frames.py video path: slice frames, write mp4v again."""
    if output_video_path.exists():
        return output_video_path

    try:
        from decord import VideoReader
    except ImportError as exc:
        raise ImportError("decord required for simulated video trimming") from exc

    vr = VideoReader(str(input_video_path))
    total = len(vr)
    max_frame = min(max_frame, total - 1)
    min_frame = max(min_frame, 0)
    indices = list(range(min_frame, max_frame + 1))
    frames = vr.get_batch(indices).asnumpy()
    frames_bgr = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    for frame in frames_bgr:
        writer.write(frame)
    writer.release()
    return output_video_path


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def get_row_for_frame(df: pd.DataFrame, frame_index: int) -> pd.Series:
    if "frame_index" in df.columns:
        rows = df[df["frame_index"] == frame_index]
        if len(rows) == 0:
            raise KeyError(f"frame_index {frame_index} not found in parquet")
        return rows.iloc[0]
    return df.iloc[frame_index]


def get_action_window(df: pd.DataFrame, start_frame_index: int, horizon: int) -> pd.DataFrame:
    if "frame_index" in df.columns:
        rows = df[(df["frame_index"] >= start_frame_index) & (df["frame_index"] < start_frame_index + horizon)]
        return rows.sort_values("frame_index")
    return df.iloc[start_frame_index : start_frame_index + horizon]


def extract_group_from_row(row: pd.Series, modality_section: dict[str, Any], group_name: str, default_key: str) -> np.ndarray:
    group = modality_section[group_name]
    original_key = group.get("original_key", default_key)
    value = np.asarray(row[original_key])
    return value[group["start"] : group["end"]]


def extract_group_from_window(
    df: pd.DataFrame,
    modality_section: dict[str, Any],
    group_name: str,
    default_key: str,
) -> np.ndarray:
    group = modality_section[group_name]
    original_key = group.get("original_key", default_key)
    values = []
    for _, row in df.iterrows():
        value = np.asarray(row[original_key])
        values.append(value[group["start"] : group["end"]])
    return np.stack(values, axis=0)


def array_diff_summary(a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor) -> dict[str, Any]:
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
    a = np.asarray(a)
    b = np.asarray(b)
    summary: dict[str, Any] = {
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
    }
    if a.shape != b.shape:
        summary["same_shape"] = False
        return summary
    diff = a.astype(np.float64) - b.astype(np.float64)
    abs_diff = np.abs(diff)
    summary.update(
        {
            "same_shape": True,
            "mae": float(abs_diff.mean()),
            "max_abs": float(abs_diff.max()),
            "l2": float(np.linalg.norm(diff)),
            "allclose_atol_1e-5": bool(np.allclose(a, b, atol=1e-5)),
            "exact_equal": bool(np.array_equal(a, b)),
        }
    )
    return summary


def tensor_tree(tree: Any, prefix: str = "") -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    if isinstance(tree, torch.Tensor):
        out[prefix or "tensor"] = tree
        return out
    if isinstance(tree, (list, tuple)):
        for i, item in enumerate(tree):
            out.update(tensor_tree(item, f"{prefix}[{i}]"))
        return out
    if isinstance(tree, dict):
        for key, value in tree.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(tensor_tree(value, next_prefix))
    return out


def build_processor(embodiment: str) -> Gr00tN1d6Processor:
    return Gr00tN1d6Processor(
        modality_configs={embodiment: MODALITY_CONFIGS[embodiment]},
        use_albumentations=True,
        shortest_image_edge=256,
        crop_fraction=0.95,
        random_rotation_angle=None,
        color_jitter_params=None,
    )


def normalize_language_value(text: Any) -> str:
    if isinstance(text, str):
        return text
    if isinstance(text, list):
        return " ".join(str(item) for item in text)
    return str(text)


def formalize_language(text: Any, enabled: bool = True) -> str:
    text = normalize_language_value(text)
    if not enabled:
        return text
    text = text.lower()
    return re.sub(r"[^\w\s]", "", text)


def collate_vlm_inputs(
    processor: Gr00tN1d6Processor,
    image_keys: list[str],
    images: dict[str, list[np.ndarray]],
    language: str,
) -> dict[str, Any]:
    vlm_inputs = processor._get_vlm_inputs(
        image_keys=image_keys,
        images=images,
        masks=None,
        image_transform=processor.eval_image_transform,
        language=formalize_language(language, processor.formalize_language),
    )
    return processor.collator([vlm_inputs]).data["inputs"]


def apply_processor_eval_transform(
    processor: Gr00tN1d6Processor,
    frame: np.ndarray,
) -> torch.Tensor:
    if processor.use_albumentations:
        transformed, _ = apply_with_replay(
            processor.eval_image_transform,
            [Image.fromarray(frame)],
            replay=None,
        )
        return transformed[0]
    return processor.eval_image_transform(Image.fromarray(frame))


def compare_tensor_batches(batch_a: dict[str, Any], batch_b: dict[str, Any]) -> dict[str, Any]:
    tensors_a = tensor_tree(batch_a)
    tensors_b = tensor_tree(batch_b)
    shared = sorted(set(tensors_a) & set(tensors_b))
    result: dict[str, Any] = {}
    for key in shared:
        result[key] = array_diff_summary(tensors_a[key], tensors_b[key])
    missing = {
        "only_a": sorted(set(tensors_a) - set(tensors_b)),
        "only_b": sorted(set(tensors_b) - set(tensors_a)),
    }
    if missing["only_a"] or missing["only_b"]:
        result["_missing"] = missing
    return result


def compact_vlm_diff(diff: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "input_ids_equal": None,
        "attention_mask_equal": None,
        "pixel_values_mae": {},
        "pixel_values_max_abs": {},
    }
    if "input_ids" in diff:
        compact["input_ids_equal"] = diff["input_ids"].get("exact_equal")
    if "attention_mask" in diff:
        compact["attention_mask_equal"] = diff["attention_mask"].get("exact_equal")

    for key, value in diff.items():
        if not key.startswith("pixel_values["):
            continue
        compact["pixel_values_mae"][key] = value.get("mae")
        compact["pixel_values_max_abs"][key] = value.get("max_abs")
    return compact


def build_video_path(meta: DatasetMeta, episode: int, view_name: str) -> Path:
    original_key = meta.modality["video"][view_name].get("original_key", f"observation.images.{view_name}")
    return meta.root / format_episode_path(
        meta.info,
        "video_path",
        episode,
        video_key=original_key,
    )


def build_parquet_path(meta: DatasetMeta, episode: int) -> Path:
    return meta.root / format_episode_path(meta.info, "data_path", episode)


def compare_views(
    raw_meta: DatasetMeta,
    pre_meta: DatasetMeta,
    processor: Gr00tN1d6Processor,
    episode: int,
    raw_frame_index: int,
    pre_step_index: int,
    raw_trim_start: int,
    raw_trim_end: int,
    views: list[str],
    video_backend: str,
    output_dir: Path,
) -> tuple[
    dict[str, Any],
    dict[str, list[np.ndarray]],
    dict[str, list[np.ndarray]],
    dict[str, list[np.ndarray]],
    dict[str, list[np.ndarray]],
    str,
]:
    raw_images: dict[str, list[np.ndarray]] = {}
    pre_images: dict[str, list[np.ndarray]] = {}
    simulated_images: dict[str, list[np.ndarray]] = {}
    simulated_full_pipeline_images: dict[str, list[np.ndarray]] = {}
    view_summary: dict[str, Any] = {}

    for view in views:
        raw_video_path = build_video_path(raw_meta, episode, view)
        pre_video_path = build_video_path(pre_meta, episode, view)
        simulated_video_path = output_dir / "simulated_preprocess_videos" / f"{view}.mp4"
        simulated_video_path = simulate_preprocessed_video(
            raw_video_path=raw_video_path,
            output_video_path=simulated_video_path,
            fps=raw_meta.info.get("fps", 30),
        )
        simulated_trimmed_video_path = output_dir / "simulated_full_pipeline_videos" / f"{view}.mp4"
        simulated_trimmed_video_path = trim_video_like_pipeline(
            input_video_path=simulated_video_path,
            output_video_path=simulated_trimmed_video_path,
            min_frame=raw_trim_start,
            max_frame=raw_trim_end,
            fps=raw_meta.info.get("fps", 30),
        )

        raw_frame = read_frame(raw_video_path, raw_frame_index, video_backend)
        pre_frame = read_frame(build_video_path(pre_meta, episode, view), pre_step_index, video_backend)
        simulated_frame = read_frame(simulated_video_path, raw_frame_index, video_backend)
        simulated_trimmed_frame = read_frame(simulated_trimmed_video_path, pre_step_index, video_backend)
        raw_preprocess_like = resize_keep_aspect(raw_frame, 256, 256)
        raw_eval = apply_processor_eval_transform(processor, raw_frame)
        pre_eval = apply_processor_eval_transform(processor, pre_frame)
        simulated_eval = apply_processor_eval_transform(processor, simulated_frame)
        simulated_trimmed_eval = apply_processor_eval_transform(processor, simulated_trimmed_frame)

        raw_images[view] = [raw_frame]
        pre_images[view] = [pre_frame]
        simulated_images[view] = [simulated_frame]
        simulated_full_pipeline_images[view] = [simulated_trimmed_frame]

        view_dir = output_dir / "images" / view
        save_image(view_dir / "raw_original.png", raw_frame)
        save_image(view_dir / "raw_preprocess_like.png", raw_preprocess_like)
        save_image(view_dir / "simulated_redecoded.png", simulated_frame)
        save_image(view_dir / "simulated_full_pipeline_redecoded.png", simulated_trimmed_frame)
        save_image(view_dir / "pre_decoded.png", pre_frame)
        save_image(view_dir / "raw_processor_eval.png", raw_eval)
        save_image(view_dir / "simulated_processor_eval.png", simulated_eval)
        save_image(view_dir / "simulated_full_pipeline_processor_eval.png", simulated_trimmed_eval)
        save_image(view_dir / "pre_processor_eval.png", pre_eval)
        make_contact_sheet(
            [
                ("raw_original", raw_frame),
                ("raw_preprocess_like", raw_preprocess_like),
                ("simulated_redecoded", simulated_frame),
                ("sim_full_pipeline", simulated_trimmed_frame),
                ("pre_decoded", pre_frame),
                ("raw_processor_eval", raw_eval),
                ("simulated_processor_eval", simulated_eval),
                ("sim_full_proc_eval", simulated_trimmed_eval),
                ("pre_processor_eval", pre_eval),
            ],
            view_dir / "contact_sheet.png",
        )

        view_summary[view] = {
            "raw_video_path": str(raw_video_path),
            "simulated_video_path": str(simulated_video_path),
            "simulated_full_pipeline_video_path": str(simulated_trimmed_video_path),
            "preprocessed_video_path": str(pre_video_path),
            "raw_original_shape": list(raw_frame.shape),
            "simulated_decoded_shape": list(simulated_frame.shape),
            "simulated_full_pipeline_decoded_shape": list(simulated_trimmed_frame.shape),
            "pre_decoded_shape": list(pre_frame.shape),
            "raw_preprocess_like_vs_simulated_redecoded": array_diff_summary(
                raw_preprocess_like,
                simulated_frame,
            ),
            "raw_preprocess_like_vs_pre_decoded": array_diff_summary(raw_preprocess_like, pre_frame),
            "simulated_redecoded_vs_pre_decoded": array_diff_summary(simulated_frame, pre_frame),
            "simulated_full_pipeline_vs_pre_decoded": array_diff_summary(
                simulated_trimmed_frame,
                pre_frame,
            ),
            "simulated_processor_eval_vs_pre_processor_eval": array_diff_summary(
                simulated_eval,
                pre_eval,
            ),
            "simulated_full_pipeline_processor_eval_vs_pre_processor_eval": array_diff_summary(
                simulated_trimmed_eval,
                pre_eval,
            ),
            "raw_processor_eval_vs_pre_processor_eval": array_diff_summary(raw_eval, pre_eval),
        }

    return (
        view_summary,
        raw_images,
        pre_images,
        simulated_images,
        simulated_full_pipeline_images,
        str(output_dir / "images"),
    )


def compare_parquet_inputs(
    raw_meta: DatasetMeta,
    pre_meta: DatasetMeta,
    episode: int,
    raw_frame_index: int,
    pre_step_index: int,
    action_horizon: int,
) -> dict[str, Any]:
    raw_df = load_parquet(build_parquet_path(raw_meta, episode))
    pre_df = load_parquet(build_parquet_path(pre_meta, episode))

    raw_row = get_row_for_frame(raw_df, raw_frame_index)
    pre_row = get_row_for_frame(pre_df, pre_step_index)
    raw_action_rows = get_action_window(raw_df, raw_frame_index, action_horizon)
    pre_action_rows = get_action_window(pre_df, pre_step_index, action_horizon)
    common_horizon = min(len(raw_action_rows), len(pre_action_rows))
    raw_action_rows = raw_action_rows.iloc[:common_horizon]
    pre_action_rows = pre_action_rows.iloc[:common_horizon]

    state_summary = {}
    for key in sorted(set(raw_meta.modality["state"]) & set(pre_meta.modality["state"])):
        state_summary[key] = array_diff_summary(
            extract_group_from_row(raw_row, raw_meta.modality["state"], key, "observation.state"),
            extract_group_from_row(pre_row, pre_meta.modality["state"], key, "observation.state"),
        )

    action_summary = {}
    for key in sorted(set(raw_meta.modality["action"]) & set(pre_meta.modality["action"])):
        action_summary[key] = array_diff_summary(
            extract_group_from_window(raw_action_rows, raw_meta.modality["action"], key, "action"),
            extract_group_from_window(pre_action_rows, pre_meta.modality["action"], key, "action"),
        )

    misc_columns = {}
    for col in ["frame_index", "timestamp", "task_index", "episode_index", "index"]:
        if col in raw_row and col in pre_row:
            raw_value = raw_row[col].item() if hasattr(raw_row[col], "item") else raw_row[col]
            pre_value = pre_row[col].item() if hasattr(pre_row[col], "item") else pre_row[col]
            misc_columns[col] = {"raw": raw_value, "preprocessed": pre_value}

    return {
        "raw_parquet_path": str(build_parquet_path(raw_meta, episode)),
        "preprocessed_parquet_path": str(build_parquet_path(pre_meta, episode)),
        "raw_row_count": int(len(raw_df)),
        "preprocessed_row_count": int(len(pre_df)),
        "action_horizon_compared": common_horizon,
        "columns": misc_columns,
        "state_groups": state_summary,
        "action_groups": action_summary,
    }


def resolve_views(raw_meta: DatasetMeta, pre_meta: DatasetMeta, requested_views: list[str] | None) -> list[str]:
    shared = sorted(set(raw_meta.modality["video"]) & set(pre_meta.modality["video"]))
    if requested_views:
        missing = sorted(set(requested_views) - set(shared))
        if missing:
            raise ValueError(f"Requested views not shared by raw/preprocessed dataset: {missing}")
        return requested_views
    return shared


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--preprocessed-root", type=Path, default=DEFAULT_PREPROCESSED_ROOT)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--pre-step",
        type=int,
        default=0,
        help="Step index in the trimmed/preprocessed timeline.",
    )
    parser.add_argument("--embodiment", type=str, default=DEFAULT_EMBODIMENT)
    parser.add_argument("--video-backend", type=str, default="torchcodec")
    parser.add_argument("--views", nargs="*", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--full-vlm-diff",
        action="store_true",
        help="Store full tensor-by-tensor VLM diffs in summary.json (larger file).",
    )
    args = parser.parse_args()

    raw_task_root = args.raw_root / args.task
    pre_task_root = args.preprocessed_root / args.task
    if not raw_task_root.exists():
        raise FileNotFoundError(f"Raw task path not found: {raw_task_root}")
    if not pre_task_root.exists():
        raise FileNotFoundError(f"Preprocessed task path not found: {pre_task_root}")

    pre_meta_preview = load_dataset_meta(pre_task_root)
    raw_meta = load_dataset_meta(raw_task_root, include_waist=pre_meta_preview.include_waist)
    pre_meta = load_dataset_meta(pre_task_root, include_waist=pre_meta_preview.include_waist)

    if args.episode not in raw_meta.episodes_map:
        raise KeyError(f"Episode {args.episode} not present in raw dataset")
    if args.episode not in pre_meta.episodes_map:
        raise KeyError(f"Episode {args.episode} not present in preprocessed dataset")

    raw_trim_start, raw_trim_end = compute_trim_range(raw_meta.info, args.episode)
    pre_trim_start, pre_trim_end = compute_trim_range(pre_meta.info, args.episode)
    expected_pre_length = raw_trim_end - raw_trim_start + 1
    if pre_trim_start != 0:
        raise ValueError(f"Expected preprocessed trim start to be 0, got {pre_trim_start}")
    if not (0 <= args.pre_step <= pre_trim_end):
        raise ValueError(f"--pre-step must be within [0, {pre_trim_end}]")
    raw_frame_index = raw_trim_start + args.pre_step

    processor = build_processor(args.embodiment)
    processor.eval()
    views = resolve_views(raw_meta, pre_meta, args.views)
    output_dir = (args.output_root / args.task / f"episode_{args.episode:06d}" / f"step_{args.pre_step:04d}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_episode_text = get_effective_episode_language(raw_meta, args.episode)
    pre_episode_text = get_effective_episode_language(pre_meta, args.episode)
    raw_segment = extract_instruction_for_frame(raw_meta.info, args.episode, raw_frame_index)
    pre_segment = extract_instruction_for_frame(pre_meta.info, args.episode, args.pre_step)

    parquet_summary = compare_parquet_inputs(
        raw_meta=raw_meta,
        pre_meta=pre_meta,
        episode=args.episode,
        raw_frame_index=raw_frame_index,
        pre_step_index=args.pre_step,
        action_horizon=16,
    )
    (
        view_summary,
        raw_images,
        pre_images,
        simulated_images,
        simulated_full_pipeline_images,
        images_dir,
    ) = compare_views(
        raw_meta=raw_meta,
        pre_meta=pre_meta,
        processor=processor,
        episode=args.episode,
        raw_frame_index=raw_frame_index,
        pre_step_index=args.pre_step,
        raw_trim_start=raw_trim_start,
        raw_trim_end=raw_trim_end,
        views=views,
        video_backend=args.video_backend,
        output_dir=output_dir,
    )

    raw_batch = collate_vlm_inputs(processor, views, raw_images, raw_episode_text)
    pre_batch = collate_vlm_inputs(processor, views, pre_images, pre_episode_text)
    vlm_raw_vs_pre = compare_tensor_batches(raw_batch, pre_batch)
    simulated_batch = collate_vlm_inputs(processor, views, simulated_images, raw_episode_text)
    simulated_full_pipeline_batch = collate_vlm_inputs(
        processor,
        views,
        simulated_full_pipeline_images,
        raw_episode_text,
    )
    vlm_simulated_vs_pre = compare_tensor_batches(simulated_batch, pre_batch)
    vlm_full_pipeline_vs_pre = compare_tensor_batches(simulated_full_pipeline_batch, pre_batch)
    vlm_raw_vs_simulated = compare_tensor_batches(raw_batch, simulated_batch)
    vlm_raw_vs_full_pipeline = compare_tensor_batches(raw_batch, simulated_full_pipeline_batch)
    vlm_compact = {
        "raw_direct_vs_preprocessed": compact_vlm_diff(vlm_raw_vs_pre),
        "simulated_once_vs_preprocessed": compact_vlm_diff(vlm_simulated_vs_pre),
        "simulated_full_pipeline_vs_preprocessed": compact_vlm_diff(vlm_full_pipeline_vs_pre),
        "raw_direct_vs_simulated_once": compact_vlm_diff(vlm_raw_vs_simulated),
        "raw_direct_vs_simulated_full_pipeline": compact_vlm_diff(vlm_raw_vs_full_pipeline),
    }

    summary = {
        "task": args.task,
        "episode": args.episode,
        "pre_step": args.pre_step,
        "raw_frame_index": raw_frame_index,
        "embodiment": args.embodiment,
        "video_backend": args.video_backend,
        "views": views,
        "trim_alignment": {
            "raw_keep_range": [raw_trim_start, raw_trim_end],
            "preprocessed_keep_range": [pre_trim_start, pre_trim_end],
            "expected_preprocessed_length_from_raw_trim": expected_pre_length,
            "recorded_preprocessed_length": pre_meta.episodes_map[args.episode]["length"],
        },
        "language": {
            "raw_effective_language": raw_episode_text,
            "preprocessed_effective_language": pre_episode_text,
            "raw_episodes_jsonl_task": raw_meta.episodes_map[args.episode].get("tasks"),
            "preprocessed_episodes_jsonl_task": pre_meta.episodes_map[args.episode].get("tasks"),
            "raw_segment": raw_segment,
            "preprocessed_segment": pre_segment,
        },
        "parquet": parquet_summary,
        "video": view_summary,
        "vlm_collated_inputs_compact": vlm_compact,
        "artifacts": {
            "output_dir": str(output_dir),
            "images_dir": images_dir,
            "summary_json": str(output_dir / "summary.json"),
        },
    }
    if args.full_vlm_diff:
        summary["vlm_collated_inputs_full"] = {
            "raw_direct_vs_preprocessed": vlm_raw_vs_pre,
            "simulated_once_vs_preprocessed": vlm_simulated_vs_pre,
            "simulated_full_pipeline_vs_preprocessed": vlm_full_pipeline_vs_pre,
            "raw_direct_vs_simulated_once": vlm_raw_vs_simulated,
            "raw_direct_vs_simulated_full_pipeline": vlm_raw_vs_full_pipeline,
        }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    compact = {
        "task": args.task,
        "episode": args.episode,
        "pre_step": args.pre_step,
        "raw_frame_index": raw_frame_index,
        "output_dir": str(output_dir),
        "comparison_definition": {
            "1_raw_resize_vs_preprocessed_decode": "raw_preprocess_like_vs_pre_decoded",
            "2_raw_reencode_once_vs_preprocessed_decode": "simulated_redecoded_vs_pre_decoded",
            "3_raw_full_pipeline_vs_preprocessed_decode": "simulated_full_pipeline_vs_pre_decoded",
        },
        "video_mae": {
            view: round(metrics["raw_preprocess_like_vs_pre_decoded"]["mae"], 6)
            for view, metrics in view_summary.items()
            if metrics["raw_preprocess_like_vs_pre_decoded"].get("same_shape")
        },
        "simulated_video_mae": {
            view: round(metrics["simulated_redecoded_vs_pre_decoded"]["mae"], 6)
            for view, metrics in view_summary.items()
            if metrics["simulated_redecoded_vs_pre_decoded"].get("same_shape")
        },
        "simulated_full_pipeline_video_mae": {
            view: round(metrics["simulated_full_pipeline_vs_pre_decoded"]["mae"], 6)
            for view, metrics in view_summary.items()
            if metrics["simulated_full_pipeline_vs_pre_decoded"].get("same_shape")
        },
        "encode_decode_only_mae": {
            view: round(metrics["raw_preprocess_like_vs_simulated_redecoded"]["mae"], 6)
            for view, metrics in view_summary.items()
            if metrics["raw_preprocess_like_vs_simulated_redecoded"].get("same_shape")
        },
        "processor_mae": {
            view: round(metrics["raw_processor_eval_vs_pre_processor_eval"]["mae"], 6)
            for view, metrics in view_summary.items()
            if metrics["raw_processor_eval_vs_pre_processor_eval"].get("same_shape")
        },
        "simulated_processor_mae": {
            view: round(metrics["simulated_processor_eval_vs_pre_processor_eval"]["mae"], 6)
            for view, metrics in view_summary.items()
            if metrics["simulated_processor_eval_vs_pre_processor_eval"].get("same_shape")
        },
        "simulated_full_pipeline_processor_mae": {
            view: round(metrics["simulated_full_pipeline_processor_eval_vs_pre_processor_eval"]["mae"], 6)
            for view, metrics in view_summary.items()
            if metrics["simulated_full_pipeline_processor_eval_vs_pre_processor_eval"].get("same_shape")
        },
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))
    print(f"Full summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
