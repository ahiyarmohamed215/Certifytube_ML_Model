from __future__ import annotations

"""Shared raw-event to feature pipeline for engagement training and inference.

The notebook uses this module when it builds the training dataset, and the API
uses the same module when backend requests send raw session events instead of
precomputed features. Keeping this logic in one file prevents training and live
prediction from drifting apart.
"""

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

RAW_EVENT_COLUMNS = [
    "event_id",
    "session_id",
    "user_id",
    "video_id",
    "video_title",
    "event_type",
    "player_state",
    "playback_rate",
    "current_time_sec",
    "video_duration_sec",
    "created_at_utc",
    "client_created_at_local",
    "client_tz_offset_min",
    "seek_from_sec",
    "seek_to_sec",
]

REQUIRED_EVENT_COLUMNS = [
    "session_id",
    "event_type",
    "created_at_utc",
]

NUMERIC_EVENT_COLUMNS = [
    "playback_rate",
    "current_time_sec",
    "video_duration_sec",
    "seek_from_sec",
    "seek_to_sec",
    "client_tz_offset_min",
]

META_COLUMNS = ["session_id", "user_id", "video_id", "video_title"]
FEATURE_COLUMNS = [
    "session_duration_sec",
    "video_duration_sec",
    "last_position_sec",
    "completed_flag",
    "watch_time_sec",
    "watch_time_ratio",
    "completion_ratio",
    "engagement_velocity",
    "num_pause",
    "total_pause_duration_sec",
    "avg_pause_duration_sec",
    "median_pause_duration_sec",
    "pause_freq_per_min",
    "long_pause_count",
    "long_pause_ratio",
    "num_seek",
    "num_seek_forward",
    "num_seek_backward",
    "total_seek_forward_sec",
    "total_seek_backward_sec",
    "avg_seek_forward_sec",
    "avg_seek_backward_sec",
    "largest_forward_seek_sec",
    "largest_backward_seek_sec",
    "seek_jump_std_sec",
    "seek_forward_ratio",
    "seek_backward_ratio",
    "skip_time_ratio",
    "rewatch_time_ratio",
    "rewatch_to_skip_ratio",
    "seek_density_per_min",
    "first_seek_time_sec",
    "early_skip_flag",
    "num_ratechange",
    "time_at_speed_lt1x_sec",
    "time_at_speed_1x_sec",
    "time_at_speed_gt1x_sec",
    "fast_ratio",
    "slow_ratio",
    "playback_speed_variance",
    "avg_playback_rate_when_playing",
    "unique_speed_levels",
    "num_buffering_events",
    "buffering_time_sec",
    "buffering_freq_per_min",
    "play_pause_ratio",
    "attention_index",
    "skim_flag",
    "deep_flag",
]

EPS = 1e-9
LONG_PAUSE_THRESHOLD = 30.0


class EventPipelineError(Exception):
    pass


def events_to_frame(events: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Convert backend event payloads into a normalized dataframe shape."""
    if not events:
        raise EventPipelineError("No events provided.")

    frame = pd.DataFrame([dict(event) for event in events])
    for column in RAW_EVENT_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan

    for column in REQUIRED_EVENT_COLUMNS:
        if column not in frame.columns:
            raise EventPipelineError(f"Missing required raw event column '{column}'.")

    return frame[RAW_EVENT_COLUMNS].copy()


def clean_raw_events(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the same raw-event cleanup rules used by the notebook and API."""
    if df_raw.empty:
        raise EventPipelineError("Raw event dataframe is empty.")

    df = df_raw.copy()

    for column in RAW_EVENT_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    for column in REQUIRED_EVENT_COLUMNS:
        if column not in df.columns:
            raise EventPipelineError(f"Missing required raw event column '{column}'.")

    df["session_id"] = df["session_id"].astype(str).str.strip()
    df["user_id"] = df["user_id"].fillna("").astype(str)
    df["video_id"] = df["video_id"].fillna("").astype(str)
    df["video_title"] = df["video_title"].fillna("").astype(str)
    df["event_type"] = df["event_type"].fillna("").astype(str).str.strip().str.lower()

    df["created_at_utc"] = pd.to_datetime(df["created_at_utc"], errors="coerce", utc=True)
    df["client_created_at_local"] = pd.to_datetime(df["client_created_at_local"], errors="coerce", utc=True)

    for column in NUMERIC_EVENT_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["player_state"] = pd.to_numeric(df["player_state"], errors="coerce").astype("Int64")

    if "event_id" in df.columns:
        # Prefer deduplication by event_id when it exists. Rows without event_id
        # are still preserved because some callers may not send that field.
        with_id = df[df["event_id"].notna()].drop_duplicates(subset=["event_id"], keep="first")
        without_id = df[df["event_id"].isna()]
        df = pd.concat([with_id, without_id], ignore_index=True)

    df = df[df["created_at_utc"].notna()].copy()
    if df.empty:
        raise EventPipelineError("No valid events remain after timestamp cleaning.")

    for column in ("current_time_sec", "video_duration_sec"):
        if column in df.columns:
            df[column] = df[column].clip(lower=0)

    if "playback_rate" in df.columns:
        valid_mask = df["playback_rate"].notna()
        df.loc[valid_mask, "playback_rate"] = df.loc[valid_mask, "playback_rate"].clip(0.25, 4.0)

    df = df[df["session_id"].str.len() > 0].copy()
    if df.empty:
        raise EventPipelineError("No valid events remain after session_id cleaning.")

    return df.sort_values(["session_id", "created_at_utc"]).reset_index(drop=True)


def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse many raw event rows into one feature row per session."""
    if df_raw.empty:
        raise EventPipelineError("Cannot build features from an empty event dataframe.")

    all_sessions: List[Dict[str, Any]] = []

    for session_id, grp in df_raw.groupby("session_id"):
        grp = grp.sort_values("created_at_utc").reset_index(drop=True)
        if grp.empty:
            continue

        user_id = str(grp["user_id"].iloc[0])
        video_id = str(grp["video_id"].iloc[0])
        video_title = str(grp["video_title"].iloc[0])
        video_duration_value = grp["video_duration_sec"].max()
        video_duration = float(video_duration_value) if pd.notna(video_duration_value) else 0.0

        watch_time = 0.0
        pause_time = 0.0
        buffer_time = 0.0
        pause_durations: List[float] = []
        seek_fwd_dists: List[float] = []
        seek_bwd_dists: List[float] = []
        speeds_used: List[float] = []
        speed_seg = {"lt1x": 0.0, "1x": 0.0, "gt1x": 0.0}
        num_pause = 0
        num_buffering = 0
        num_seek = 0
        num_seek_fwd = 0
        num_seek_bwd = 0
        num_ratechange = 0
        completed = 0
        first_seek_time = None
        last_position = 0.0
        prev_state = None
        prev_time = None

        # Walk each session chronologically and simulate watch/pause/buffer time
        # from the previous player state until the next event timestamp.
        for _, row in grp.iterrows():
            cur_time = row["created_at_utc"]
            ev_type = str(row["event_type"]).lower()
            state = row["player_state"]
            pos = float(row["current_time_sec"]) if pd.notna(row["current_time_sec"]) else 0.0
            rate = float(row["playback_rate"]) if pd.notna(row["playback_rate"]) else 1.0
            last_position = max(last_position, pos)
            speeds_used.append(rate)

            if prev_time is not None and prev_state is not None:
                dt = max(0.0, (cur_time - prev_time).total_seconds())
                if prev_state == 1:
                    watch_time += dt
                    if rate < 1.0:
                        speed_seg["lt1x"] += dt
                    elif rate == 1.0:
                        speed_seg["1x"] += dt
                    else:
                        speed_seg["gt1x"] += dt
                elif prev_state == 2:
                    pause_time += dt
                    pause_durations.append(dt)
                elif prev_state == 3:
                    buffer_time += dt

            seek_from = row["seek_from_sec"]
            seek_to = row["seek_to_sec"]

            if ev_type == "pause":
                num_pause += 1
            elif ev_type == "buffering":
                num_buffering += 1
            elif ev_type == "ended":
                completed = 1
            elif ev_type == "seek" or (pd.notna(seek_from) and pd.notna(seek_to)):
                sfrom = float(seek_from) if pd.notna(seek_from) else 0.0
                sto = float(seek_to) if pd.notna(seek_to) else 0.0
                delta = sto - sfrom
                num_seek += 1
                if delta > 0:
                    num_seek_fwd += 1
                    seek_fwd_dists.append(abs(delta))
                elif delta < 0:
                    num_seek_bwd += 1
                    seek_bwd_dists.append(abs(delta))
                if first_seek_time is None:
                    first_seek_time = max(0.0, (cur_time - grp["created_at_utc"].iloc[0]).total_seconds())
            elif ev_type == "ratechange":
                num_ratechange += 1

            if pd.notna(state):
                prev_state = int(state)
            prev_time = cur_time

        session_duration = max(0.0, (grp["created_at_utc"].iloc[-1] - grp["created_at_utc"].iloc[0]).total_seconds())
        total_seek_forward = sum(seek_fwd_dists)
        total_seek_backward = sum(seek_bwd_dists)
        session_minutes = session_duration / 60.0 + EPS
        watched_time = watch_time + EPS
        all_seek = seek_fwd_dists + seek_bwd_dists

        # Each dictionary below becomes one session-level training/inference row.
        all_sessions.append(
            {
                "session_id": session_id,
                "user_id": user_id,
                "video_id": video_id,
                "video_title": video_title,
                "session_duration_sec": session_duration,
                "video_duration_sec": video_duration,
                "last_position_sec": last_position,
                "completed_flag": completed,
                "watch_time_sec": watch_time,
                "watch_time_ratio": watch_time / (session_duration + EPS),
                "completion_ratio": last_position / (video_duration + EPS),
                "engagement_velocity": watch_time / (session_duration + EPS),
                "num_pause": num_pause,
                "total_pause_duration_sec": pause_time,
                "avg_pause_duration_sec": float(np.mean(pause_durations)) if pause_durations else 0.0,
                "median_pause_duration_sec": float(np.median(pause_durations)) if pause_durations else 0.0,
                "pause_freq_per_min": num_pause / session_minutes,
                "long_pause_count": sum(1 for duration in pause_durations if duration > LONG_PAUSE_THRESHOLD),
                "long_pause_ratio": sum(1 for duration in pause_durations if duration > LONG_PAUSE_THRESHOLD) / (num_pause + EPS),
                "num_seek": num_seek,
                "num_seek_forward": num_seek_fwd,
                "num_seek_backward": num_seek_bwd,
                "total_seek_forward_sec": total_seek_forward,
                "total_seek_backward_sec": total_seek_backward,
                "avg_seek_forward_sec": float(np.mean(seek_fwd_dists)) if seek_fwd_dists else 0.0,
                "avg_seek_backward_sec": float(np.mean(seek_bwd_dists)) if seek_bwd_dists else 0.0,
                "largest_forward_seek_sec": max(seek_fwd_dists) if seek_fwd_dists else 0.0,
                "largest_backward_seek_sec": max(seek_bwd_dists) if seek_bwd_dists else 0.0,
                "seek_jump_std_sec": float(np.std(all_seek)) if len(all_seek) > 1 else 0.0,
                "seek_forward_ratio": num_seek_fwd / (num_seek + EPS),
                "seek_backward_ratio": num_seek_bwd / (num_seek + EPS),
                "skip_time_ratio": total_seek_forward / (video_duration + EPS),
                "rewatch_time_ratio": total_seek_backward / (video_duration + EPS),
                "rewatch_to_skip_ratio": total_seek_backward / (total_seek_forward + total_seek_backward + EPS),
                "seek_density_per_min": num_seek / session_minutes,
                "first_seek_time_sec": first_seek_time if first_seek_time is not None else 0.0,
                "early_skip_flag": 1 if (first_seek_time is not None and first_seek_time < video_duration * 0.1) else 0,
                "num_ratechange": num_ratechange,
                "time_at_speed_lt1x_sec": speed_seg["lt1x"],
                "time_at_speed_1x_sec": speed_seg["1x"],
                "time_at_speed_gt1x_sec": speed_seg["gt1x"],
                "fast_ratio": speed_seg["gt1x"] / watched_time,
                "slow_ratio": speed_seg["lt1x"] / watched_time,
                "playback_speed_variance": float(np.var(speeds_used)) if len(speeds_used) > 1 else 0.0,
                "avg_playback_rate_when_playing": float(np.mean(speeds_used)) if speeds_used else 1.0,
                "unique_speed_levels": len(set(speeds_used)),
                "num_buffering_events": num_buffering,
                "buffering_time_sec": buffer_time,
                "buffering_freq_per_min": num_buffering / session_minutes,
                "play_pause_ratio": watch_time / (watch_time + pause_time + EPS),
                "attention_index": (watch_time / (session_duration + EPS)) * (last_position / (video_duration + EPS)),
                "skim_flag": 1 if total_seek_forward / (video_duration + EPS) > 0.3 else 0,
                "deep_flag": 1 if (last_position / (video_duration + EPS) > 0.8 and total_seek_backward > 0) else 0,
                "engagement_label": np.nan,
            }
        )

    feature_frame = pd.DataFrame(all_sessions)
    if feature_frame.empty:
        raise EventPipelineError("No session-level features could be built from the provided events.")

    return feature_frame


def compute_features_from_events(
    events: Sequence[Mapping[str, Any]],
    expected_session_id: str | None = None,
) -> Dict[str, float]:
    """Build the exact 49-feature model payload for a single-session request."""
    raw_df = events_to_frame(events)
    clean_df = clean_raw_events(raw_df)

    # The API contract is one request per session. That keeps the output shape
    # simple: one request -> one feature row -> one score + explanation.
    session_ids = clean_df["session_id"].astype(str).unique().tolist()
    if len(session_ids) != 1:
        raise EventPipelineError(
            f"Expected exactly one session in request events, found {len(session_ids)}: {session_ids}"
        )

    actual_session_id = session_ids[0]
    if expected_session_id and actual_session_id != str(expected_session_id):
        raise EventPipelineError(
            f"Session mismatch: request session_id='{expected_session_id}' but event payload belongs to '{actual_session_id}'."
        )

    feature_frame = build_features(clean_df)
    if len(feature_frame) != 1:
        raise EventPipelineError(
            f"Expected a single session feature row, but built {len(feature_frame)} rows."
        )

    row = feature_frame.iloc[0]
    return {column: float(row[column]) for column in FEATURE_COLUMNS}
