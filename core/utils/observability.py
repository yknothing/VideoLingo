# ------------
# Unified Observability: logging + metrics (minimal, incremental)
# ------------

import os
import json
import time
import logging

# lazy imports to avoid heavy deps at import time
_INIT_DONE = False


def _read_bool(key, default=False):
    try:
        from core.utils.config_utils import load_key
        v = load_key(key)
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ["1", "true", "yes", "on"]
    except Exception:
        pass
    return default


def _read_str(key, default=None):
    try:
        from core.utils.config_utils import load_key
        v = load_key(key)
        if v is None or v == "":
            return default
        return str(v)
    except Exception:
        return default


def _ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def init_logging():
    # idempotent init
    global _INIT_DONE
    if _INIT_DONE:
        return

    # read level from config, default INFO
    level_str = _read_str("debug.log_level", "INFO").upper()
    level_map = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }
    level = level_map.get(level_str, logging.INFO)

    logger = logging.getLogger()
    logger.setLevel(level)

    # avoid duplicate handlers
    if not logger.handlers:
        console = logging.StreamHandler()
        fmt = "%(asctime)s %(levelname)s [%(name)s] video_id=%(video_id)s stage=%(stage)s op=%(op)s attempt=%(attempt)s %(message)s"
        console.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(console)

    _INIT_DONE = True


def _current_video_id():
    try:
        from core.utils.path_adapter import get_path_adapter
        return get_path_adapter().get_current_video_id()
    except Exception:
        return None


def _log_dir(video_id=None):
    try:
        from core.utils.path_adapter import get_path_adapter
        adapter = get_path_adapter()
        if video_id:
            adapter.set_current_video_id(video_id)
        return adapter.get_log_dir()
    except Exception:
        # fallback global log dir
        base = _read_str("video_storage.temp_dir", None)
        if not base:
            base = "temp"
        path = os.path.join(base, "log")
        _ensure_dir(path)
        return path


def _metrics_dir(video_id=None):
    # store metrics near logs for per-video; fallback to global output/metrics
    log_dir = _log_dir(video_id)
    mdir = os.path.join(log_dir, "metrics")
    _ensure_dir(mdir)
    return mdir


def _default_fields(extra):
    fields = {
        "video_id": extra.get("video_id") or _current_video_id() or "-",
        "stage": extra.get("stage") or "-",
        "op": extra.get("op") or "-",
        "attempt": extra.get("attempt") or "-",
    }
    return fields


def log_event(level, message, **extra):
    # initialize once
    init_logging()

    # build fields
    fields = _default_fields(extra)
    logger = logging.getLogger(extra.get("logger") or __name__)

    # write to console logger with structured format
    try:
        if str(level).lower() == "debug":
            logger.debug(message, extra=fields)
        elif str(level).lower() in ["warn", "warning"]:
            logger.warning(message, extra=fields)
        elif str(level).lower() == "error":
            logger.error(message, extra=fields)
        else:
            logger.info(message, extra=fields)
    except Exception:
        # never break pipeline on logging errors
        pass

    # json event to file
    try:
        event = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "level": str(level).upper(),
            "message": message,
        }
        event.update(fields)
        for k, v in extra.items():
            if k not in event:
                event[k] = v

        fdir = _log_dir(fields.get("video_id"))
        fpath = os.path.join(fdir, "events.jsonl")
        with open(fpath, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass


class time_block:
    # simple context manager for timing
    def __init__(self, label, **extra):
        self.label = label
        self.extra = extra
        self.start = None

    def __enter__(self):
        self.start = time.time()
        log_event("debug", f"start: {self.label}", **self.extra)
        return self

    def __exit__(self, exc_type, exc, tb):
        dur_ms = int((time.time() - self.start) * 1000) if self.start else -1
        e = dict(self.extra)
        e["duration_ms"] = dur_ms
        if exc:
            e["error"] = str(exc)[:200]
            log_event("error", f"fail: {self.label}", **e)
        else:
            log_event("info", f"end: {self.label}", **e)
        return False


def inc_counter(name, value=1, **labels):
    # write counter increment as jsonl
    try:
        vid = labels.get("video_id") or _current_video_id()
        mdir = _metrics_dir(vid)
        fpath = os.path.join(mdir, "counters.jsonl")
        payload = {
            "ts": int(time.time()),
            "metric": name,
            "value": value,
        }
        payload.update(_default_fields(labels))
        for k, v in labels.items():
            if k not in payload:
                payload[k] = v
        with open(fpath, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def observe_histogram(name, value, **labels):
    # append observations; aggregation can be offline
    try:
        vid = labels.get("video_id") or _current_video_id()
        mdir = _metrics_dir(vid)
        fpath = os.path.join(mdir, "histograms.jsonl")
        payload = {
            "ts": int(time.time()),
            "metric": name,
            "value": float(value),
        }
        payload.update(_default_fields(labels))
        for k, v in labels.items():
            if k not in payload:
                payload[k] = v
        with open(fpath, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


