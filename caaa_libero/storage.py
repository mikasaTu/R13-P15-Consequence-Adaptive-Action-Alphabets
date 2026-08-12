"""Atomic, hash-checked storage helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile

import numpy as np


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_tree(root, exclude=(".git", "__pycache__")):
    digest = hashlib.sha256()
    root = os.path.abspath(root)
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in exclude)
        for name in sorted(files):
            if name.endswith((".pyc", ".pyo")):
                continue
            path = os.path.join(current, name)
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            digest.update(relative.encode("utf-8") + b"\0")
            with open(path, "rb") as handle:
                while True:
                    block = handle.read(1024 * 1024)
                    if not block:
                        break
                    digest.update(block)
            digest.update(b"\0")
    return digest.hexdigest()


def atomic_json(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + os.path.basename(path), dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_text(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + os.path.basename(path), dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_npz(path, **arrays):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + os.path.basename(path), suffix=".npz", dir=os.path.dirname(path))
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def mark_complete(payload_path, metadata=None):
    metadata = dict(metadata or {})
    metadata.update(
        {
            "payload": os.path.basename(payload_path),
            "payload_bytes": os.path.getsize(payload_path),
            "payload_sha256": sha256_file(payload_path),
            "complete": True,
        }
    )
    marker = payload_path + ".complete.json"
    atomic_json(marker, metadata)
    return marker


def validate_complete(payload_path):
    marker = payload_path + ".complete.json"
    if not os.path.isfile(payload_path) or not os.path.isfile(marker):
        return False, "missing_payload_or_marker"
    try:
        with open(marker, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except Exception as error:
        return False, "invalid_marker:%s" % (error,)
    if not metadata.get("complete"):
        return False, "marker_not_complete"
    if metadata.get("payload_bytes") != os.path.getsize(payload_path):
        return False, "size_mismatch"
    if metadata.get("payload_sha256") != sha256_file(payload_path):
        return False, "sha256_mismatch"
    return True, metadata

