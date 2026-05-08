"""Public Figshare helpers for BOLD5000 Release 2.0.

These utilities intentionally use the public article endpoint only. They do not
require an API token, account, Kaggle login, Google Drive login, or paid service.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import re
import shutil

import pandas as pd
import requests
from tqdm import tqdm


FIGSHARE_API = "https://api.figshare.com/v2/articles/{article_id}"


@dataclass(frozen=True)
class FigshareFile:
    """One downloadable Figshare file entry."""

    id: int
    name: str
    size: int
    download_url: str

    @property
    def size_gb(self) -> float:
        return self.size / (1024 ** 3)


def fetch_article_metadata(article_id: int, timeout: int = 60) -> dict[str, Any]:
    """Fetch public Figshare article metadata without authentication."""
    url = FIGSHARE_API.format(article_id=article_id)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def parse_figshare_files(article: dict[str, Any]) -> list[FigshareFile]:
    """Parse the ``files`` array from a Figshare article payload."""
    files: list[FigshareFile] = []
    for item in article.get("files", []) or []:
        url = item.get("download_url") or item.get("downloadUrl") or item.get("url_private_api") or ""
        files.append(
            FigshareFile(
                id=int(item.get("id", 0)),
                name=str(item.get("name", "")),
                size=int(item.get("size", 0) or 0),
                download_url=str(url),
            )
        )
    return files


def files_to_dataframe(files: Iterable[FigshareFile]) -> pd.DataFrame:
    """Convert Figshare file records to a display/save table."""
    rows = [
        {
            "id": f.id,
            "name": f.name,
            "size_bytes": f.size,
            "size_gb": round(f.size_gb, 4),
            "download_url": f.download_url,
        }
        for f in files
    ]
    return pd.DataFrame(rows, columns=["id", "name", "size_bytes", "size_gb", "download_url"])


def filter_files(
    files: Iterable[FigshareFile],
    regex: str | None = None,
    subjects: list[str] | tuple[str, ...] | None = None,
) -> list[FigshareFile]:
    """Filter files by regex and subject tokens such as ``CSI1``.

    Subject filtering is intentionally permissive. A file is kept if any subject
    token appears anywhere in the file name.
    """
    selected = list(files)
    if regex:
        pat = re.compile(regex, flags=re.IGNORECASE)
        selected = [f for f in selected if pat.search(f.name)]
    if subjects:
        subject_tokens = [s.lower() for s in subjects]
        selected = [f for f in selected if any(s in f.name.lower() for s in subject_tokens)]
    return selected


def ensure_free_space(path: str | Path, required_gb: float) -> None:
    """Raise a clean error if the filesystem has less than required space."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(target).free / (1024 ** 3)
    if free_gb < required_gb:
        raise RuntimeError(
            f"Not enough free disk space in {target}. Required {required_gb:.2f} GB, available {free_gb:.2f} GB."
        )


def download_file(file: FigshareFile, output_dir: str | Path, chunk_size: int = 1024 * 1024) -> Path:
    """Download one public Figshare file with resume-skipping if already complete."""
    if not file.download_url:
        raise ValueError(f"No download_url present for Figshare file {file.name}")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / file.name
    if out_path.exists() and out_path.stat().st_size == file.size:
        return out_path
    with requests.get(file.download_url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", file.size or 0))
        with out_path.open("wb") as fh, tqdm(total=total, unit="B", unit_scale=True, desc=file.name) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)
                    bar.update(len(chunk))
    return out_path


def download_selected(files: Iterable[FigshareFile], output_dir: str | Path, min_free_gb: float = 1.0) -> list[Path]:
    """Download selected Figshare files after checking available disk space."""
    files = list(files)
    required_gb = max(min_free_gb, sum(f.size_gb for f in files) + 1.0)
    ensure_free_space(output_dir, required_gb)
    return [download_file(file, output_dir) for file in files]
