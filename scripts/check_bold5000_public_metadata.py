from __future__ import annotations

from pathlib import Path
import json

from fmri_testing.config import ensure_dirs, load_config
from fmri_testing.data.figshare import fetch_article_metadata, files_to_dataframe, filter_files, parse_figshare_files


def main() -> None:
    cfg = load_config("configs/default.yaml")
    ensure_dirs(cfg)
    article_id = int(cfg["data"].get("figshare_article_id", 14456124))
    article = fetch_article_metadata(article_id)
    files = parse_figshare_files(article)
    df = files_to_dataframe(files)
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "bold5000_file_manifest.csv"
    json_path = out / "bold5000_file_manifest.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({"article": {"id": article_id, "title": article.get("title")}, "files": df.to_dict(orient="records")}, indent=2), encoding="utf-8")

    preferred = filter_files(files, regex="TYPED-FITHRF-GLMDENOISE-RR")
    csi1 = filter_files(files, subjects=["CSI1"])
    stim = filter_files(files, regex="stim|image|scene")

    total_gb = float(df["size_gb"].sum()) if len(df) else 0.0
    print(f"Article title: {article.get('title')}")
    print(f"Article id: {article_id}")
    print(f"Number of files: {len(files)}")
    print(f"Total size, GB: {total_gb:.2f}")
    print("\nAll files:")
    for file in files:
        print(f"- {file.name} ({file.size_gb:.2f} GB)")
    print("\nMatching TYPED-FITHRF-GLMDENOISE-RR:")
    for file in preferred:
        print(f"- {file.name} ({file.size_gb:.2f} GB)")
    print("\nMatching CSI1:")
    for file in csi1:
        print(f"- {file.name} ({file.size_gb:.2f} GB)")
    print("\nPotential stimulus/image files:")
    for file in stim:
        print(f"- {file.name} ({file.size_gb:.2f} GB)")
    print(f"\nWrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
