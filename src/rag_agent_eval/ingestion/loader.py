from pathlib import Path


def load_markdown_files(source_dir: str) -> list[dict]:
    """Load all .md files from a directory, return list of {path, text}."""
    docs = []
    for path in Path(source_dir).rglob("*.md"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            docs.append({"path": str(path), "text": text})
    return docs
