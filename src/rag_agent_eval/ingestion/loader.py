from pathlib import Path

_SUPPORTED_EXTENSIONS = {".md"}


def load_documents(source_dir: str) -> list[dict]:
    """Load documents from a directory. Dispatches by file extension.

    To add a new format, write a _load_X() function and register its
    extension in _LOADERS.
    """
    docs = []
    for path in Path(source_dir).rglob("*"):
        if path.suffix.lower() in _LOADERS:
            docs.extend(_LOADERS[path.suffix.lower()](path))
    return docs


def _load_markdown(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return []
    return [{"path": str(path), "text": text}]


_LOADERS = {
    ".md": _load_markdown,
}
