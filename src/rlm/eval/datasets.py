"""OOLONG benchmark dataset loading from HuggingFace."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class EvalTask(BaseModel):
    """A single evaluation task from the OOLONG benchmark."""
    id: int
    context_window_id: int
    context_text: str
    question: str
    answer: str
    answer_type: str
    task_group: str
    task: str
    dataset: str
    context_len: int


def download_oolong_synth(cache_dir: Path | None = None) -> Path:
    """Download the oolong-synth dataset from HuggingFace.

    Args:
        cache_dir: Directory to cache the dataset. Defaults to ~/.cache/rlm-nix-wasm/datasets/.

    Returns:
        Path to the cached dataset directory.
    """
    from datasets import load_dataset  # type: ignore[import-untyped]

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "rlm-nix-wasm" / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("oolongbench/oolong-synth", cache_dir=str(cache_dir))
    return cache_dir / "oolong-synth" if ds is not None else cache_dir


def load_oolong_synth_tasks(
    dataset_name: str = "trec_coarse",
    context_len: int = 65536,
    split: str = "validation",
    cache_dir: Path | None = None,
) -> list[EvalTask]:
    """Load OOLONG-synth tasks filtered by dataset and context length.

    Args:
        dataset_name: Dataset to filter (e.g., "trec_coarse").
        context_len: Context window length to filter.
        split: HuggingFace split name. trec_coarse lives in "validation".
        cache_dir: Optional cache directory for the dataset.

    Returns:
        List of EvalTask models, grouped by context_window_id.
    """
    from datasets import load_dataset

    kwargs: dict[str, str] = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)

    ds = load_dataset("oolongbench/oolong-synth", split=split, **kwargs)

    tasks: list[EvalTask] = []
    for i, row in enumerate(ds):
        if row["dataset"] != dataset_name:
            continue
        if row["context_len"] != context_len:
            continue

        # Strip "ANSWER_TYPE." prefix from answer_type (e.g., "ANSWER_TYPE.LABEL" -> "LABEL")
        answer_type = row["answer_type"]
        if answer_type.startswith("ANSWER_TYPE."):
            answer_type = answer_type[len("ANSWER_TYPE."):]

        tasks.append(EvalTask(
            id=i,
            context_window_id=row["context_window_id"],
            context_text=row["context_window_text"],
            question=row["question"],
            answer=row["answer"],
            answer_type=answer_type,
            task_group=row["task_group"],
            task=row["task"],
            dataset=row["dataset"],
            context_len=row["context_len"],
        ))

    # Sort by context_window_id so same-context tasks run together (cache reuse)
    tasks.sort(key=lambda t: (t.context_window_id, t.id))
    return tasks
