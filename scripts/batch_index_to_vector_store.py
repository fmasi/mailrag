#!/usr/bin/env python3
"""Batch index all emails from Azure Blob to the configured vector store.

Processes emails in batches to avoid exceeding disk/memory limits.
A checkpoint file is maintained so the script can resume after interruption.

Usage:
    python scripts/batch_index_to_vector_store.py
    python scripts/batch_index_to_vector_store.py --time-limit 3h
    python scripts/batch_index_to_vector_store.py --time-limit 90m
    python scripts/batch_index_to_vector_store.py --time-limit 5400s
    python scripts/batch_index_to_vector_store.py --time-limit 5400

The --time-limit flag stops the script before starting a new batch when adding
another batch would exceed the wall-clock budget.  The current batch always
finishes.  The checkpoint is written after every batch, so re-running the script
automatically resumes from where it left off.
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

# Ensure the project root is on ``sys.path`` so ``src`` is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config.settings import RAGConfig  # noqa: E402
from src.data.loaders.mail_archive_x import MailArchiveXLoader  # noqa: E402
from src.data.noise_filter import NoiseFilter  # noqa: E402
from src.ingest import selection  # noqa: E402

# Source-agnostic folder-selection logic lives in src/ingest/selection.py so the
# same prefix / level-2 / direct-files rules drive both Azure blob names and
# local .eml directory trees. Re-export under the historical private names this
# script (and tests/test_batch_index_selection.py) reference.
_normalize_prefix = selection.normalize_prefix
_blob_matches_rule = selection.matches_rule
_discover_blob_structure = selection.discover_structure
_require_questionary = selection._require_questionary
_prompt_guided_selection = selection.prompt_guided_selection

CHECKPOINT_FILE = os.path.join(os.path.dirname(__file__), ".vector_batch_checkpoint.txt")
CHECKPOINT_VERSION = 2


def _parse_time_limit(value: str) -> float:
    """Parse a human-readable duration string into seconds.

    Accepted formats:
        3h    → 10800.0
        90m   → 5400.0
        5400s → 5400.0
        5400  → 5400.0  (bare number treated as seconds)

    Raises:
        ValueError: if the string cannot be parsed.
    """
    value = value.strip()
    try:
        if value.endswith("h"):
            return float(value[:-1]) * 3600
        if value.endswith("m"):
            return float(value[:-1]) * 60
        if value.endswith("s"):
            return float(value[:-1])
        return float(value)
    except ValueError:
        raise ValueError(
            f"Invalid time limit {value!r}. "
            "Use formats like '3h', '90m', '5400s', or a plain number of seconds."
        )


def _time_budget_exhausted(elapsed: float, batch_times: list, limit_secs: float) -> bool:
    """Return True if starting another batch would likely exceed the time limit.

    The estimate uses the mean of all completed batch durations.  When no
    batches have completed yet (first batch) this always returns False — at
    least one batch always runs regardless of the time limit.

    Args:
        elapsed:     Seconds since the run started (monotonic clock).
        batch_times: Duration in seconds of each previously completed batch.
        limit_secs:  Total allowed run time in seconds.
    """
    if not batch_times:
        return False
    mean_batch = sum(batch_times) / len(batch_times)
    return elapsed + mean_batch >= limit_secs


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as a compact human-readable string.

    Examples: 45 → '45s', 90 → '1m 30s', 3661 → '1h 01m 01s'
    """
    secs = int(seconds)
    h, remainder = divmod(secs, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _selection_label(rule: dict) -> str:
    """Return a human-readable label for a saved selection rule."""
    rule_type = rule["type"]
    if rule_type == "prefix":
        return rule["value"] or "[entire container]"
    if rule_type == "direct-root-files":
        return f"{rule['root']} [direct files only]"
    if rule_type == "container-root":
        return "[container root files]"
    return str(rule)


def _selection_lookup(
    selection_rules: list[dict],
) -> tuple[set[str], set[str], bool]:
    """Return lookup sets for selected prefixes and direct-file rules."""
    selected_prefixes = {rule["value"] for rule in selection_rules if rule["type"] == "prefix"}
    selected_direct_roots = {
        rule["root"] for rule in selection_rules if rule["type"] == "direct-root-files"
    }
    include_container_root = any(rule["type"] == "container-root" for rule in selection_rules)
    return selected_prefixes, selected_direct_roots, include_container_root


def _render_selection_tree(
    folder_tree: dict[str, dict],
    has_container_root_files: bool,
    selection_rules: list[dict],
) -> list[str]:
    """Render a tree-style selection summary for guided mode."""
    selected_prefixes, selected_direct_roots, include_container_root = _selection_lookup(
        selection_rules
    )
    lines = ["  [x] selected  [~] partial  [ ] skipped"]

    if has_container_root_files:
        marker = "[x]" if include_container_root else "[ ]"
        lines.append(f"  {marker} [container root files]")

    for root in sorted(folder_tree):
        entry = folder_tree[root]
        child_prefixes = sorted(entry["children"])
        root_selected = root in selected_prefixes
        direct_selected = root in selected_direct_roots
        child_states = [child in selected_prefixes for child in child_prefixes]
        selected_items = int(direct_selected) + sum(int(state) for state in child_states)
        total_items = int(entry["has_direct_files"]) + len(child_prefixes)

        if root_selected:
            root_marker = "[x]"
        elif selected_items == 0:
            root_marker = "[ ]"
        elif selected_items == total_items:
            root_marker = "[x]"
        else:
            root_marker = "[~]"

        lines.append(f"  {root_marker} {root}")

        child_items: list[tuple[str, str]] = []
        if entry["has_direct_files"]:
            direct_marker = "[x]" if root_selected or direct_selected else "[ ]"
            child_items.append((direct_marker, f"{root} [direct files only]"))

        for child_prefix in child_prefixes:
            child_marker = "[x]" if root_selected or child_prefix in selected_prefixes else "[ ]"
            child_items.append((child_marker, child_prefix))

        child_lines: list[str] = []
        for index, (child_marker, child_label) in enumerate(child_items):
            connector = "`--" if index == len(child_items) - 1 else "|--"
            child_lines.append(f"      {connector} {child_marker} {child_label}")

        lines.extend(child_lines)

    return lines


def _read_checkpoint_state() -> dict | None:
    """Return persisted checkpoint state, if any.

    Supports the legacy plain-text format where the file only stored the last
    indexed blob name.
    """
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            contents = f.read().strip()
        if not contents:
            return None
        try:
            return json.loads(contents)
        except json.JSONDecodeError:
            return {
                "version": 1,
                "last_blob_name": contents,
            }
    return None


def _write_checkpoint_state(state: dict) -> None:
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def _remove_checkpoint() -> None:
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)


def _build_checkpoint_state(mode: str, selection_rules: list[dict]) -> dict:
    return {
        "version": CHECKPOINT_VERSION,
        "mode": mode,
        "selection_rules": selection_rules,
        "last_blob_name": "",
    }


def _selection_signature(mode: str, selection_rules: list[dict]) -> str:
    """Build a stable signature for comparing requested and saved selection."""
    payload = {
        "mode": mode,
        "selection_rules": selection_rules,
    }
    return json.dumps(payload, sort_keys=True)


def _filter_blobs_by_selection(blobs: list, selection_rules: list[dict]) -> list:
    """Filter Azure blob objects using the current selection rules."""
    return [
        blob
        for blob in blobs
        if any(_blob_matches_rule(blob.name, rule) for rule in selection_rules)
    ]


def _confirm_guided_selection(
    folder_tree: dict[str, dict],
    has_container_root_files: bool,
    selection_rules: list[dict],
    matched_total: int,
) -> bool:
    """Print selection summary and ask for final confirmation."""
    questionary = _require_questionary()
    print("\nSelection summary:")
    for line in _render_selection_tree(
        folder_tree,
        has_container_root_files,
        selection_rules,
    ):
        print(line)
    print(f"  Matched {matched_total} .eml blob(s)")
    confirmed = questionary.confirm(
        "Start indexing with this selection?",
        default=True,
    ).ask()
    if confirmed is None:
        raise KeyboardInterrupt
    return confirmed


def _download_blob(container_client, blob, temp_dir: str) -> str:
    """Download a single blob into the temp directory and return its local path."""
    blob_client = container_client.get_blob_client(blob.name)
    local_path = os.path.join(temp_dir, blob.name)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as handle:
        handle.write(blob_client.download_blob().readall())
    return local_path


def _purge_noise_source_blobs(container_client, noise_emails: list) -> tuple[int, int]:
    """
    Delete the source .eml blobs for noise emails from Azure Blob Storage.

    source_id values are local paths like '/tmp/tmpXXX/folder/file.eml'; the
    blob name is everything after the leading '/tmp/<dir>/' prefix.

    Returns (purged, errors).
    """
    blob_paths = [re.sub(r"^/tmp/[^/]+/", "", e.source_id) for e in noise_emails if e.source_id]
    errors = 0
    for path in blob_paths:
        try:
            container_client.delete_blob(path)
        except Exception as exc:
            print(f"  Warning: could not delete blob '{path}': {exc}")
            errors += 1
    purged = len(blob_paths) - errors
    print(f"  Purged {purged}/{len(blob_paths)} noise blob(s) from source.")
    return purged, errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--time-limit",
        metavar="DURATION",
        default=None,
        help=(
            "Stop before starting a new batch if doing so would exceed this "
            "wall-clock budget. Accepts: 3h, 90m, 5400s, or a plain number of "
            "seconds. The current batch always completes. Checkpoint is saved "
            "after every batch so the next run resumes cleanly."
        ),
    )
    parser.add_argument(
        "--prefix",
        metavar="PREFIX",
        default=None,
        help=(
            "Legacy mode: recursively index everything under this blob prefix. "
            "Without this flag, the script starts with guided folder selection."
        ),
    )
    parser.add_argument(
        "--max-batches",
        metavar="N",
        type=int,
        default=None,
        help=(
            "Stop after processing at most N batches. Useful for benchmarking "
            "throughput (e.g. --max-batches 1) without waiting for a full run. "
            "Checkpoint is saved after each batch so the next run resumes cleanly."
        ),
    )
    parser.add_argument(
        "--purge-source",
        action="store_true",
        help=(
            "Delete noise emails from Azure Blob Storage at index time. "
            "When the noise filter blocks an email, its .eml file is immediately "
            "deleted from the source container. Has no effect if no noise rules "
            "are loaded or no emails are filtered. Use with care — deletions are "
            "irreversible."
        ),
    )
    args = parser.parse_args()

    time_limit_secs: float | None = None
    if args.time_limit:
        time_limit_secs = _parse_time_limit(args.time_limit)
    max_batches: int | None = args.max_batches

    load_dotenv()
    batch_size = int(os.environ.get("RAG_INDEX_BATCH_SIZE", "200"))
    download_workers = int(os.environ.get("RAG_DOWNLOAD_WORKERS", "8"))
    # Indexing needs embeddings, but not an LLM client.
    RAGConfig.initialize_settings(include_llm=False)

    # Load noise filter once — applied to every batch before embedding
    noise_filter = NoiseFilter.from_project_rules()
    if noise_filter.is_empty():
        print("No noise rules loaded — all emails will be indexed.")
    else:
        print(f"Noise filter active: {noise_filter.category_names()}")

    # Lazy imports so the script fails fast on missing env vars above
    from azure.storage.blob import BlobServiceClient

    from src.storage.persist import StorageManager

    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is not set")
    container_name = os.environ.get("AZURE_BLOB_CONTAINER", "eml-archive")

    service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = service_client.get_container_client(container_name)

    container_blobs = [b for b in container_client.list_blobs() if b.name.endswith(".eml")]
    blob_names = [blob.name for blob in container_blobs]
    folder_tree: dict[str, dict] = {}
    has_container_root_files = False

    checkpoint_state = _read_checkpoint_state()
    if args.prefix is not None:
        mode = "prefix"
        selection_rules = [{"type": "prefix", "value": _normalize_prefix(args.prefix)}]
    else:
        mode = "guided"
        selection_rules = []

    checkpoint = ""
    if checkpoint_state:
        checkpoint = checkpoint_state.get("last_blob_name", "")
        if checkpoint_state.get("version") == 1:
            legacy_prefix = _normalize_prefix(
                args.prefix or os.environ.get("AZURE_BLOB_PREFIX", "")
            )
            mode = "prefix"
            selection_rules = [{"type": "prefix", "value": legacy_prefix}]
        else:
            saved_mode = checkpoint_state["mode"]
            saved_rules = checkpoint_state["selection_rules"]
            if args.prefix is not None:
                requested_signature = _selection_signature(mode, selection_rules)
                saved_signature = _selection_signature(saved_mode, saved_rules)
                if requested_signature != saved_signature:
                    raise ValueError(
                        "Checkpoint exists for a different indexing selection. "
                        "Resume without --prefix, or clear the checkpoint before starting a new run."
                    )
            mode = saved_mode
            selection_rules = saved_rules
    elif args.prefix is None:
        mode = "guided"

    if mode == "guided":
        folder_tree, has_container_root_files = _discover_blob_structure(blob_names)

    if mode == "guided" and not checkpoint_state:
        selection_rules = _prompt_guided_selection(folder_tree, has_container_root_files)
        if not selection_rules:
            print("No folders selected. Nothing to index.")
            return

    all_blobs = _filter_blobs_by_selection(container_blobs, selection_rules)
    total = len(all_blobs)

    if mode == "guided" and not checkpoint_state:
        if not _confirm_guided_selection(
            folder_tree,
            has_container_root_files,
            selection_rules,
            total,
        ):
            print("Aborted before indexing.")
            return

    if checkpoint_state:
        print("Resuming saved selection:")
        if mode == "guided":
            for line in _render_selection_tree(
                folder_tree,
                has_container_root_files,
                selection_rules,
            ):
                print(line)
        else:
            for label in (_selection_label(rule) for rule in selection_rules):
                print(f"  - {label}")
    elif mode == "prefix":
        print(f"Using legacy prefix mode: {_selection_label(selection_rules[0])}")

    print(f"Found {total} .eml blobs in '{container_name}' after selection")
    print(
        "Indexing config: "
        f"batch_size={batch_size}, "
        f"download_workers={download_workers}, "
        f"embedding_batch_size={RAGConfig.EMBEDDING_BATCH_SIZE}, "
        f"embedding_workers={RAGConfig.EMBEDDING_NUM_WORKERS}"
    )
    if time_limit_secs is not None:
        print(f"Time limit: {_fmt_duration(time_limit_secs)}")

    if total == 0:
        print("No matching blobs found. Nothing to index.")
        return

    checkpoint_state = _build_checkpoint_state(mode, selection_rules)
    _write_checkpoint_state(checkpoint_state)

    # Resume from checkpoint
    if checkpoint:
        skip = 0
        for i, b in enumerate(all_blobs):
            if b.name == checkpoint:
                skip = i + 1
                break
        all_blobs = all_blobs[skip:]
        if skip:
            print(f"Resuming after checkpoint '{checkpoint}' — skipping {skip} blobs")
        else:
            print(
                f"Checkpoint '{checkpoint}' was not found in the current selection; "
                "starting from the beginning of the saved selection."
            )

    indexed = total - len(all_blobs)
    total_batches = (len(all_blobs) + batch_size - 1) // batch_size

    run_start = time.monotonic()
    batch_times: list[float] = []

    for batch_start in range(0, len(all_blobs), batch_size):
        batch_number = (batch_start // batch_size) + 1

        # --- Max-batches pre-check ---
        if max_batches is not None and batch_number > max_batches:
            print(
                f"\n--max-batches {max_batches} reached. "
                f"Checkpoint saved at '{_read_checkpoint_state()['last_blob_name']}'. Re-run to continue."
            )
            break

        # --- Time-limit pre-check (runs before every batch except the first) ---
        if time_limit_secs is not None:
            elapsed = time.monotonic() - run_start
            if _time_budget_exhausted(elapsed, batch_times, time_limit_secs):
                mean = sum(batch_times) / len(batch_times)
                print(
                    f"\nTime limit: stopping before batch {batch_number}/{total_batches}."
                    f" Elapsed {_fmt_duration(elapsed)} of {_fmt_duration(time_limit_secs)},"
                    f" mean batch {_fmt_duration(mean)} — next batch would exceed limit."
                    f"\nCheckpoint saved at '{_read_checkpoint_state()['last_blob_name']}'. Re-run to continue."
                )
                break

        batch_blobs = all_blobs[batch_start : batch_start + batch_size]
        print(f"\nBatch {batch_number}/{total_batches}: downloading {len(batch_blobs)} blobs...")

        batch_wall_start = time.monotonic()

        with tempfile.TemporaryDirectory() as temp_dir:
            if download_workers <= 1:
                for blob in batch_blobs:
                    _download_blob(container_client, blob, temp_dir)
            else:
                with ThreadPoolExecutor(max_workers=download_workers) as executor:
                    list(
                        executor.map(
                            lambda current_blob: _download_blob(
                                container_client, current_blob, temp_dir
                            ),
                            batch_blobs,
                        )
                    )

            loader = MailArchiveXLoader(temp_dir)
            emails = loader.load()

        clean_emails = [e for e in emails if not noise_filter.is_noise(e)]
        noise_emails = [e for e in emails if noise_filter.is_noise(e)]
        filtered_count = len(noise_emails)
        if filtered_count:
            print(f"  Noise filter: skipped {filtered_count}/{len(emails)} emails")
            if args.purge_source:
                _purge_noise_source_blobs(container_client, noise_emails)

        documents = [
            email.to_document(doc_id=f"{email.source}_{indexed + i}")
            for i, email in enumerate(clean_emails)
        ]

        _, ingest_stats = StorageManager.create_and_save_index(
            documents,
            verbose=True,
            return_stats=True,
        )

        last_blob = batch_blobs[-1].name
        checkpoint_state["last_blob_name"] = last_blob
        _write_checkpoint_state(checkpoint_state)
        indexed += len(batch_blobs)

        batch_elapsed = time.monotonic() - batch_wall_start
        batch_times.append(batch_elapsed)

        elapsed = time.monotonic() - run_start
        mean = sum(batch_times) / len(batch_times)
        remaining_batches = total_batches - batch_number
        pct = 100 * batch_number // total_batches
        eta = _fmt_duration(remaining_batches * mean)

        progress = (
            f"[{pct}%] batch {batch_number}/{total_batches} | "
            f"docs {indexed}/{total} | "
            f"batch {_fmt_duration(batch_elapsed)} | "
            f"elapsed {_fmt_duration(elapsed)} | "
            f"ETA ~{eta}"
        )
        if time_limit_secs is not None:
            budget_remaining = time_limit_secs - elapsed
            est_more = max(0, int(budget_remaining / mean))
            progress += (
                f" | budget {_fmt_duration(elapsed)}/{_fmt_duration(time_limit_secs)}"
                f" (~{est_more} more batch(es))"
            )
        if ingest_stats is not None and ingest_stats.get("combined_secs", 0.0) > 0.0:
            embed_secs = ingest_stats["embed_secs"]
            upload_secs = ingest_stats["upload_secs"]
            combined_secs = ingest_stats["combined_secs"]
            embed_pct = 100.0 * embed_secs / combined_secs
            upload_pct = 100.0 * upload_secs / combined_secs
            progress += (
                " | stage split "
                f"embed {embed_pct:.1f}%/{_fmt_duration(embed_secs)} "
                f"upload {upload_pct:.1f}%/{_fmt_duration(upload_secs)}"
            )
        print(progress)

    print(f"\nDone — {indexed}/{total} documents indexed to {RAGConfig.VECTOR_STORE_PROVIDER}.")
    if indexed == total:
        _remove_checkpoint()


if __name__ == "__main__":
    main()
