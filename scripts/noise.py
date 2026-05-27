#!/usr/bin/env python3
"""Noise management tool — discover, purge and deep-clean noisy emails.

Subcommands
-----------
  discover     Scan the Qdrant index for unknown sender domains and classify
               them with the LLM.  Runs in interactive mode by default:
               for each domain you are shown the LLM verdict and prompted:

                 [y] Add noise rule    [n] Skip (this run)    [w] Whitelist (never re-propose)
                 [2] Deep inspect      [3] Read an email

               Use --auto to write rules without prompting (original behaviour).
               Dedicated domains get a sender_domains rule; general-purpose
               domains (gmail, outlook …) get narrow sender_patterns /
               subject_patterns rules when possible.
               Results are merged into config/noise_rules.yaml — no duplicate
               keys are ever created.  Review via git diff, commit to approve.

  purge        Delete emails that match the current noise_rules.yaml from
               Qdrant and optionally from Azure Blob Storage.

  deep-clean   For general-purpose domains where no reliable rule could be
               created, classify each email individually with the LLM and
               delete confirmed noise from Qdrant and Azure Blob.  If the LLM
               can extract a reusable pattern after the batch it is also merged
               into noise_rules.yaml.

Usage
-----
  python scripts/noise.py discover                        # interactive (default)
  python scripts/noise.py discover --auto                 # auto-write rules, no prompts
  python scripts/noise.py discover --auto --deep-clean    # auto discover + deep-clean
  python scripts/noise.py discover --dry-run
  python scripts/noise.py purge
  python scripts/noise.py purge --dry-run
  python scripts/noise.py deep-clean
  python scripts/noise.py deep-clean --domain gmail.com
  python scripts/noise.py deep-clean --dry-run
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

_RULES_PATH     = Path(__file__).resolve().parent.parent / "config" / "noise_rules.yaml"
_WHITELIST_PATH = Path(__file__).resolve().parent.parent / "config" / "whitelist_domains.yaml"

# Domains shared by real people and businesses — blanket domain rules here
# cause massive false positives.  The LLM is asked for narrow patterns instead.
_GENERAL_PURPOSE_DOMAINS = frozenset({
    "gmail.com", "googlemail.com",
    "outlook.com", "hotmail.com", "hotmail.fr", "hotmail.co.uk",
    "live.com", "live.fr", "live.co.uk",
    "yahoo.com", "yahoo.fr", "yahoo.co.uk", "yahoo.ca", "yahoo.de",
    "microsoft.com",
    "icloud.com", "me.com", "mac.com",
    "protonmail.com", "pm.me",
    "aol.com", "msn.com",
})

_DISCOVER_MIN_EMAILS  = 10   # default minimum unique emails to consider a domain
_DEEP_CLEAN_BATCH     = 10   # emails per LLM classification call
_LLM_BODY_CHARS       = int(os.getenv("NOISE_LLM_BODY_CHARS", "800"))  # chars sent to LLM per email
_MIN_EMAILS_FOR_RULE  = 5    # minimum noise emails before attempting rule extraction
_DEFAULT_INSPECT_COUNT = 10  # emails fetched for interactive deep-inspect / read


# ── Shared helpers ────────────────────────────────────────────────────────────

def _extract_domain(sender: str) -> str | None:
    m = re.search(r"@([\w.\-]+)", sender.lower())
    return m.group(1) if m else None


def _domain_to_key(domain: str) -> str:
    """dots and hyphens → underscores, guarantees a valid YAML key."""
    return re.sub(r"[.\-]", "_", domain)


def _blob_path(source_id: str) -> str:
    return re.sub(r"^/tmp/[^/]+/", "", source_id)


def _extract_body(payload: dict) -> str:
    """
    Extract the email body text from a Qdrant point payload.

    LlamaIndex stores the node text in '_node_content' as a JSON string
    (not as a top-level payload field), so we parse that first.  Direct
    'body'/'text'/'content' keys are checked as a fallback for any manually
    constructed payloads.
    """
    node_content = payload.get("_node_content", "")
    if node_content:
        try:
            node_data = json.loads(node_content)
            text = node_data.get("text") or node_data.get("content") or ""
            if text:
                return text
        except Exception:
            pass
    return payload.get("body", "") or payload.get("text", "") or payload.get("content", "")


def _confirm(prompt: str) -> bool:
    try:
        answer = input(f"\n{prompt} [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")


# ── Whitelist helpers ─────────────────────────────────────────────────────────

def _load_whitelist() -> frozenset:
    """Return the set of domains in config/whitelist_domains.yaml."""
    import yaml
    try:
        with open(_WHITELIST_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return frozenset(d.lower() for d in (data.get("domains") or []))
    except FileNotFoundError:
        return frozenset()
    except Exception as exc:
        print(f"Warning: could not load whitelist: {exc}")
        return frozenset()


def _save_whitelist_domain(domain: str) -> None:
    """Append a domain to config/whitelist_domains.yaml (idempotent)."""
    import yaml
    try:
        try:
            with open(_WHITELIST_PATH, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            data = {}
        domains = list(data.get("domains") or [])
        if domain.lower() not in [d.lower() for d in domains]:
            domains.append(domain.lower())
        with open(_WHITELIST_PATH, "w", encoding="utf-8") as f:
            f.write("# Domains confirmed as legitimate — excluded from noise discovery.\n")
            f.write("# Managed by: python scripts/noise.py discover  (choose [w] at the prompt)\n")
            f.write("# To un-whitelist a domain: delete its line below and commit.\n\n")
            yaml.dump({"domains": domains}, f, default_flow_style=False, allow_unicode=True)
        print(f"  Whitelisted '{domain}' — saved to config/whitelist_domains.yaml")
    except Exception as exc:
        print(f"  Warning: could not save whitelist: {exc}")


# ── YAML state ────────────────────────────────────────────────────────────────

def _load_existing_state() -> tuple[set, set]:
    """Return (category_keys, sender_domains) currently in noise_rules.yaml."""
    import yaml
    try:
        with open(_RULES_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cats = data.get("categories") or {}
        keys = set(cats.keys())
        domains: set = set()
        for cfg in cats.values():
            for d in (cfg or {}).get("sender_domains", []):
                domains.add(d.lower())
        return keys, domains
    except Exception:
        return set(), set()


def _merge_rules_to_yaml(new_rules: list) -> int:
    """
    Append new rules to noise_rules.yaml, skipping any whose domain is already
    covered and ensuring every key is unique.  Returns the number written.
    """
    existing_keys, existing_domains = _load_existing_state()
    to_write = []

    for rule in new_rules:
        domain = rule.get("domain", "")
        if domain.lower() in existing_domains:
            continue
        base_key = _domain_to_key(domain)
        key, suffix = base_key, 2
        while key in existing_keys:
            key = f"{base_key}_{suffix}"
            suffix += 1
        rule = {**rule, "key": key}
        existing_keys.add(key)
        existing_domains.add(domain.lower())
        to_write.append(rule)

    if not to_write:
        return 0

    with open(_RULES_PATH, "a", encoding="utf-8") as f:
        for rule in to_write:
            f.write(f"\n  {rule['key']}:\n")
            f.write(f"    description: {rule['description']}\n")
            if rule.get("sender_domains"):
                f.write("    sender_domains:\n")
                for d in rule["sender_domains"]:
                    f.write(f"      - {d}\n")
            if rule.get("sender_patterns"):
                f.write("    sender_patterns:\n")
                for p in rule["sender_patterns"]:
                    f.write(f"      - '{p}'\n")
            if rule.get("subject_patterns"):
                f.write("    subject_patterns:\n")
                for p in rule["subject_patterns"]:
                    f.write(f"      - '{p}'\n")

    return len(to_write)


# ── Qdrant helpers ────────────────────────────────────────────────────────────

def _scroll_all(qdrant, collection: str):
    """Yield (point_id, payload) for every point in the collection."""
    offset = None
    while True:
        results, next_offset = qdrant.scroll(
            collection_name=collection, limit=256, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for point in results:
            yield point.id, (point.payload or {})
        if next_offset is None:
            break
        offset = next_offset


def _scroll_sender_stats(qdrant, collection: str) -> dict:
    """Return per-domain stats across the full collection."""
    stats: dict = defaultdict(lambda: {
        "unique_emails": set(), "sample_senders": [], "sample_subjects": [],
    })
    offset = None
    total = 0
    while True:
        results, next_offset = qdrant.scroll(
            collection_name=collection, limit=256, offset=offset,
            with_payload=["sender", "subject", "source_id"], with_vectors=False,
        )
        for point in results:
            p = point.payload or {}
            sender = p.get("sender", "").strip()
            domain = _extract_domain(sender)
            if not domain:
                continue
            e = stats[domain]
            sid = p.get("source_id", "")
            if sid:
                e["unique_emails"].add(sid)
            if sender and len(e["sample_senders"]) < 8 and sender not in e["sample_senders"]:
                e["sample_senders"].append(sender)
            subj = p.get("subject", "").strip()
            if subj and len(e["sample_subjects"]) < 8:
                e["sample_subjects"].append(subj)
        total += len(results)
        print(f"  Scanned {total} vectors...", end="\r")
        if next_offset is None:
            break
        offset = next_offset
    print()
    return {
        domain: {
            "unique_emails": len(entry["unique_emails"]),
            "sample_senders": entry["sample_senders"],
            "sample_subjects": entry["sample_subjects"],
        }
        for domain, entry in stats.items()
    }


def _fetch_domain_email_sample(qdrant, collection: str, domain: str, n: int) -> list:
    """
    Scroll the collection and return up to n unique emails from domain,
    accumulating body text from all chunks up to _LLM_BODY_CHARS per email.
    Stops adding new source_ids once n are found (but finishes the current
    scroll batch to collect extra body chunks for already-found emails).
    """
    emails: dict = {}
    offset = None
    while True:
        results, next_offset = qdrant.scroll(
            collection_name=collection, limit=256, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for point in results:
            p = point.payload or {}
            sender = p.get("sender", "")
            if _extract_domain(sender) != domain:
                continue
            sid = p.get("source_id", "")
            if not sid:
                continue
            if sid not in emails:
                if len(emails) >= n:
                    continue  # already have enough unique emails; skip new ones
                emails[sid] = {"sender": sender, "subject": p.get("subject", ""), "body": ""}
            entry = emails[sid]
            chunk = _extract_body(p)
            if chunk and len(entry["body"]) < _LLM_BODY_CHARS:
                entry["body"] += chunk[: _LLM_BODY_CHARS - len(entry["body"])]
        if next_offset is None:
            break
        offset = next_offset
    return list(emails.values())


def _scroll_domain_emails(qdrant, collection: str, target_domains: set) -> dict:
    """
    Scroll and group chunks by source_id for the given domains.
    Returns {domain: {source_id: {sender, subject, body, point_ids}}}.
    """
    emails: dict = defaultdict(lambda: defaultdict(lambda: {
        "sender": "", "subject": "", "body": "", "point_ids": [],
    }))
    offset = None
    total = 0
    while True:
        results, next_offset = qdrant.scroll(
            collection_name=collection, limit=256, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for point in results:
            p = point.payload or {}
            sender = p.get("sender", "")
            domain = _extract_domain(sender)
            if not domain or domain not in target_domains:
                continue
            sid = p.get("source_id", "")
            if not sid:
                continue
            entry = emails[domain][sid]
            entry["point_ids"].append(point.id)
            if not entry["sender"]:
                entry["sender"] = sender
            if not entry["subject"]:
                entry["subject"] = p.get("subject", "")
            chunk = _extract_body(p)
            if chunk and len(entry["body"]) < _LLM_BODY_CHARS:
                entry["body"] += chunk[: _LLM_BODY_CHARS - len(entry["body"])]
        total += len(results)
        print(f"  Scanned {total} vectors...", end="\r")
        if next_offset is None:
            break
        offset = next_offset
    print()
    return {domain: dict(sids) for domain, sids in emails.items()}


def _delete_qdrant_points(qdrant, collection: str, point_ids: list) -> int:
    from qdrant_client.models import PointIdsList
    deleted = 0
    for start in range(0, len(point_ids), 1000):
        batch = point_ids[start: start + 1000]
        qdrant.delete(collection_name=collection, points_selector=PointIdsList(points=batch))
        deleted += len(batch)
        print(f"  Deleted {deleted}/{len(point_ids)} vectors...", end="\r")
    print()
    return deleted


def _delete_blobs(connection_string: str, container: str, blob_paths: list) -> int:
    from azure.storage.blob import BlobServiceClient
    cc = BlobServiceClient.from_connection_string(connection_string).get_container_client(container)
    deleted = errors = 0
    for i, path in enumerate(blob_paths, 1):
        try:
            cc.delete_blob(path)
            deleted += 1
        except Exception as exc:
            print(f"  Warning: could not delete '{path}': {exc}")
            errors += 1
        if i % 50 == 0:
            print(f"  Deleted {deleted}/{len(blob_paths)} blobs...", end="\r")
    print()
    if errors:
        print(f"  {errors} blob(s) could not be deleted (may already be absent).")
    return deleted


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _call_llm(client, model: str, prompt: str):
    try:
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0,
        )
        text = response.choices[0].message.content.strip()
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        return json.loads(text)
    except Exception as exc:
        print(f"  LLM error: {exc}")
        return None


def _llm_classify_dedicated(client, model: str, domain: str, subjects: list, count: int) -> dict | None:
    subjects_block = "\n".join(f"- {s[:100]}" for s in subjects)
    prompt = f"""Classify whether all emails from '{domain}' are noise for a business RAG system.

Unique emails: {count}
Sample subjects:
{subjects_block}

Noise = newsletters, automated notifications, marketing, job alerts, social media,
system digests, or anything NOT a real human business conversation.

Answer ONLY with JSON (no markdown):
{{"is_noise": true or false, "description": "one sentence if noise, else empty string"}}"""
    return _call_llm(client, model, prompt)


def _llm_classify_general_domain(client, model: str, domain: str,
                                  senders: list, subjects: list, count: int) -> dict | None:
    senders_block = "\n".join(f"- {s[:120]}" for s in senders)
    subjects_block = "\n".join(f"- {s[:100]}" for s in subjects)
    prompt = f"""Emails from '{domain}' may include real business emails — DO NOT suggest blocking the whole domain.

Unique emails: {count}
Sample senders:
{senders_block}
Sample subjects:
{subjects_block}

If these are clearly automated/noise, suggest narrow Python regex patterns:
- sender_patterns: regex against the full sender string
- subject_patterns: regex against the subject line
Patterns must be specific enough to avoid false positives on real human emails.
If no reliable narrow pattern exists, set is_noise to false.

Answer ONLY with JSON (no markdown):
{{"is_noise": true or false, "sender_patterns": [], "subject_patterns": [], "description": ""}}"""
    return _call_llm(client, model, prompt)


def _llm_classify_email_batch(client, model: str, domain: str, batch: list) -> list | None:
    lines = []
    for i, e in enumerate(batch, 1):
        preview = e["body"][:_LLM_BODY_CHARS].replace("\n", " ").strip()
        lines.append(f'Email {i}:\n  Sender: {e["sender"][:100]}\n  Subject: {e["subject"][:100]}\n  Body: {preview}')
    prompt = f"""Classify each email from '{domain}' as noise or legitimate for a business RAG system.

Noise = newsletters, automated notifications, marketing, social media, calendar spam,
or anything NOT a real human business conversation.

{chr(10).join(lines)}

Answer ONLY with a JSON array of booleans (true=noise), one per email in order.
Example for 3 emails: [true, false, true]"""
    result = _call_llm(client, model, prompt)
    if not isinstance(result, list) or len(result) != len(batch):
        return None
    return [bool(r) for r in result]


def _llm_extract_rule(client, model: str, domain: str, noise_emails: list) -> dict | None:
    examples = "\n".join(
        f'- Sender: {e["sender"][:100]}  |  Subject: {e["subject"][:80]}'
        for e in noise_emails[:20]
    )
    prompt = f"""These emails from '{domain}' were confirmed noise for a business RAG system:

{examples}

Suggest a narrow Python regex that captures these noise emails without blocking
legitimate business emails from '{domain}'.  Only suggest if confident.

Answer ONLY with JSON (no markdown):
{{"has_rule": true or false, "sender_patterns": [], "subject_patterns": [], "description": ""}}"""
    return _call_llm(client, model, prompt)


# ── Interactive discover helpers ──────────────────────────────────────────────

def _result_to_rule(domain: str, result: dict | None, is_general: bool) -> dict | None:
    """
    Convert an LLM classification result into a rule dict for _merge_rules_to_yaml.
    Returns None when no rule can be built (e.g. general-purpose domain with no patterns).
    """
    if is_general:
        patterns_s   = (result or {}).get("sender_patterns") or []
        patterns_sub = (result or {}).get("subject_patterns") or []
        if not patterns_s and not patterns_sub:
            return None
        return {
            "domain": domain,
            "description": (result or {}).get("description") or f"Noise patterns from {domain}",
            "sender_patterns":  patterns_s,
            "subject_patterns": patterns_sub,
        }
    else:
        return {
            "domain": domain,
            "description": (result or {}).get("description") or f"Noise from {domain}",
            "sender_domains": [domain],
        }


def _interactive_domain_prompt(
    domain: str,
    entry: dict,
    initial_result: dict | None,
    is_general: bool,
    qdrant,
    collection: str,
    llm,
    model: str,
    inspect_count: int,
) -> str:
    """
    Show the interactive menu for a single candidate domain.
    Returns: 'rule' | 'skip' | 'whitelist'
    """
    deep_inspect_summary = None   # (noise_n, total_n, noise_subjects, clean_subjects)
    sample_emails: list | None = None   # fetched lazily on [2] or [3]
    read_index = 0

    while True:
        print(f"\n{'═' * 62}")
        label = "[general-purpose]" if is_general else "[dedicated]"
        print(f"  Domain : {domain}  ({entry['unique_emails']} unique emails)  {label}")

        if initial_result is not None:
            verdict = "NOISE" if initial_result.get("is_noise") else "CLEAN"
            desc = initial_result.get("description", "")
            suffix = f"  — {desc}" if desc else ""
            label = "LLM (initial)" if deep_inspect_summary is not None else "LLM    "
            print(f"  {label}: {verdict}{suffix}")

        if deep_inspect_summary is not None:
            noise_n, total_n, noise_subjs, clean_subjs = deep_inspect_summary
            pct = int(100 * noise_n / total_n) if total_n else 0
            print(f"\n  Deep inspect: {noise_n}/{total_n} emails noise ({pct}%)")
            if noise_subjs:
                print(f"  Noise  : {noise_subjs[0][:65]}")
                for s in noise_subjs[1:3]:
                    print(f"           {s[:65]}")
            if clean_subjs:
                print(f"  Clean  : {clean_subjs[0][:65]}")
                for s in clean_subjs[1:3]:
                    print(f"           {s[:65]}")

        print(f"\n  Sample subjects:")
        for subj in entry["sample_subjects"][:5]:
            print(f"    - {subj[:72]}")

        print()
        print(f"  [y] Add noise rule    [n] Skip (this run)    [w] Whitelist (never re-propose)")
        print(f"  [2] Deep inspect      [3] Read an email")
        try:
            choice = input("  Choice: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return "skip"

        if choice == "y":
            return "rule"
        elif choice == "n":
            return "skip"
        elif choice == "w":
            _save_whitelist_domain(domain)
            return "whitelist"
        elif choice == "2":
            if sample_emails is None:
                print(f"\n  Fetching up to {inspect_count} emails from '{domain}'...")
                sample_emails = _fetch_domain_email_sample(qdrant, collection, domain, inspect_count)
                read_index = 0
            if not sample_emails:
                print(f"  No emails found for '{domain}' in the index.")
                continue
            print(f"  Classifying {len(sample_emails)} emails with LLM...")
            batch = [{"sender": e["sender"], "subject": e["subject"], "body": e["body"]}
                     for e in sample_emails]
            results = _llm_classify_email_batch(llm, model, domain, batch)
            if results is None:
                print("  LLM returned an unexpected response — try again.")
                continue
            noise_subjs = [sample_emails[i]["subject"] for i, r in enumerate(results) if r]
            clean_subjs = [sample_emails[i]["subject"] for i, r in enumerate(results) if not r]
            deep_inspect_summary = (sum(results), len(results), noise_subjs, clean_subjs)
            # loop back — menu re-draws with the new deep-inspect summary
        elif choice == "3":
            if sample_emails is None:
                print(f"\n  Fetching up to {inspect_count} emails from '{domain}'...")
                sample_emails = _fetch_domain_email_sample(qdrant, collection, domain, inspect_count)
                read_index = 0
            if not sample_emails:
                print(f"  No emails found for '{domain}' in the index.")
                continue
            if read_index >= len(sample_emails):
                print(f"  No more emails to show (all {len(sample_emails)} already shown).")
                continue
            # Show emails one at a time; ask after each whether to continue
            while read_index < len(sample_emails):
                email = sample_emails[read_index]
                read_index += 1
                print(f"\n  ── Email {read_index}/{len(sample_emails)} ──────────────────────────────────────────────")
                print(f"  From   : {email['sender'][:100]}")
                print(f"  Subject: {email['subject'][:100]}")
                print(f"  Body   :")
                body_lines = email["body"][:_LLM_BODY_CHARS].split("\n")
                for line in body_lines[:30]:
                    print(f"    {line[:120]}")
                if len(body_lines) > 30:
                    print(f"    [...truncated...]")
                print(f"  {'─' * 60}")
                if read_index >= len(sample_emails):
                    print("  (No more emails in sample.)")
                    break
                try:
                    see_more = input("  See another email? (y/n): ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if see_more not in ("y", "yes"):
                    break
            # fall through to re-draw the menu
        else:
            print("  Invalid choice — enter y, n, w, 2, or 3.")


# ── Subcommand: discover ──────────────────────────────────────────────────────

def cmd_discover(args, qdrant, llm, model, azure_conn_str, azure_container, collection):
    mode = "(DRY RUN) " if args.dry_run else ("(AUTO) " if args.auto else "(INTERACTIVE) ")
    print(f"\n{'=' * 60}")
    print(f"  Discover {mode}")
    print(f"  Model: {model}")
    print(f"{'=' * 60}")

    _, existing_domains = _load_existing_state()
    whitelist = _load_whitelist()

    print(f"\nScanning sender domains in '{collection}'...")
    stats = _scroll_sender_stats(qdrant, collection)

    whitelisted_skipped = [
        domain for domain, entry in stats.items()
        if entry["unique_emails"] >= args.min_emails
        and domain.lower() not in existing_domains
        and domain.lower() in whitelist
    ]
    candidates = [
        (domain, entry)
        for domain, entry in sorted(stats.items(), key=lambda x: -x[1]["unique_emails"])
        if entry["unique_emails"] >= args.min_emails
        and domain.lower() not in existing_domains
        and domain.lower() not in whitelist
    ]

    print(f"  Known/covered domains skipped.")
    if whitelisted_skipped:
        sample = ", ".join(whitelisted_skipped[:5])
        suffix = "..." if len(whitelisted_skipped) > 5 else ""
        print(f"  Whitelisted domains skipped: {len(whitelisted_skipped)}  ({sample}{suffix})")
    print(f"  Candidate domains: {len(candidates)}")

    if not candidates:
        print("\nAll significant domains are already covered or whitelisted.")
        return

    print(f"\nClassifying {len(candidates)} domain(s) with LLM...\n")

    new_rules: list = []
    ambiguous: list = []  # (domain, entry) — no reliable rule possible

    for domain, entry in candidates:
        unique = entry["unique_emails"]
        is_general = domain.lower() in _GENERAL_PURPOSE_DOMAINS
        label = "  [general-purpose]" if is_general else ""
        print(f"  {domain} ({unique} emails){label}...")

        if is_general:
            result = _llm_classify_general_domain(
                llm, model, domain, entry["sample_senders"], entry["sample_subjects"], unique)
        else:
            result = _llm_classify_dedicated(llm, model, domain, entry["sample_subjects"], unique)

        if result is None:
            print(f"    -> LLM error — skipping")
            continue

        if args.auto:
            # ── Auto mode: apply LLM result directly (original behaviour) ──
            if is_general:
                if not result.get("is_noise"):
                    print(f"    -> no reliable narrow rule — queued for deep-clean")
                    ambiguous.append((domain, entry))
                    continue
                patterns_s   = result.get("sender_patterns") or []
                patterns_sub = result.get("subject_patterns") or []
                if not patterns_s and not patterns_sub:
                    print(f"    -> LLM returned no patterns — queued for deep-clean")
                    ambiguous.append((domain, entry))
                    continue
                print(f"    -> NOISE (narrow patterns): {result.get('description')}")
                new_rules.append({
                    "domain": domain,
                    "description": result["description"],
                    "sender_patterns":  patterns_s,
                    "subject_patterns": patterns_sub,
                })
            else:
                if not result.get("is_noise"):
                    print(f"    -> clean")
                    continue
                print(f"    -> NOISE: {result.get('description')}")
                new_rules.append({
                    "domain": domain,
                    "description": result["description"],
                    "sender_domains": [domain],
                })
        else:
            # ── Interactive mode ────────────────────────────────────────────
            decision = _interactive_domain_prompt(
                domain, entry, result, is_general,
                qdrant, collection, llm, model, args.inspect_count,
            )
            if decision == "rule":
                rule = _result_to_rule(domain, result, is_general)
                if rule:
                    new_rules.append(rule)
                    print(f"    -> Rule queued.")
                else:
                    print(f"    -> No pattern available for general-purpose domain '{domain}'.")
                    print(f"       Use 'deep-clean --domain {domain}' to classify individual emails.")
            elif decision == "whitelist":
                print(f"    -> Whitelisted.")
            else:
                print(f"    -> Skipped.")

    print(f"\n{'=' * 60}")
    print(f"  New rules found      : {len(new_rules)}")
    print(f"  Ambiguous (no rule)  : {len(ambiguous)}")
    print(f"{'=' * 60}")

    if ambiguous:
        print("\n  Ambiguous domains (no reliable rule):")
        for domain, entry in ambiguous:
            print(f"    {domain} ({entry['unique_emails']} emails)")

    if not args.dry_run and new_rules:
        written = _merge_rules_to_yaml(new_rules)
        print(f"\nMerged {written} rule(s) into noise_rules.yaml.")
        print("Review the git diff and commit the rules you want to keep.")
    elif args.dry_run and new_rules:
        print("\n(Dry run — noise_rules.yaml was not modified.)")

    # Offer deep-clean for ambiguous domains (auto mode only; interactive users
    # can run deep-clean separately or re-run discover and choose [2])
    if ambiguous and not args.dry_run and args.auto:
        if args.deep_clean or _confirm(
            f"Run deep-clean (per-email LLM classification) on {len(ambiguous)} ambiguous domain(s)?"
        ):
            ambiguous_domains = {domain for domain, _ in ambiguous}
            _deep_clean_domains(ambiguous_domains, qdrant, llm, model,
                                azure_conn_str, azure_container, collection, dry_run=False)


# ── Subcommand: purge ─────────────────────────────────────────────────────────

def cmd_purge(args, qdrant, azure_conn_str, azure_container, collection):
    from src.data.noise_filter import NoiseFilter

    print(f"\n{'=' * 60}")
    print(f"  Purge {'(DRY RUN) ' if args.dry_run else ''}")
    print(f"{'=' * 60}")

    noise_filter = NoiseFilter.from_project_rules()
    if noise_filter.is_empty():
        print("No rules found in noise_rules.yaml — nothing to purge.")
        return
    print(f"  Rules loaded: {len(noise_filter.category_names())} categories")

    print(f"\nScanning collection '{collection}'...")
    matches: dict = defaultdict(list)
    seen: dict = defaultdict(set)
    total_points = 0

    for point_id, payload in _scroll_all(qdrant, collection):
        total_points += 1
        matched, category = noise_filter.match_payload(payload)
        if not matched:
            continue
        source_id = payload.get("source_id", "")
        matches[category].append({"point_id": point_id, "source_id": source_id,
                                   "subject": payload.get("subject", "")})
        if source_id:
            seen[category].add(source_id)

    total_vectors = sum(len(v) for v in matches.values())
    total_emails  = sum(len(s) for s in seen.values())
    print(f"  Vectors scanned : {total_points}")
    print(f"  Noise vectors   : {total_vectors}")
    print(f"  Unique emails   : {total_emails}")

    if not matches:
        print("\nNo noise found — index is clean.")
        return

    print()
    for category, hits in matches.items():
        print(f"  [{category}]  {len(seen[category])} emails / {len(hits)} vectors")
        shown = set()
        for h in hits:
            s = h["subject"][:80]
            if s not in shown:
                print(f"    Sample: {s}")
                shown.add(s)
            if len(shown) >= 3:
                break
        print()

    if args.dry_run:
        print("(Dry run — no changes made.)")
        return

    all_point_ids = [h["point_id"] for hits in matches.values() for h in hits]
    if _confirm(f"Delete {total_vectors} vectors from Qdrant '{collection}'?"):
        _delete_qdrant_points(qdrant, collection, all_point_ids)
        print(f"  Done — {total_vectors} vectors removed.")
    else:
        print("  Skipped Qdrant deletion.")

    all_blob_paths = list({_blob_path(h["source_id"])
                           for hits in matches.values() for h in hits if h["source_id"]})
    if not azure_conn_str:
        print("\nAZURE_STORAGE_CONNECTION_STRING not set — skipping blob deletion.")
        return
    if _confirm(f"Also delete {len(all_blob_paths)} .eml files from Azure Blob '{azure_container}'?"):
        _delete_blobs(azure_conn_str, azure_container, all_blob_paths)
    else:
        print("  Skipped Azure Blob deletion.")


# ── Subcommand: deep-clean ────────────────────────────────────────────────────

def _deep_clean_domains(target_domains: set, qdrant, llm, model,
                        azure_conn_str: str, azure_container: str,
                        collection: str, dry_run: bool) -> None:
    """Core deep-clean logic shared by the deep-clean subcommand and discover --auto --deep-clean."""
    print(f"\nScrolling '{collection}' for emails from: {sorted(target_domains)}...")
    domain_emails = _scroll_domain_emails(qdrant, collection, target_domains)

    if not domain_emails:
        print("No emails found from target domains.")
        return

    for domain, emails in sorted(domain_emails.items()):
        print(f"\n{'─' * 60}")
        print(f"  Domain: {domain}  ({len(emails)} unique emails)")
        print(f"{'─' * 60}")

        email_list = list(emails.items())
        noise_source_ids: list = []
        noise_emails_meta: list = []
        classified = 0

        print(f"  Classifying in batches of {_DEEP_CLEAN_BATCH}...")
        for start in range(0, len(email_list), _DEEP_CLEAN_BATCH):
            batch_items = email_list[start: start + _DEEP_CLEAN_BATCH]
            batch_data = [{"sender": e["sender"], "subject": e["subject"], "body": e["body"]}
                          for _, e in batch_items]
            results = _llm_classify_email_batch(llm, model, domain, batch_data)
            if results is None:
                print(f"  Warning: unexpected LLM response at batch {start} — skipping batch.")
                continue
            for (sid, entry), is_noise in zip(batch_items, results):
                if is_noise:
                    noise_source_ids.append(sid)
                    noise_emails_meta.append({"sender": entry["sender"], "subject": entry["subject"]})
            classified += len(batch_items)
            print(f"  Classified {classified}/{len(email_list)}...", end="\r")

        print()
        noise_count = len(noise_source_ids)
        print(f"  Noise     : {noise_count}")
        print(f"  Legitimate: {len(email_list) - noise_count}")

        if noise_count == 0:
            print("  No noise found — skipping.")
            continue

        print(f"\n  Sample noise emails:")
        for meta in noise_emails_meta[:5]:
            print(f"    [{meta['sender'][:50]}]  {meta['subject'][:60]}")

        # Attempt rule extraction
        if noise_count >= _MIN_EMAILS_FOR_RULE:
            print(f"\n  Attempting rule extraction from {noise_count} noise emails...")
            rule = _llm_extract_rule(llm, model, domain, noise_emails_meta)
            if rule and rule.get("has_rule"):
                print(f"  Rule suggested: {rule.get('description')}")
                if rule.get("sender_patterns"):
                    print(f"    sender_patterns : {rule['sender_patterns']}")
                if rule.get("subject_patterns"):
                    print(f"    subject_patterns: {rule['subject_patterns']}")
                if not dry_run and _confirm("Add this rule to noise_rules.yaml?"):
                    written = _merge_rules_to_yaml([{
                        "domain": domain,
                        "description": rule["description"],
                        "sender_patterns":  rule.get("sender_patterns") or [],
                        "subject_patterns": rule.get("subject_patterns") or [],
                    }])
                    if written:
                        print("  Rule merged. Review via git diff and commit to keep it.")
            else:
                print("  No reliable rule found — cleaning without a rule.")
        else:
            print(f"  Too few noise emails ({noise_count}) for reliable rule extraction.")

        if dry_run:
            print(f"\n  (Dry run — skipping deletion of {noise_count} emails.)")
            continue

        noise_point_ids = [pid for sid in noise_source_ids for pid in emails[sid]["point_ids"]]

        print()
        if _confirm(f"Delete {len(noise_point_ids)} vectors ({noise_count} emails) from Qdrant?"):
            _delete_qdrant_points(qdrant, collection, noise_point_ids)
            print(f"  Done — {len(noise_point_ids)} vectors removed.")
        else:
            print("  Skipped Qdrant deletion.")

        if azure_conn_str:
            blob_paths = [_blob_path(sid) for sid in noise_source_ids]
            if _confirm(f"Also delete {len(blob_paths)} .eml files from Azure Blob?"):
                _delete_blobs(azure_conn_str, azure_container, blob_paths)
            else:
                print("  Skipped Azure Blob deletion.")
        else:
            print("  AZURE_STORAGE_CONNECTION_STRING not set — skipping blob deletion.")


def cmd_deep_clean(args, qdrant, llm, model, azure_conn_str, azure_container, collection):
    print(f"\n{'=' * 60}")
    print(f"  Deep Clean {'(DRY RUN) ' if args.dry_run else ''}")
    print(f"  Model: {model}")
    print(f"{'=' * 60}")

    if args.domain:
        target_domains = {args.domain.lower()}
    else:
        _, existing_domains = _load_existing_state()
        target_domains = _GENERAL_PURPOSE_DOMAINS - existing_domains
        if not target_domains:
            print("\nAll general-purpose domains are already covered by rules.")
            return
        print(f"\nAuto-detected domains: {sorted(target_domains)}")

    _deep_clean_domains(target_domains, qdrant, llm, model,
                        azure_conn_str, azure_container, collection, dry_run=args.dry_run)

    print(f"\n{'=' * 60}")
    print("  Deep clean complete.")
    print(f"{'=' * 60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="noise.py",
        description="Noise management tool — discover, purge and deep-clean noisy emails.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # discover
    p_disc = sub.add_parser("discover", help="Find new noise rules from the indexed collection")
    p_disc.add_argument("--min-emails", type=int, default=_DISCOVER_MIN_EMAILS,
                        help=f"Minimum unique emails per domain to consider (default: {_DISCOVER_MIN_EMAILS})")
    p_disc.add_argument("--dry-run", action="store_true",
                        help="Print suggestions without writing to noise_rules.yaml")
    p_disc.add_argument("--auto", action="store_true",
                        help="Write rules automatically without interactive prompts")
    p_disc.add_argument("--deep-clean", action="store_true",
                        help="(--auto only) Automatically deep-clean ambiguous domains after discovery")
    p_disc.add_argument("--inspect-count", type=int, default=_DEFAULT_INSPECT_COUNT,
                        help=f"Emails to fetch per domain for [2] deep-inspect and [3] read (default: {_DEFAULT_INSPECT_COUNT})")

    # purge
    p_purge = sub.add_parser("purge", help="Delete noise matched by noise_rules.yaml from Qdrant and Azure Blob")
    p_purge.add_argument("--dry-run", action="store_true", help="Show matches without deleting")

    # deep-clean
    p_dc = sub.add_parser("deep-clean", help="Per-email LLM classification for ambiguous domains")
    p_dc.add_argument("--domain", metavar="DOMAIN",
                      help="Target a specific domain (default: auto-detect all uncovered general-purpose domains)")
    p_dc.add_argument("--dry-run", action="store_true", help="Classify but do not delete")

    args = parser.parse_args()

    qdrant_url      = os.getenv("QDRANT_URL", "http://host.docker.internal:6333").strip()
    qdrant_api_key  = os.getenv("QDRANT_API_KEY", "").strip() or None
    collection      = os.getenv("QDRANT_COLLECTION_NAME", "email-rag").strip()
    azure_conn_str  = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "").strip()
    azure_container = os.getenv("AZURE_BLOB_CONTAINER", "eml-archive").strip()

    from qdrant_client import QdrantClient
    qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    if args.command == "purge":
        cmd_purge(args, qdrant, azure_conn_str, azure_container, collection)
        return

    # discover and deep-clean both need the LLM
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("Error: OPENAI_API_KEY not set.")
        sys.exit(1)
    from openai import OpenAI
    llm   = OpenAI(api_key=api_key)
    model = os.getenv("RAG_LLM_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"

    if args.command == "discover":
        cmd_discover(args, qdrant, llm, model, azure_conn_str, azure_container, collection)
    elif args.command == "deep-clean":
        cmd_deep_clean(args, qdrant, llm, model, azure_conn_str, azure_container, collection)


if __name__ == "__main__":
    main()
