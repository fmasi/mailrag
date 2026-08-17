"""Guard against private-corpus identifiers reaching the public repo.

The corpus this project was built on is real work and personal mail. A single real
sender address, committed to a public repo, is enough to start de-anonymising the
dataset, and the project's whole argument is that email is too sensitive to hand over.

Two checks, deliberately narrow so they stay worth keeping:

1. **Known private-corpus domains must never reappear.** These are domains that were
   found in tracked files and removed. A regression here means a fixture was copied
   from real mail again.
2. **No `first.last@` addresses on non-documentation domains.** That shape is what a
   real person's address looks like, and it is what both real leaks looked like
   (`eric.levander@…`, `augusto.cezar@…`). Throwaway fixtures such as `x@y.com` or
   `bob@initech.com` are not person-shaped and do not fire.

Neither check tries to be exhaustive. A broad "flag every unusual domain" rule fires on
dozens of harmless fixtures, and a noisy guard is a guard someone eventually deletes.
"""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Domains that were carrying real identifiers in tracked files and were replaced.
# Removed 2026-08-17: see the anonymisation pass in the public-text work.
BANNED_DOMAINS = {
    "windriver.com",
    "windriversystems.onmicrosoft.com",
    "layer7tech.com",
    "clientesiberdrola.pt",
    "tripadvisor.com",
}

# first.last@domain or first_last@domain, which is how real people's addresses look.
PERSON_SHAPED = re.compile(r"\b([A-Za-z]{2,}[._][A-Za-z]{2,})@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")

ANY_ADDRESS = re.compile(r"[A-Za-z][A-Za-z0-9._%+-]*@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Reserved for documentation and testing (RFC 2606 / RFC 6761), so no real person owns them.
RESERVED_SUFFIXES = (".example", ".test", ".invalid", ".localhost")
RESERVED_DOMAINS = {"example.com", "example.org", "example.net", "localhost"}

# Real domains where a person-shaped local part is expected and harmless.
#   enron.com - the public corpus, already public and the field standard
PERSON_SHAPE_ALLOWED_DOMAINS = {"enron.com"}

# eval/ holds committed public-Enron fixtures, which legitimately carry Enron addresses.
SKIP_PREFIXES = ("eval/",)
SKIP_SUFFIXES = (".png", ".jpg", ".jpeg", ".svg", ".ico", ".pdf", ".lock", ".woff", ".woff2")


def _tracked_text_files() -> list[str]:
    """Tracked text files, excluding this one.

    This file names every pattern it forbids, so scanning itself would fail the moment
    it was committed. `git ls-files` does not list it until then, which would have made
    that failure a surprise at commit time rather than here.
    """
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    self_path = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
    return [
        f
        for f in out
        if f != self_path
        and not f.startswith(SKIP_PREFIXES)
        and not f.lower().endswith(SKIP_SUFFIXES)
    ]


def _read(rel: str) -> str:
    try:
        return (REPO_ROOT / rel).read_text(encoding="utf-8")
    except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
        return ""


def _is_documentation_domain(domain: str) -> bool:
    domain = domain.lower().rstrip(".")
    return domain in RESERVED_DOMAINS or domain.endswith(RESERVED_SUFFIXES)


class TestNoPrivateIdentifiersAreCommitted(unittest.TestCase):
    def test_removed_private_domains_do_not_come_back(self):
        offenders: dict[str, list[str]] = {}
        for rel in _tracked_text_files():
            lowered = _read(rel).lower()
            for domain in BANNED_DOMAINS:
                if domain in lowered:
                    offenders.setdefault(domain, []).append(rel)

        self.assertEqual(
            offenders,
            {},
            "A domain from the private corpus is back in a tracked file. Replace the "
            "fixture with a `.example` address:\n"
            + "\n".join(f"  {d} in {sorted(set(f))}" for d, f in sorted(offenders.items())),
        )

    def test_no_person_shaped_addresses_outside_documentation_domains(self):
        offenders: dict[str, list[str]] = {}
        for rel in _tracked_text_files():
            for match in PERSON_SHAPED.finditer(_read(rel)):
                domain = match.group(2).lower()
                if _is_documentation_domain(domain) or domain in PERSON_SHAPE_ALLOWED_DOMAINS:
                    continue
                offenders.setdefault(match.group(0), []).append(rel)

        self.assertEqual(
            offenders,
            {},
            "Addresses shaped like a real person's on a domain someone could own. Use a "
            "`.example` domain instead:\n"
            + "\n".join(f"  {a} in {sorted(set(f))}" for a, f in sorted(offenders.items())),
        )

    def test_the_scan_is_not_vacuous(self):
        """Both checks above assert absence, so prove the machinery actually looks."""
        files = _tracked_text_files()
        self.assertGreater(len(files), 100, "expected many tracked text files")

        addresses = sum(len(ANY_ADDRESS.findall(_read(f))) for f in files)
        self.assertGreater(addresses, 50, "expected fixtures to contain many addresses")

        # And the person-shaped pattern must still match the shape it was written for,
        # otherwise the second check passes because the regex stopped working.
        self.assertRegex("dana.reyes@northwind.example", PERSON_SHAPED)
        self.assertIsNone(PERSON_SHAPED.search("x@y.com"))


if __name__ == "__main__":
    unittest.main()
