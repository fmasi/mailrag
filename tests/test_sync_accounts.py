"""Account config + secret resolution — the "any user, any provider" seam (#101)."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from unittest import mock

from src.config.secrets import SecretError, resolve_secret
from src.sync.accounts import (
    AccountConfig,
    account_from_dict,
    get_account,
    load_accounts,
    parse_cadence,
)
from src.sync.sources import FolderRole, resolve_role


class TestParseCadence(unittest.TestCase):
    def test_parses_each_unit(self):
        self.assertEqual(parse_cadence("30s"), 30)
        self.assertEqual(parse_cadence("15m"), 900)
        self.assertEqual(parse_cadence("12h"), 43200)
        self.assertEqual(parse_cadence("1d"), 86400)

    def test_tolerates_whitespace_and_case(self):
        self.assertEqual(parse_cadence(" 12H "), 43200)

    def test_rejects_nonsense(self):
        for bad in ("", "12", "h", "-1h", "0h", "soon"):
            with self.assertRaises(ValueError):
                parse_cadence(bad)


class TestAccountFromDict(unittest.TestCase):
    def test_minimal_account_gets_sensible_defaults(self):
        cfg = account_from_dict({"id": "a"})
        self.assertEqual(cfg.source, "imap")
        self.assertEqual(cfg.port, 993)
        self.assertEqual(cfg.cadence_seconds(), 43200)
        self.assertIn(FolderRole.SENT, cfg.include_roles)
        self.assertEqual(cfg.exclude_roles, [FolderRole.JUNK, FolderRole.TRASH])

    def test_an_id_is_required(self):
        with self.assertRaises(ValueError):
            account_from_dict({"host": "imap.example.com"})

    def test_unknown_keys_are_rejected_at_load_time(self):
        """A typo must surface now, not inside a 3am scheduled run nobody watches."""
        with self.assertRaises(ValueError) as ctx:
            account_from_dict({"id": "a", "hsot": "typo"})
        self.assertIn("hsot", str(ctx.exception))

    def test_an_invalid_role_is_rejected(self):
        with self.assertRaises(ValueError):
            account_from_dict({"id": "a", "include_roles": ["inbox", "nonsense"]})

    def test_an_invalid_cadence_is_rejected_at_load_time(self):
        with self.assertRaises(ValueError):
            account_from_dict({"id": "a", "cadence": "whenever"})

    def test_an_invalid_folder_role_override_is_rejected(self):
        with self.assertRaises(ValueError):
            account_from_dict({"id": "a", "folder_roles": {"Weird": "nope"}})

    def test_roles_can_be_narrowed(self):
        cfg = account_from_dict({"id": "a", "include_roles": ["inbox"], "exclude_roles": []})
        self.assertEqual(cfg.include_roles, [FolderRole.INBOX])
        self.assertEqual(cfg.exclude_roles, [])


class TestScope(unittest.TestCase):
    def test_the_default_scope_is_everything_but_junk_and_trash(self):
        cfg = AccountConfig(id="a")
        self.assertTrue(cfg.wants(FolderRole.INBOX))
        self.assertTrue(cfg.wants(FolderRole.SENT))
        self.assertTrue(cfg.wants(FolderRole.ARCHIVE))
        self.assertTrue(cfg.wants(FolderRole.OTHER))
        self.assertFalse(cfg.wants(FolderRole.JUNK))
        self.assertFalse(cfg.wants(FolderRole.TRASH))

    def test_drafts_are_out_of_scope_by_default(self):
        """Unsent drafts are not correspondence and would pollute threads."""
        self.assertFalse(AccountConfig(id="a").wants(FolderRole.DRAFTS))

    def test_exclusion_beats_inclusion(self):
        cfg = AccountConfig(
            id="a", include_roles=[FolderRole.INBOX], exclude_roles=[FolderRole.INBOX]
        )
        self.assertFalse(cfg.wants(FolderRole.INBOX))


class TestResolveRole(unittest.TestCase):
    def test_special_use_flags_win_over_names(self):
        self.assertEqual(resolve_role("Weird Name", flags=["\\Sent"]), FolderRole.SENT)

    def test_icloud_literal_names_map_correctly(self):
        """iCloud advertises no SPECIAL-USE, so the name table is the only signal."""
        self.assertEqual(resolve_role("Sent Messages"), FolderRole.SENT)
        self.assertEqual(resolve_role("Deleted Messages"), FolderRole.TRASH)
        self.assertEqual(resolve_role("Archive"), FolderRole.ARCHIVE)
        self.assertEqual(resolve_role("INBOX"), FolderRole.INBOX)

    def test_gmail_and_exchange_names_map_correctly(self):
        self.assertEqual(resolve_role("Sent Mail"), FolderRole.SENT)
        self.assertEqual(resolve_role("Sent Items"), FolderRole.SENT)
        self.assertEqual(resolve_role("Junk E-mail"), FolderRole.JUNK)

    def test_an_unknown_folder_is_other_not_junk(self):
        """Guessing JUNK would silently drop a user's filed mail."""
        self.assertEqual(resolve_role("Projects/2026"), FolderRole.OTHER)

    def test_a_subfolder_is_not_the_inbox(self):
        self.assertEqual(resolve_role("INBOX/Receipts"), FolderRole.OTHER)

    def test_a_user_override_beats_everything(self):
        self.assertEqual(
            resolve_role("Sent Messages", flags=["\\Sent"], overrides={"Sent Messages": "other"}),
            FolderRole.OTHER,
        )

    def test_overrides_are_case_insensitive(self):
        self.assertEqual(resolve_role("Weird", overrides={"weird": "archive"}), FolderRole.ARCHIVE)

    def test_an_invalid_override_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_role("Weird", overrides={"Weird": "nonsense"})


class TestLoadAccounts(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)

    def _write(self, text: str) -> str:
        path = os.path.join(self.d, "accounts.yaml")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def test_a_missing_file_is_an_empty_list_not_an_error(self):
        """'Sync isn't set up yet' is a normal state the CLI explains."""
        self.assertEqual(load_accounts(os.path.join(self.d, "nope.yaml")), [])

    def test_loads_multiple_accounts(self):
        path = self._write(
            "accounts:\n"
            "  - id: personal\n"
            "    host: imap.mail.me.com\n"
            "    collection: personal\n"
            "  - id: work\n"
            "    source: maildir\n"
            "    path: /mail/work\n"
            "    collection: work\n"
        )
        accounts = load_accounts(path)
        self.assertEqual([a.id for a in accounts], ["personal", "work"])
        self.assertEqual(accounts[1].source, "maildir")
        self.assertEqual(accounts[0].collection, "personal")

    def test_two_accounts_may_feed_one_collection(self):
        path = self._write(
            "accounts:\n  - id: a\n    collection: shared\n  - id: b\n    collection: shared\n"
        )
        self.assertEqual({a.collection for a in load_accounts(path)}, {"shared"})

    def test_duplicate_ids_are_rejected(self):
        """Duplicate ids would silently share a ledger and fight over cursors."""
        path = self._write("accounts:\n  - id: a\n  - id: a\n")
        with self.assertRaises(ValueError):
            load_accounts(path)

    def test_a_file_without_an_accounts_key_is_rejected(self):
        path = self._write("something_else: 1\n")
        with self.assertRaises(ValueError):
            load_accounts(path)

    def test_an_empty_accounts_list_is_allowed(self):
        self.assertEqual(load_accounts(self._write("accounts:\n")), [])

    def test_get_account_finds_by_id(self):
        path = self._write("accounts:\n  - id: personal\n")
        self.assertEqual(get_account("personal", path).id, "personal")

    def test_get_account_lists_known_ids_when_it_misses(self):
        path = self._write("accounts:\n  - id: personal\n")
        with self.assertRaises(KeyError) as ctx:
            get_account("nope", path)
        self.assertIn("personal", str(ctx.exception))


class TestResolveSecret(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)

    def test_env_scheme(self):
        with mock.patch.dict(os.environ, {"MAILRAG_TEST_SECRET": "hunter2"}):
            self.assertEqual(resolve_secret("env:MAILRAG_TEST_SECRET"), "hunter2")

    def test_env_scheme_missing_variable(self):
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(SecretError):
            resolve_secret("env:NOT_SET_ANYWHERE")

    def test_file_scheme_reads_the_first_line(self):
        path = os.path.join(self.d, "s.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("hunter2\nignored\n")
        self.assertEqual(resolve_secret(f"file:{path}"), "hunter2")

    def test_file_scheme_missing_file(self):
        with self.assertRaises(SecretError):
            resolve_secret(f"file:{os.path.join(self.d, 'nope')}")

    def test_file_scheme_empty_file(self):
        path = os.path.join(self.d, "empty.txt")
        open(path, "w").close()
        with self.assertRaises(SecretError):
            resolve_secret(f"file:{path}")

    def test_a_bare_literal_is_refused(self):
        """accounts.yaml gets copied, pasted and backed up — it is not a password store."""
        with self.assertRaises(SecretError) as ctx:
            resolve_secret("hunter2")
        self.assertIn("reference", str(ctx.exception))

    def test_an_empty_reference_is_refused(self):
        with self.assertRaises(SecretError):
            resolve_secret("")

    def test_an_unknown_scheme_is_refused_without_echoing_it(self):
        """A value like "hunter2:" parses as scheme="hunter2" — echoing the
        scheme back would leak the password (#101 review finding)."""
        with self.assertRaises(SecretError) as ctx:
            resolve_secret("vault:secret/mail")
        self.assertNotIn("vault", str(ctx.exception))
        self.assertNotIn("secret/mail", str(ctx.exception))

    def test_a_scheme_with_no_target_is_refused(self):
        with self.assertRaises(SecretError):
            resolve_secret("env:")

    def test_keychain_scheme_shells_out_to_security(self):
        with (
            mock.patch("src.config.secrets.shutil.which", return_value="/usr/bin/security"),
            mock.patch("src.config.secrets.subprocess.run") as run,
        ):
            run.return_value = mock.Mock(returncode=0, stdout="hunter2\n", stderr="")
            self.assertEqual(resolve_secret("keychain:mailrag.imap.personal"), "hunter2")
        self.assertIn("find-generic-password", run.call_args[0][0])

    def test_keychain_missing_item_explains_how_to_add_one(self):
        with (
            mock.patch("src.config.secrets.shutil.which", return_value="/usr/bin/security"),
            mock.patch("src.config.secrets.subprocess.run") as run,
        ):
            run.return_value = mock.Mock(returncode=44, stdout="", stderr="not found")
            with self.assertRaises(SecretError) as ctx:
                resolve_secret("keychain:mailrag.imap.personal")
        self.assertIn("add-generic-password", str(ctx.exception))

    def test_keychain_without_security_points_at_the_portable_schemes(self):
        """A Linux user must not hit a macOS-only dead end."""
        with mock.patch("src.config.secrets.shutil.which", return_value=None):
            with self.assertRaises(SecretError) as ctx:
                resolve_secret("keychain:x")
        self.assertIn("env:", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()


class TestSecretRedaction(unittest.TestCase):
    """The 'plaintext rejected' guard used to interpolate the rejected literal —
    i.e. the password — into an error the runner writes to ~/.mailrag/sync.log
    and persists in sync_runs.message, replayed every cadence tick (#101 review
    finding)."""

    def test_a_rejected_literal_is_never_echoed(self):
        secret = "hunter2-correct-horse-battery"
        with self.assertRaises(SecretError) as ctx:
            resolve_secret(secret)
        self.assertNotIn(secret, str(ctx.exception))
        self.assertIn("withheld", str(ctx.exception))

    def test_a_malformed_reference_does_not_echo_its_value(self):
        with self.assertRaises(SecretError) as ctx:
            resolve_secret("hunter2-correct-horse:")
        self.assertNotIn("hunter2", str(ctx.exception))

    def test_the_error_still_says_what_to_do(self):
        with self.assertRaises(SecretError) as ctx:
            resolve_secret("hunter2")
        for scheme in ("keychain:", "env:", "file:"):
            self.assertIn(scheme, str(ctx.exception))

    def test_an_empty_reference_is_described_as_empty(self):
        with self.assertRaises(SecretError) as ctx:
            resolve_secret("")
        self.assertIn("empty", str(ctx.exception))


class TestEmbedSummaryConfig(unittest.TestCase):
    """Sync hardcoded embed_summary=True, so its policy fingerprint could never
    match a collection built without --embed-summary; the guard then raised on
    every run and the catch-all filed it as a Qdrant outage (#101 review)."""

    def test_it_defaults_to_true(self):
        self.assertTrue(account_from_dict({"id": "a"}).embed_summary)

    def test_it_can_be_turned_off_to_match_the_collection(self):
        self.assertFalse(account_from_dict({"id": "a", "embed_summary": False}).embed_summary)
