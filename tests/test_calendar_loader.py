"""Loader integration: calendar invites collapse to a one-line summary,
ordinary mail is left untouched. Imports the loader, so runs in the devcontainer.
"""
import os
import tempfile
import unittest

from src.data.loaders.mail_archive_x import MailArchiveXLoader

_CAL_EML = b"""From: Alice <alice@example.com>
To: bob@initech.com
Subject: Invitation: ACP sync
Date: Wed, 1 Jan 2025 10:00:00 +0000
Message-ID: <inv@example.com>
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="BOUND"

--BOUND
Content-Type: text/plain; charset=utf-8

You are invited to ACP sync.

--BOUND
Content-Type: text/calendar; method=REQUEST; charset=utf-8

BEGIN:VCALENDAR
BEGIN:VEVENT
SUMMARY:ACP sync
ORGANIZER;CN=Alice:mailto:alice@example.com
DTSTART:20250101T100000Z
DTEND:20250101T103000Z
LOCATION:Teams
ATTENDEE;ROLE=REQ-PARTICIPANT:mailto:bob@initech.com
ATTENDEE;ROLE=REQ-PARTICIPANT:mailto:carol@example.com
END:VEVENT
END:VCALENDAR

--BOUND--
"""

_PLAIN_EML = b"""From: Carol <carol@example.com>
To: dave@umbrella.com
Subject: Re: budget numbers
Date: Wed, 1 Jan 2025 11:00:00 +0000
Message-ID: <plain@example.com>
Content-Type: text/plain; charset=utf-8

Let's discuss the numbers tomorrow.
"""


def _load_one(raw):
    with tempfile.TemporaryDirectory() as root:
        with open(os.path.join(root, "m.eml"), "wb") as fh:
            fh.write(raw)
        return MailArchiveXLoader(root).load()[0]


class TestCalendarLoaderIntegration(unittest.TestCase):
    def test_invite_collapses_to_one_line_summary(self):
        email = _load_one(_CAL_EML)
        self.assertTrue(email.body.startswith("[Calendar] ACP sync"), email.body)
        self.assertNotIn("BEGIN:VCALENDAR", email.body)
        self.assertIn("2 attendee", email.body)
        self.assertEqual(email.body.count("\n"), 0)

    def test_ordinary_mail_is_not_summarized(self):
        email = _load_one(_PLAIN_EML)
        self.assertEqual(email.body, "Let's discuss the numbers tomorrow.")
        self.assertNotIn("[Calendar]", email.body)


if __name__ == "__main__":
    unittest.main()
