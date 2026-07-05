"""Tests for calendar-invite detection + one-line summarization (stdlib-only)."""

import unittest

from src.data import calendar_summary as cal

SAMPLE_ICS = """BEGIN:VCALENDAR
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
"""


class TestIsCalendarSubject(unittest.TestCase):
    def test_matches_known_prefixes_case_insensitively(self):
        for s in [
            "Canceled: x",
            "Accepted: y",
            "Declined: z",
            "Invitation: a",
            "Updated invitation: b",
            "tentative: c",
        ]:
            self.assertTrue(cal.is_calendar_subject(s), s)

    def test_rejects_normal_subjects(self):
        self.assertFalse(cal.is_calendar_subject("Re: ACP test cases"))
        self.assertFalse(cal.is_calendar_subject(""))


class TestIsCalendar(unittest.TestCase):
    def test_true_when_calendar_part_present(self):
        self.assertTrue(cal.is_calendar(subject="Re: x", body="hi", calendar_text=SAMPLE_ICS))

    def test_true_when_vcalendar_in_body(self):
        self.assertTrue(cal.is_calendar(subject="Re: x", body=SAMPLE_ICS))

    def test_true_when_subject_prefix(self):
        self.assertTrue(cal.is_calendar(subject="Canceled: standup", body="plain"))

    def test_false_for_ordinary_mail(self):
        self.assertFalse(cal.is_calendar(subject="Re: budget", body="some text"))


class TestExtractVcalendarFields(unittest.TestCase):
    def test_extracts_event_fields(self):
        f = cal.extract_vcalendar_fields(SAMPLE_ICS)
        self.assertEqual(f["summary"], "ACP sync")
        self.assertEqual(f["organizer"], "alice@example.com")
        self.assertEqual(f["dtstart"], "2025-01-01 10:00")
        self.assertEqual(f["dtend"], "2025-01-01 10:30")
        self.assertEqual(f["location"], "Teams")
        self.assertEqual(f["attendee_count"], 2)


class TestSummarizeCalendar(unittest.TestCase):
    def test_builds_one_line_from_ics(self):
        line = cal.summarize_calendar("Invitation: ACP sync", SAMPLE_ICS)
        self.assertEqual(line.count("\n"), 0)
        self.assertIn("[Calendar] ACP sync", line)
        self.assertIn("organizer: alice@example.com", line)
        self.assertIn("when: 2025-01-01 10:00", line)
        self.assertIn("2025-01-01 10:30", line)
        self.assertIn("location: Teams", line)
        self.assertIn("2 attendee", line)

    def test_falls_back_to_subject_without_ics(self):
        line = cal.summarize_calendar("Canceled: standup", "no calendar content here")
        self.assertEqual(line, "[Calendar] Canceled: standup")


if __name__ == "__main__":
    unittest.main()
