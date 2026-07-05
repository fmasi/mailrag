"""Unit tests for reply-chain stripping and HTML-to-text conversion.

These tests cover _strip_reply_chain and _HTMLTextExtractor in
src/data/loaders/mail_archive_x.py.

When adapting this project to a new email source, run the debug script to
identify separator patterns that are not yet handled, then add test cases here
before extending _REPLY_SEPARATOR_RE or _strip_reply_chain.

See docs/EMAIL_PREPROCESSING.md for the full workflow.
"""

import unittest

from src.data.loaders.mail_archive_x import (
    MailArchiveXLoader,
    _HTMLTextExtractor,
)


class TestStripReplyChainNoReply(unittest.TestCase):
    """Bodies with no reply chain should be returned unchanged."""

    def test_plain_body_unchanged(self):
        body = "Hi Alice,\n\nPlease send me the report.\n\nThanks, Bob"
        self.assertEqual(MailArchiveXLoader._strip_reply_chain(body), body)

    def test_empty_body(self):
        self.assertEqual(MailArchiveXLoader._strip_reply_chain(""), "")

    def test_whitespace_only_body(self):
        self.assertEqual(MailArchiveXLoader._strip_reply_chain("   \n\n  "), "")

    def test_on_word_not_attribution(self):
        # "On" at the start of a sentence that is NOT a reply attribution.
        body = "On the topic of budgets, see below.\n\nDetails here."
        self.assertEqual(MailArchiveXLoader._strip_reply_chain(body), body)

    def test_from_word_mid_body_not_outlook_header(self):
        # "From" that is not followed by Date:/To:/Subject: — should NOT strip.
        body = "Heard from the team yesterday.\n\nFrom what I understand, it's fine."
        self.assertEqual(MailArchiveXLoader._strip_reply_chain(body), body)


class TestStripReplyChainQuotedLines(unittest.TestCase):
    """Lines starting with '>' should truncate the body."""

    def test_single_quoted_line(self):
        body = "Sure, sounds good.\n\n> Original text here."
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertEqual(result, "Sure, sounds good.")

    def test_double_quoted_lines(self):
        body = "Agreed.\n\n>> deeply nested quote"
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertEqual(result, "Agreed.")

    def test_trailing_blank_lines_stripped(self):
        body = "My reply.\n\n\n> Quoted line"
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertEqual(result, "My reply.")


class TestStripReplyChainOutlookSeparators(unittest.TestCase):
    """Outlook '-----Original Message-----' and '___...' separators."""

    def test_original_message_dashes_both_sides(self):
        body = "See below.\n\n-----Original Message-----\nFrom: alice@example.com"
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertEqual(result, "See below.")

    def test_original_message_case_insensitive(self):
        body = "See below.\n\n-----ORIGINAL MESSAGE-----\nFrom: alice@example.com"
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertEqual(result, "See below.")

    def test_outlook_underscore_separator(self):
        body = "Quick note.\n\n________________________________\nFrom: alice@example.com"
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertEqual(result, "Quick note.")

    def test_long_dash_separator(self):
        # 10+ dashes on a line by themselves
        body = "Hi there.\n\n------------------------------\nOriginal content"
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertEqual(result, "Hi there.")

    def test_forwarded_message(self):
        body = "FYI.\n\n-----Forwarded Message-----\nFrom: someone@example.com"
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertEqual(result, "FYI.")


class TestStripReplyChainGlobexSeparator(unittest.TestCase):
    """Globex-style '---------' + 'Original Message ---------' pattern."""

    def test_dashes_only_line(self):
        # 9 dashes (just under the 10-char [-*_]{10,} threshold — caught by -{3,}$)
        body = "Thanks.\n\n---------\nOriginal Message ---------\nFrom: alice"
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertEqual(result, "Thanks.")

    def test_original_message_on_its_own_line(self):
        # "Original Message ---------" as the separator line
        body = "Got it.\n\nOriginal Message ---------\nFrom: alice"
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertEqual(result, "Got it.")

    def test_original_message_with_leading_dashes(self):
        body = "OK.\n\n--------- Original Message ---------\nSender: Bob"
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertEqual(result, "OK.")


class TestStripReplyChainOnWrote(unittest.TestCase):
    """'On <date> ... wrote:' reply attribution lines."""

    def test_single_line_on_wrote(self):
        body = "Will do.\n\nOn Mon, Jan 1, 2024, Alice wrote:\n> Original text"
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertEqual(result, "Will do.")

    def test_multiline_on_wrote_two_lines(self):
        body = (
            "Noted.\n\n"
            "On Monday, January 1, 2024 at 9:00 AM,\n"
            "Alice <alice@example.com> wrote:\n"
            "> Original text"
        )
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertEqual(result, "Noted.")

    def test_multiline_on_wrote_three_lines(self):
        body = (
            "Thanks.\n\n"
            "On Monday, January 1, 2024\n"
            "at 9:00 AM, Alice\n"
            "<alice@example.com> wrote:\n"
            "> Original text"
        )
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertEqual(result, "Thanks.")

    def test_on_wrote_with_timestamp(self):
        body = "Sure.\n\nOn 2/13/2024 1:48 PM, M. Vefa Bicakci wrote:\n> Text"
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertEqual(result, "Sure.")


class TestStripReplyChainOutlookInlineHeader(unittest.TestCase):
    """Outlook 'From: / Date: / To: / Subject:' inline reply header."""

    def test_outlook_from_date_to_subject(self):
        body = (
            "Short reply.\n\n"
            "From: Alice <alice@example.com>\n"
            "Date: Monday, April 29, 2024 at 8:55 AM\n"
            "To: Bob <bob@example.com>\n"
            "Subject: Re: Project update\n"
            "Original message body here."
        )
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertEqual(result, "Short reply.")

    def test_outlook_from_date_subject_no_to(self):
        # Date + Subject without To: — still 2 continuation matches, should strip.
        body = (
            "Quick answer.\n\n"
            "From: Alice <alice@example.com>\n"
            "Date: Monday, April 29, 2024\n"
            "Subject: Re: Something\n"
            "Old content."
        )
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertEqual(result, "Quick answer.")

    def test_from_at_start_of_body_not_stripped(self):
        # "From:" at the very start of the body (result is empty) should NOT trigger.
        body = "From: Alice <alice@example.com>\nDate: today\nTo: Bob\nHello Bob."
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertEqual(result, body)

    def test_from_without_continuation_not_stripped(self):
        # "From what I saw..." followed by unrelated lines — should not strip.
        body = "Reply text.\n\nFrom what I saw in the meeting,\nit was very clear.\nMore text."
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertEqual(result, body)


class TestStripReplyChainKoreanOutlookHeader(unittest.TestCase):
    """Korean Outlook inline reply headers (보낸 사람 / 날짜 / 받는 사람 / 주제)."""

    def test_korean_from_date_to_subject(self):
        body = (
            "안녕하세요,\n\n"
            "문서 첨부하여 메일 드립니다.\n\n"
            "보낸 사람: Kim, Jun <Jun.Kim@example.com>\n"
            "날짜: 목요일, 2024년 7월 4일 오후 1:23\n"
            "받는 사람: Jung, Hyun-woo <hw@example.com>\n"
            "주제: Re: Bi-weekly sync\n"
            "Original message content here."
        )
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertIn("안녕하세요", result)
        self.assertIn("문서 첨부하여", result)
        self.assertNotIn("Original message content", result)

    def test_korean_with_cc(self):
        # 참조 (CC) as a continuation header — still triggers with ≥2 matches.
        body = (
            "확인했습니다.\n\n"
            "보낸 사람: Alice <alice@example.com>\n"
            "날짜: 2024년 3월 15일\n"
            "참조: Bob <bob@example.com>\n"
            "주제: 프로젝트 업데이트\n"
            "Old content."
        )
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertIn("확인했습니다", result)
        self.assertNotIn("Old content", result)

    def test_korean_header_at_body_start_not_stripped(self):
        # 보낸 사람: at the very start (result empty) must not trigger.
        body = (
            "보낸 사람: Alice <alice@example.com>\n"
            "날짜: 2024년 3월 15일\n"
            "받는 사람: Bob\n"
            "실제 이메일 본문입니다."
        )
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertEqual(result, body)

    def test_korean_first_line_preserved_invariant(self):
        body = (
            "네, 맞습니다.\n\n"
            "보낸 사람: Kim <kim@example.com>\n"
            "날짜: 2024년 1월\n"
            "받는 사람: Lee <lee@example.com>\n"
            "주제: 회의\n"
            "이전 내용."
        )
        stripped = MailArchiveXLoader._strip_reply_chain(body)
        first_raw = next(l.strip() for l in body.splitlines() if l.strip())
        first_stripped = next(l.strip() for l in stripped.splitlines() if l.strip())
        self.assertEqual(first_raw, first_stripped)


class TestStripReplyChainAsianOutlookHeaders(unittest.TestCase):
    """Outlook inline reply headers for Asian languages.

    Covers: Korean, Japanese, Chinese Simplified, Chinese Traditional,
    Thai, Vietnamese, Indonesian/Malay.

    Each test verifies:
    - New reply content above the header block is preserved.
    - Content below the header block is stripped.
    - Header at the very start of the body (no prior content) is NOT stripped.
    """

    def _assert_strips_correctly(self, body: str, preserved: str, stripped: str) -> None:
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertIn(preserved, result, f"Expected preserved text not found: {preserved!r}")
        self.assertNotIn(stripped, result, f"Expected stripped text still present: {stripped!r}")

    def _assert_not_stripped_at_start(self, body: str) -> None:
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertEqual(result, body)

    # ── Korean ────────────────────────────────────────────────────────────────

    def test_korean_strips_correctly(self):
        body = "확인했습니다.\n\n보낸 사람: Kim <kim@example.com>\n날짜: 2024년 3월\n받는 사람: Lee\n주제: 회의\n이전 내용."
        self._assert_strips_correctly(body, "확인했습니다", "이전 내용")

    def test_korean_not_stripped_at_start(self):
        self._assert_not_stripped_at_start(
            "보낸 사람: Kim\n날짜: 2024\n받는 사람: Lee\n주제: 회의\n본문."
        )

    # ── Japanese ──────────────────────────────────────────────────────────────

    def test_japanese_strips_correctly(self):
        body = "ご確認ください。\n\n差出人: Tanaka <t@example.com>\n日付: 2024年3月\n宛先: Suzuki\n件名: 更新\n古いメール。"
        self._assert_strips_correctly(body, "ご確認ください", "古いメール")

    def test_japanese_not_stripped_at_start(self):
        self._assert_not_stripped_at_start(
            "差出人: Tanaka\n日付: 2024年\n宛先: Suzuki\n件名: テスト\nメール本文。"
        )

    # ── Chinese Simplified ────────────────────────────────────────────────────

    def test_chinese_simplified_strips_correctly(self):
        body = "请确认。\n\n发件人: Wang <wang@example.com>\n日期: 2024年3月\n收件人: Li\n主题: 项目更新\n旧邮件内容。"
        self._assert_strips_correctly(body, "请确认", "旧邮件内容")

    def test_chinese_simplified_not_stripped_at_start(self):
        self._assert_not_stripped_at_start(
            "发件人: Wang\n日期: 2024\n收件人: Li\n主题: 测试\n邮件正文。"
        )

    # ── Chinese Traditional ───────────────────────────────────────────────────

    def test_chinese_traditional_strips_correctly(self):
        body = "請確認。\n\n寄件人: Chen <chen@example.com>\n日期: 2024年3月\n收件者: Lin\n主旨: 專案更新\n舊郵件內容。"
        self._assert_strips_correctly(body, "請確認", "舊郵件內容")

    def test_chinese_traditional_not_stripped_at_start(self):
        self._assert_not_stripped_at_start(
            "寄件人: Chen\n日期: 2024\n收件者: Lin\n主旨: 測試\n郵件正文。"
        )

    # ── Thai ──────────────────────────────────────────────────────────────────

    def test_thai_strips_correctly(self):
        body = "รับทราบครับ\n\nจาก: Somchai <s@example.com>\nวันที่: 15 มีนาคม 2024\nถึง: Malee\nหัวเรื่อง: อัปเดตโครงการ\nเนื้อหาเดิม"
        self._assert_strips_correctly(body, "รับทราบครับ", "เนื้อหาเดิม")

    def test_thai_not_stripped_at_start(self):
        self._assert_not_stripped_at_start(
            "จาก: Somchai\nวันที่: 2024\nถึง: Malee\nหัวเรื่อง: ทดสอบ\nข้อความ"
        )

    # ── Vietnamese ────────────────────────────────────────────────────────────

    def test_vietnamese_strips_correctly(self):
        body = "Đã xác nhận.\n\nTừ: Nguyen <n@example.com>\nNgày: 15 tháng 3 2024\nTới: Tran\nChủ đề: Cập nhật dự án\nNội dung cũ."
        self._assert_strips_correctly(body, "Đã xác nhận", "Nội dung cũ")

    def test_vietnamese_not_stripped_at_start(self):
        self._assert_not_stripped_at_start(
            "Từ: Nguyen\nNgày: 2024\nTới: Tran\nChủ đề: Test\nNội dung."
        )

    # ── Indonesian / Malay ────────────────────────────────────────────────────

    def test_indonesian_strips_correctly(self):
        body = "Sudah dikonfirmasi.\n\nDari: Budi <b@example.com>\nTanggal: 15 Maret 2024\nKepada: Sari\nPerihal: Pembaruan proyek\nKonten lama."
        self._assert_strips_correctly(body, "Sudah dikonfirmasi", "Konten lama")

    def test_indonesian_not_stripped_at_start(self):
        self._assert_not_stripped_at_start(
            "Dari: Budi\nTanggal: 2024\nKepada: Sari\nPerihal: Tes\nIsi pesan."
        )


class TestStripReplyChainJapaneseOutlookHeader(unittest.TestCase):
    """Japanese Outlook inline reply headers (差出人 / 日付 / 宛先 / 件名)."""

    def test_japanese_from_date_to_subject(self):
        body = (
            "ご確認ください。\n\n"
            "差出人: Tanaka <tanaka@example.com>\n"
            "日付: 2024年3月15日 9:00\n"
            "宛先: Suzuki <suzuki@example.com>\n"
            "件名: Re: プロジェクト更新\n"
            "Original message content here."
        )
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertIn("ご確認ください", result)
        self.assertNotIn("Original message content", result)

    def test_japanese_header_at_body_start_not_stripped(self):
        body = (
            "差出人: Tanaka <tanaka@example.com>\n"
            "日付: 2024年3月15日\n"
            "宛先: Suzuki <suzuki@example.com>\n"
            "件名: テスト\n"
            "メール本文です。"
        )
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertEqual(result, body)

    def test_japanese_first_line_preserved_invariant(self):
        body = (
            "承知しました。\n\n"
            "差出人: Tanaka <tanaka@example.com>\n"
            "日付: 2024年1月\n"
            "宛先: Suzuki <suzuki@example.com>\n"
            "件名: 会議\n"
            "古いメール内容。"
        )
        stripped = MailArchiveXLoader._strip_reply_chain(body)
        first_raw = next(l.strip() for l in body.splitlines() if l.strip())
        first_stripped = next(l.strip() for l in stripped.splitlines() if l.strip())
        self.assertEqual(first_raw, first_stripped)


class TestStripReplyChainBottomPosting(unittest.TestCase):
    """Bottom-posted emails: quoted block at top, new reply at the bottom.

    Bottom-posting is common in technical mailing lists (Linux kernel, IETF,
    open-source communities).  The structure is:

        > Alice wrote: original question
        > more quoted lines

        New reply at the very bottom.

    The '>' quoted prefix must be discarded; only the trailing reply is kept.

    Known gap: inline/interleaved posting (reply text woven between individual
    quoted paragraphs) is not handled.  See docs/EMAIL_PREPROCESSING.md.
    """

    def test_simple_bottom_post(self):
        # Single '>' context line + reply: top-posting keeps both (the quoted
        # line provides useful context for RAG).  The reply must be present.
        body = "> Alice wrote: let's meet Tuesday\n\nYes, that works for me."
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertIn("Yes, that works for me.", result)

    def test_bottom_post_multiple_quoted_lines(self):
        body = "> quoted line 1\n> quoted line 2\n> quoted line 3\n\nMy reply here."
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertEqual(result, "My reply here.")

    def test_bottom_post_multiline_reply(self):
        # Single '>' line: top-posting keeps the quoted context + full reply.
        body = "> Original question\n\nFirst line of reply.\nSecond line of reply.\n\nThanks."
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertIn("First line of reply.", result)
        self.assertIn("Second line of reply.", result)
        self.assertIn("Thanks.", result)

    def test_bottom_post_with_blank_lines_in_quoted_prefix(self):
        body = "> line 1\n>\n> line 2\n\nBottom reply."
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertEqual(result, "Bottom reply.")

    def test_bottom_post_reply_not_lost(self):
        """Core invariant: the new reply must always appear in the output.

        For a single '>' context line, top-posting keeps it alongside the reply
        (the quoted line provides useful context for RAG retrieval).  The reply
        content itself must never be absent from the result.
        """
        body = "> quoted original\n\nThis is the actual reply."
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertIn("This is the actual reply.", result)

    def test_bottom_post_deeply_nested_quotes(self):
        body = ">> deeply nested\n>> more nested\n> outer quote\n\nFinal reply."
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertEqual(result, "Final reply.")

    def test_forwarded_only_not_treated_as_bottom_post(self):
        """Emails starting with -----Forwarded----- (not '>') must be preserved in full."""
        body = (
            "---------- Forwarded message ----------\n"
            "From: Alice <alice@example.com>\n"
            "Subject: Q3 Report\n\n"
            "Please find the report attached."
        )
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertEqual(result, body)

    def test_bottom_post_stripped_body_shorter_than_raw(self):
        """Stripping a bottom-posted email must produce a shorter (or equal) result."""
        body = "> quoted line 1\n> quoted line 2\n\nShort reply."
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertLessEqual(len(result), len(body))

    def test_bottom_post_only_quoted_lines_no_reply(self):
        """Email consisting entirely of quoted lines — return full body (no real reply found)."""
        body = "> quoted only\n> nothing else"
        result = MailArchiveXLoader._strip_reply_chain(body)
        # Falls back to full body — no content to lose
        self.assertIn("quoted only", result)


class TestStripReplyChainLeadingAttribution(unittest.TestCase):
    """Bug #3: bottom-posted emails that LEAD with an attribution line.

    Structure:
        On <date>, <Person> wrote:
        > quoted block
        > more quoted lines

        Actual reply content here.

    Before the fix, _strip_reply_chain appended the attribution line to
    result (has_real_content was False so the attribution check was skipped),
    then broke at the first '>' line — leaving *only* the attribution in
    top_result.  Because top_nonquoted was non-empty (the attribution line),
    the bottom-posting fallback never triggered and the real reply was lost.

    After the fix, a result that ends with "wrote:" and has ≤3 non-blank
    lines is treated as attribution-only, forcing a bottom-extraction pass.
    """

    def test_leading_attribution_reply_preserved(self):
        """Core case: leading attribution + quoted block + reply below."""
        body = (
            "On 2024-01-15, Alice <alice@example.com> wrote:\n"
            "> Can you confirm the meeting time?\n"
            "\n"
            "Yes, 3 PM works for me."
        )
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertIn("Yes, 3 PM works for me.", result)
        self.assertNotIn("Can you confirm", result)

    def test_leading_attribution_multiline_reply_preserved(self):
        """Multi-line reply content after attribution + quoted block."""
        body = (
            "On 2024-03-01, Bob wrote:\n"
            "> Please send the report.\n"
            "> It is urgent.\n"
            "\n"
            "Hi Bob,\n"
            "I have attached the report.\n"
            "Let me know if you need anything else."
        )
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertIn("Hi Bob", result)
        self.assertIn("I have attached the report.", result)
        self.assertNotIn("Please send the report.", result)

    def test_leading_attribution_wrapped_across_two_lines(self):
        """Attribution that wraps: 'On <date>,\\n<Name> wrote:'."""
        body = (
            "On 2024-06-10,\n"
            "Charlie <charlie@example.com> wrote:\n"
            "> Here is the original question.\n"
            "\n"
            "Thanks for reaching out."
        )
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertIn("Thanks for reaching out.", result)
        self.assertNotIn("Here is the original question.", result)

    def test_leading_attribution_no_bottom_reply_returns_full_body(self):
        """Attribution at top but no real reply below — return full body (no content to lose)."""
        body = (
            "On 2024-01-15, Alice wrote:\n"
            "> Only quoted content here.\n"
            "> Nothing below."
        )
        result = MailArchiveXLoader._strip_reply_chain(body)
        # No real reply found — full body preserved.
        self.assertIn("Only quoted content here.", result)

    def test_real_content_above_attribution_not_affected(self):
        """Existing top-posting case: real content above attribution is unaffected."""
        body = (
            "My reply is above.\n"
            "\n"
            "On 2024-01-15, Alice wrote:\n"
            "> quoted original"
        )
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertIn("My reply is above.", result)
        self.assertNotIn("quoted original", result)

    def test_attribution_only_result_ends_without_wrote(self):
        """A non-attribution lead line is not confused with the new trigger."""
        body = (
            "Forwarded by Dave:\n"
            "> Some quoted content\n"
            "\n"
            "Bottom reply here."
        )
        result = MailArchiveXLoader._strip_reply_chain(body)
        # "Forwarded by Dave:" does not end in "wrote:" → existing top-posting
        # path keeps it; real content (the lead line) is preserved.
        self.assertIn("Forwarded by Dave:", result)


class TestHTMLTextExtractor(unittest.TestCase):
    """_HTMLTextExtractor converts email HTML to plain text."""

    def _extract(self, html: str) -> str:
        ex = _HTMLTextExtractor()
        ex.feed(html)
        return ex.get_text()

    def test_plain_text_passthrough(self):
        self.assertEqual(self._extract("Hello world"), "Hello world")

    def test_br_becomes_newline(self):
        result = self._extract("Line one<br>Line two")
        self.assertIn("\n", result)
        self.assertIn("Line one", result)
        self.assertIn("Line two", result)

    def test_p_tag_introduces_newline(self):
        result = self._extract("<p>First</p><p>Second</p>")
        self.assertIn("First", result)
        self.assertIn("Second", result)

    def test_style_tag_content_stripped(self):
        result = self._extract("<style>.foo { color: red; }</style>Hello")
        self.assertNotIn("color", result)
        self.assertIn("Hello", result)

    def test_script_tag_content_stripped(self):
        result = self._extract("<script>alert('x')</script>Safe text")
        self.assertNotIn("alert", result)
        self.assertIn("Safe text", result)

    def test_nested_tags(self):
        result = self._extract("<div><p>Hello <strong>world</strong></p></div>")
        self.assertIn("Hello", result)
        self.assertIn("world", result)

    def test_html_entity_amp(self):
        result = self._extract("A &amp; B")
        self.assertIn("&", result)

    def test_html_entity_nbsp(self):
        result = self._extract("A&nbsp;B")
        self.assertIn("A", result)
        self.assertIn("B", result)

    def test_realistic_outlook_html(self):
        html = (
            "<html><body>"
            "<div>Hi team,</div>"
            "<div><br></div>"
            "<div>Please review the attached.</div>"
            "<style>body { font-family: Arial; }</style>"
            "<div>________________________________</div>"
            "<div>From: Alice</div><div>Date: Monday</div>"
            "</body></html>"
        )
        result = self._extract(html)
        self.assertIn("Hi team", result)
        self.assertIn("Please review", result)
        self.assertNotIn("font-family", result)

    def test_html_reply_chain_stripped_after_extraction(self):
        # Full pipeline: HTML-only email with Outlook underscore separator.
        html = (
            "<div>Short reply here.</div>"
            "<div>________________________________</div>"
            "<div>From: Alice</div><div>Original content here.</div>"
        )
        ex = _HTMLTextExtractor()
        ex.feed(html)
        plain = ex.get_text()
        result = MailArchiveXLoader._strip_reply_chain(plain)
        self.assertIn("Short reply here", result)
        self.assertNotIn("Original content", result)


class TestStripReplyChainInvariants(unittest.TestCase):
    """Invariant: stripping only ever removes content from the tail.

    The first non-blank line of the stripped body must always equal the first
    non-blank line of the raw body.  If this is ever violated it means stripping
    fired too early and consumed real email content.
    """

    def _first_nonblank(self, text: str) -> str:
        for line in text.splitlines():
            if line.strip():
                return line.strip()
        return ""

    def _assert_first_line_preserved(self, raw: str) -> None:
        stripped = MailArchiveXLoader._strip_reply_chain(raw)
        if not raw.strip():
            return  # empty input is fine
        self.assertEqual(
            self._first_nonblank(stripped),
            self._first_nonblank(raw),
            f"First line changed after stripping.\n"
            f"  raw first line   : {self._first_nonblank(raw)!r}\n"
            f"  stripped first   : {self._first_nonblank(stripped)!r}",
        )

    def test_invariant_plain_no_reply(self):
        self._assert_first_line_preserved(
            "Hi team,\n\nPlease review the attached doc.\n\nThanks, Alice"
        )

    def test_invariant_with_quoted_lines(self):
        self._assert_first_line_preserved(
            "Agreed.\n\n> Original message here.\n> More quoted text."
        )

    def test_invariant_with_outlook_separator(self):
        self._assert_first_line_preserved(
            "See below.\n\n________________________________\nFrom: alice@example.com"
        )

    def test_invariant_with_on_wrote(self):
        self._assert_first_line_preserved("Will do.\n\nOn Mon Jan 1 2024, Alice wrote:\n> Sure.")

    def test_invariant_with_outlook_inline_header(self):
        self._assert_first_line_preserved(
            "Quick reply.\n\nFrom: Alice <a@b.com>\nDate: Monday\nTo: Bob\nSubject: Re: X\nOld body."
        )

    def test_invariant_with_forwarded_message(self):
        self._assert_first_line_preserved(
            "FYI.\n\n-----Forwarded Message-----\nFrom: someone@example.com"
        )

    def test_stripped_body_never_longer_than_raw(self):
        """Stripping must never ADD content."""
        bodies = [
            "Short email.",
            "Hi.\n\n> Quoted.",
            "Text.\n\n-----Original Message-----\nOld stuff.",
            "Reply.\n\nFrom: x@y.com\nDate: Mon\nTo: z\nSubject: S\nOld.",
        ]
        for body in bodies:
            stripped = MailArchiveXLoader._strip_reply_chain(body)
            self.assertLessEqual(
                len(stripped),
                len(body),
                f"Stripping increased length for body: {body!r}",
            )

    def test_gt_in_code_not_stripped_if_inline(self):
        """'>' embedded inside a sentence should not trigger stripping."""
        # A line like "if x > 0:" does NOT start with '>' so it is safe.
        body = "The condition is if x > 0 then proceed.\n\nSee the code above."
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertEqual(result, body)

    def test_forwarded_only_email_preserved(self):
        """Email whose body IS the forwarded message (no new content above) must not be wiped."""
        body = (
            "---------- Forwarded message ----------\n"
            "From: Alice <alice@example.com>\n"
            "Subject: Q3 Report\n\n"
            "Please find the report attached."
        )
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertEqual(result, body)

    def test_reply_only_email_separator_on_first_line_preserved(self):
        """'-----Original Message-----' as the very first line must be preserved."""
        body = "-----Original Message-----\nFrom: Alice\n\nOriginal content here."
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertEqual(result, body)

    def test_multiple_reply_patterns_stops_at_first(self):
        """Only the first reply boundary is used — content between two chains is not kept."""
        body = (
            "New reply here.\n\n"
            "-----Original Message-----\n"
            "First chain content.\n\n"
            "-----Original Message-----\n"
            "Second chain content."
        )
        result = MailArchiveXLoader._strip_reply_chain(body)
        self.assertEqual(result, "New reply here.")
        self.assertNotIn("First chain", result)
        self.assertNotIn("Second chain", result)


if __name__ == "__main__":
    unittest.main()
