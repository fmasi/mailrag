"""The message lifecycle state machine (src/sync/lifecycle.py).

These tests exist because the implicit version — two write-once nullable
timestamps with transition logic spread across three functions — produced the
same class of defect in four consecutive review rounds. The properties below are
the ones that kept breaking, now stated once against the transition table
instead of being re-checked at every call site.
"""

from __future__ import annotations

import itertools
import os
import shutil
import tempfile
import unittest

from src.sync.lifecycle import (
    NEEDS_INDEX,
    NEEDS_INDEX_AFTER_JUDGE,
    NEEDS_JUDGE,
    TERMINAL,
    Event,
    State,
    is_pending,
    next_state,
)
from src.sync.state import SyncState


class TestTransitionTable(unittest.TestCase):
    def test_a_new_message_owes_judging(self):
        self.assertIn(State.NEW, NEEDS_JUDGE)

    def test_indexing_before_judging_is_its_own_state(self):
        """Not a flag combination. Naming it is what makes the re-queue a table
        entry rather than something each caller must remember."""
        self.assertEqual(next_state(State.NEW, Event.INDEXED), State.INDEXED_UNJUDGED)
        self.assertIn(State.INDEXED_UNJUDGED, NEEDS_JUDGE)

    def test_judging_an_already_indexed_message_returns_it_to_the_index_queue(self):
        """THE transition the old design kept dropping on one branch or another —
        the summary must be able to reach the vector."""
        self.assertEqual(next_state(State.INDEXED_UNJUDGED, Event.JUDGED), State.JUDGED)
        self.assertIn(State.JUDGED, NEEDS_INDEX)
        self.assertIn(State.JUDGED, NEEDS_INDEX_AFTER_JUDGE)

    def test_the_happy_path_reaches_done(self):
        s = State.NEW
        s = next_state(s, Event.JUDGED)
        s = next_state(s, Event.INDEXED)
        self.assertEqual(s, State.DONE)

    def test_re_judging_a_done_message_sends_it_back_for_re_indexing(self):
        """A changed rubric must not leave a stale vector in place."""
        self.assertEqual(next_state(State.DONE, Event.JUDGED), State.JUDGED)

    def test_abandoning_is_reversible(self):
        """An irreversible terminal state turns any bug in the abandon logic into
        permanent data loss — which is exactly what happened."""
        self.assertEqual(next_state(State.NEW, Event.ABANDONED), State.ABANDONED)
        self.assertEqual(next_state(State.ABANDONED, Event.REQUEUED), State.NEW)

    def test_unknown_pairs_are_no_ops_not_errors(self):
        """Stages legitimately re-report work (a cached judgment, an idempotent
        re-index); raising would push defensiveness back into every caller."""
        self.assertEqual(next_state(State.DONE, Event.INDEXED), State.DONE)
        self.assertEqual(next_state(State.NEW, Event.REQUEUED), State.NEW)

    def test_every_state_event_pair_stays_inside_the_enum(self):
        for state, event in itertools.product(State, Event):
            self.assertIsInstance(next_state(state, event), State)

    def test_terminal_states_owe_no_work(self):
        for state in TERMINAL:
            self.assertFalse(is_pending(state, "judged"))
            self.assertFalse(is_pending(state, "indexed"))
            self.assertFalse(is_pending(state, "indexed", judge_configured=False))

    def test_no_state_owes_work_it_cannot_reach(self):
        """Every non-terminal state must be able to make progress, or it is a
        trap — the 'stuck' failure mode from the audits."""
        for state in State:
            if state in TERMINAL:
                continue
            reachable = {next_state(state, e) for e in Event} - {state}
            self.assertTrue(reachable, f"{state} cannot progress")

    def test_new_mail_waits_for_judging_only_when_a_judge_stage_exists(self):
        self.assertFalse(is_pending(State.NEW, "indexed", judge_configured=True))
        self.assertTrue(is_pending(State.NEW, "indexed", judge_configured=False))

    def test_an_unknown_stage_is_rejected(self):
        with self.assertRaises(ValueError):
            is_pending(State.NEW, "embedded")


class TestStateStoreIntegration(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)
        self.state = SyncState(os.path.join(self.d, "s.db"))
        self.addCleanup(self.state.close)
        self.state.record_fetched("a", message_key="k1")

    def _state(self, key="k1"):
        return self.state.rows("a", [key])[0]["state"]

    def test_the_ledger_walks_the_same_path(self):
        self.assertEqual(self._state(), State.NEW.value)
        self.state.mark_indexed("a", ["k1"])
        self.assertEqual(self._state(), State.INDEXED_UNJUDGED.value)
        self.state.mark_judged("a", ["k1"])
        self.assertEqual(self._state(), State.JUDGED.value)
        self.state.mark_indexed("a", ["k1"])
        self.assertEqual(self._state(), State.DONE.value)

    def test_a_successful_stage_clears_a_recorded_error(self):
        self.state.record_error("a", "k1", "one-off timeout")
        self.assertEqual(self.state.counts("a")["errors"], 1)
        self.state.mark_judged("a", ["k1"])
        self.assertEqual(self.state.counts("a")["errors"], 0)

    def test_abandon_records_a_reason_and_is_visible(self):
        self.state.abandon("a", ["k1"], "spool file is missing")
        self.assertEqual(self._state(), State.ABANDONED.value)
        rows = self.state.abandoned("a")
        self.assertEqual(len(rows), 1)
        self.assertIn("missing", rows[0]["error"])

    def test_requeue_restores_a_message_to_the_front_of_the_pipeline(self):
        self.state.mark_judged("a", ["k1"])
        self.state.abandon("a", ["k1"], "gave up")
        self.state.requeue("a")
        self.assertEqual(self._state(), State.NEW.value)
        self.assertEqual(self.state.counts("a")["errors"], 0)
        self.assertEqual(self.state.counts("a")["pending_judge"], 1)

    def test_requeue_with_no_keys_restores_everything_abandoned(self):
        self.state.record_fetched("a", message_key="k2")
        self.state.abandon("a", ["k1", "k2"], "gave up")
        self.assertEqual(self.state.requeue("a"), 2)

    def test_applying_an_event_to_an_unknown_message_is_a_no_op(self):
        self.assertEqual(self.state.mark_judged("a", ["nope"]), 0)

    def test_state_survives_reopening(self):
        path = os.path.join(self.d, "reopen.db")
        with SyncState(path) as s:
            s.record_fetched("a", message_key="k1")
            s.mark_indexed("a", ["k1"])
        with SyncState(path) as s:
            self.assertEqual(s.rows("a", ["k1"])[0]["state"], State.INDEXED_UNJUDGED.value)


if __name__ == "__main__":
    unittest.main()
