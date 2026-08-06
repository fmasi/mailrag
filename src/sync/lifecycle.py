"""The message lifecycle, as an explicit state machine.

This module exists because the implicit version did not work. For three
iterations the lifecycle was two write-once nullable timestamps (``judged_at``,
``indexed_at``) plus ``error`` plus ``attempts``, with the transition logic
spread across ``judge_pending``, ``index_pending`` and ``abandon``. Four
independent review rounds each found the same class of defect, and always for
the same reason: a transition was applied on one code path and forgotten on the
adjacent one. Re-queueing an already-indexed message after it was finally judged
was implemented on the judge success path and missing from the judge *failure*
path; clearing an error was done in one place and not another; a message could
be marked done by a branch that had written nothing.

None of those are possible to express here. There is one transition table and
one function that applies it. A caller says *what happened* — ``JUDGED``,
``INDEXED``, ``ABANDONED`` — and the resulting state is a property of the table,
not of the caller's diligence.

The states
----------

``NEW``
    Spooled, nothing else. Needs judging and indexing.
``INDEXED_UNJUDGED``
    In the vector store, but embedded before its summary existed. This is a
    *distinct* state rather than a flag combination, because it is the one the
    old design kept losing: it must return to the index queue once judged, and
    naming it makes that a table entry instead of a thing to remember.
``JUDGED``
    Has its Pass-2 judgment; awaiting (re-)indexing.
``DONE``
    Judged and indexed. Terminal under normal operation.
``ABANDONED``
    Given up on, with a reason. Terminal but **reversible** — see ``REQUEUED``.
    Reversibility is deliberate: an irreversible terminal state turns any bug in
    the abandon logic into permanent, unrecoverable data loss, which is exactly
    what happened when an LLM outage was mistaken for a per-message failure.

The events
----------

``JUDGED`` / ``INDEXED``
    A stage completed for this message.
``ABANDONED``
    This message cannot succeed by retrying (its spool file is gone, the chunker
    rejects it, it has failed too many times).
``REQUEUED``
    An operator is putting an abandoned message back in play.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Tuple


class State(str, Enum):
    NEW = "new"
    INDEXED_UNJUDGED = "indexed_unjudged"
    JUDGED = "judged"
    DONE = "done"
    ABANDONED = "abandoned"


class Event(str, Enum):
    JUDGED = "judged"
    INDEXED = "indexed"
    ABANDONED = "abandoned"
    REQUEUED = "requeued"


# (state, event) -> state. A pair that is absent is a NO-OP, not an error: stages
# legitimately re-report work (a cached Pass-2 hit, an idempotent re-index), and
# a lifecycle that raised on those would make every caller defensive again.
_TABLE: Dict[Tuple[State, Event], State] = {
    (State.NEW, Event.JUDGED): State.JUDGED,
    (State.NEW, Event.INDEXED): State.INDEXED_UNJUDGED,
    # The transition the old design kept dropping: being judged after having been
    # indexed puts the message BACK in the index queue, so the summary reaches
    # the vector. It is now impossible to apply this on one branch and not another.
    (State.INDEXED_UNJUDGED, Event.JUDGED): State.JUDGED,
    (State.JUDGED, Event.INDEXED): State.DONE,
    # Judging an already-done message (a re-judge with a changed rubric) sends it
    # back for re-indexing rather than silently keeping the stale vector.
    (State.DONE, Event.JUDGED): State.JUDGED,
    (State.NEW, Event.ABANDONED): State.ABANDONED,
    (State.INDEXED_UNJUDGED, Event.ABANDONED): State.ABANDONED,
    (State.JUDGED, Event.ABANDONED): State.ABANDONED,
    (State.ABANDONED, Event.REQUEUED): State.NEW,
}

# States still owing work, per stage. Derived from the enum rather than from
# "timestamp IS NULL" so a new state cannot silently fall out of a queue.
NEEDS_JUDGE = frozenset({State.NEW, State.INDEXED_UNJUDGED})
NEEDS_INDEX = frozenset({State.NEW, State.JUDGED})
# What may be indexed when a judge stage IS configured: waiting for judgments
# that are coming. NEW is excluded so mail is not embedded without its summary.
NEEDS_INDEX_AFTER_JUDGE = frozenset({State.JUDGED})
TERMINAL = frozenset({State.DONE, State.ABANDONED})


def next_state(current: State, event: Event) -> State:
    """Apply *event* to *current*. Unknown pairs leave the state unchanged."""
    return _TABLE.get((current, event), current)


def is_pending(state: State, stage: str, *, judge_configured: bool = True) -> bool:
    """Does a message in *state* still owe work for *stage*?

    ``judge_configured`` exists because "indexable" genuinely differs between
    deployments: with no LLM configured there is no judge stage to wait for, so
    NEW mail is indexed directly (and re-indexed later if a judgment ever
    arrives). With one configured, NEW mail waits.
    """
    if stage == "judged":
        return state in NEEDS_JUDGE
    if stage == "indexed":
        return state in (NEEDS_INDEX_AFTER_JUDGE if judge_configured else NEEDS_INDEX)
    raise ValueError(f"unknown stage {stage!r}")
