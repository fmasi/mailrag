"""Decide what ``sync --status`` should warn about, given the ledger's last rows.

Split out of the CLI's print loop because it is the only part of ``--status``
that makes a judgement, and inline in a loop it had no test — which is how a 48 h
staleness threshold survived against a 4 h schedule, unable to notice a whole
missed day.

The design constraint comes from the hardware, not the code: this runs on a
laptop that legitimately spends nights closed and unplugged, where macOS parks
launchd's timers entirely. **No runs during that window is correct behaviour.**
So age alone can never mean "broken" — it can only mean "stale", and it has to
be tolerant enough that a normal night stays silent, or the warning becomes noise
and gets ignored.

Everything that IS unambiguously broken therefore warns on its own evidence
instead of waiting for a clock:

* a newest run that failed — the schedule is firing and not succeeding, which
  looks perfectly healthy to an age check because the newest attempt is always
  recent;
* a run that never finished — the normal shape of a tick killed by sleep or
  shutdown, which leaves ``completed_at`` NULL;
* never once having succeeded — an install that has never worked at all.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Mapping, Optional

# Longer than any ordinary closed-lid night, shorter than a lost day. The old
# value (48 h) was chosen when the interval was 12 h; against the 4 h interval
# actually installed it allowed twelve consecutive missed ticks to pass in
# silence.
STALE_AFTER_HOURS = 24

# A run still in flight is not a failure, and 'ok' is the success. Anything else
# ('partial', 'error', …) means the tick ran and did not fully work.
_IN_FLIGHT = "running"
_SUCCESS = "ok"


def _field(row: Mapping, key: str):
    """Read *key* from a dict OR a ``sqlite3.Row``.

    The ledger hands out ``sqlite3.Row``, which supports subscripting but not
    ``.get()`` — a difference invisible to tests that pass dicts, and one that
    surfaces as AttributeError on the first genuinely failed run, i.e. exactly
    when the health check is the thing you are relying on. Row raises IndexError
    for an absent column, dict raises KeyError.
    """
    try:
        return row[key]
    except (KeyError, IndexError):
        return None


def staleness_hours(completed_at: Optional[str], now: datetime) -> Optional[float]:
    """Hours since *completed_at*, or None if it is missing/unparseable.

    None rather than 0.0 on bad input: a missing timestamp is an absence of
    evidence, and treating it as "just synced" would silence the check exactly
    when the ledger is damaged.
    """
    if not completed_at:
        return None
    try:
        return (now - datetime.fromisoformat(completed_at)).total_seconds() / 3600.0
    except (TypeError, ValueError):
        return None


def health_warnings(
    *,
    last_run: Optional[Mapping],
    last_success: Optional[Mapping],
    now: datetime,
    stale_after_hours: int = STALE_AFTER_HOURS,
) -> List[str]:
    """Warnings for one account, worst-first. Empty means healthy.

    Pure so it can be tested without a database or a clock. The caller owns all
    printing and the informational lines (folder cursors, counts, the last run's
    own summary); this returns only the lines that assert something is wrong.
    """
    warnings: List[str] = []

    # Never run at all is a fresh install, not a fault — the caller prints
    # "last run: never" and that is the whole story.
    if last_run is None:
        return warnings

    status = _field(last_run, "status") or ""

    if status == _IN_FLIGHT:
        warnings.append("the last run never finished (killed or still running)")
    elif status != _SUCCESS:
        # Deliberately independent of age: this is the failure mode an age check
        # structurally cannot see.
        warnings.append(
            f"the last run ended in '{status}' — the schedule is firing but not succeeding"
        )

    if last_success is None:
        warnings.append("no run has EVER completed successfully")
        return warnings

    stale = staleness_hours(_field(last_success, "completed_at"), now)
    if stale is not None and stale > stale_after_hours:
        # Measured from the last SUCCESS, not the last attempt: 120 consecutive
        # failed ticks would otherwise look fresh because the newest attempt is
        # two minutes old.
        warnings.append(
            f"index is stale — last SUCCESSFUL sync was {stale:.0f}h ago "
            f"(expected within {stale_after_hours}h whenever the machine is awake)"
        )

    return warnings
