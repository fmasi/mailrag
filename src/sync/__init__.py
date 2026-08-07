"""Continuous sync: keep an indexed collection fresh (issue #101).

mailrag is built from a one-shot backup export, so an index is a snapshot that
silently rots — ingest to 1 January and nothing after it is ever searchable. This
package closes that gap for **any** account on **any** provider.

The design rests on one observation: every existing pipeline stage is driven by
``.eml`` files on disk. So a syncer never needs to know about chunking, rubrics or
vectors. Its whole job is:

1. ask a :class:`~src.sync.sources.MessageSource` what is new,
2. spool those messages as ``.eml`` files into the corpus tree,
3. let the existing stages run over the delta.

Because the Pass-2 cache is content-addressed, re-running those stages over
old+new mail costs LLM calls only for the new mail, and the cleaning rubric stays
automatically consistent with the already-cleaned corpus.

Provider specifics live entirely behind ``MessageSource``: an implementation
contributes only *how you enumerate new mail* and *what a cursor is*.
"""
