# Why this runs on your own hardware

The first time I pointed a cloud AI at my inbox it felt like a superpower. Then I
thought about what it had actually required: handing my entire email history to
someone else's servers to make it searchable. For contracts, receipts, and the record
of who agreed to what, that is a bad trade at any level of vendor trust.

So mailrag was built the other way round. Open models, your machine, your disk, and
nothing that has to leave your network.

## How sensitive is email? Look at what's public

Here is the argument I find most persuasive, and it comes from the data rather than from
principle.

As far as I have been able to establish, nobody has ever published their own private
mailbox. Every public email corpus of any size exists because someone **lost control**
of one. Enron came out of a federal investigation. The Avocado collection came out of a
company's liquidation. The FOIA sets came out of public-records law, and the assorted
political archives came out of leaks.

There are consented exceptions, and they prove the point. Mailing lists like Apache,
LKML and W3C are public by design. They are also nothing like private correspondence:
no terse replies, no shared context you had to be in the room for, no half-sentence that
only makes sense as the fourth message in a chain.

That is why a dataset from **2001** is still the field standard twenty-five years later,
and why the nearest alternative is another defunct company's mail sitting behind a
licence. An entire research area works on the mail of people who never agreed to any of
it, because everyone who could consent has quietly decided not to.

*If you know of a corpus I have missed, please
[open an issue](https://github.com/fmasi/mailrag/issues). I would genuinely like to be
wrong about this.*

## What the scarcity implies

The absence of consented data is the argument for the architecture. If email is too
sensitive to leave its owner, and twenty-five years of empty shelves says it is, then
you don't move the mailbox to the model. You move the model to the mailbox.

That costs something, and the honest version of the trade is worth stating. Local
retrieval means open embedders instead of a hosted API, a Qdrant container on your own
disk, and roughly 8 GB of RAM or VRAM if you also want a local model writing the
answers. Indexing 32,000 emails takes hours on a laptop rather than minutes on someone
else's cluster.

What it buys is that the question "where did my mail go" has a boring answer.

## Email as private context

There is a second reason, and it took me longer to notice than the privacy one.

These are not only emails. They are **context**. A faithful record of what was said and
agreed is what an AI agent needs in order to be useful about your work rather than
generically competent. Kept private and self-owned, that record gives you total recall
without renting your memory to a vendor.

mailrag is one such source, for email. [parley](https://github.com/fmasi/parley) is
another, for calls and meetings, and it needs completely different machinery (on-device
audio and diarization). The two don't talk to each other. My agents know about both and
reach for whichever fits the question. The goal was never a single application. It's a
private, open stack of context that I own.

## Related reading

- [`CLAIMS.md`](CLAIMS.md) tracks which numbers a stranger can reproduce and which stay
  author-reported, which is the same honesty problem in a different form.
- [`BENCHMARK.md`](BENCHMARK.md) covers `make bench`, the public retrieval number that
  needs no key and no private data.
- [`BACKENDS.md`](BACKENDS.md) covers the two seams where cloud is an option rather than
  a requirement, and what you give up at each.
