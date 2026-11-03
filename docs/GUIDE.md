# A friendly guide to mailrag

What mailrag does, what to expect when you run it, and how to choose, without assuming
you have read the code. For the terse verb-by-verb reference, see [`VERBS.md`](VERBS.md).

## The big picture

mailrag turns a pile of `.eml` files into a question-answering assistant over your mail.
The work splits into two questions:

1. **Scope.** Which mail do we even consider? Your work folders, or everything?
2. **Noise and context.** Of that mail, what is junk, and what context should be
   attached so retrieval works?

The second question is a funnel of cheap-to-expensive steps, and the golden rule is that
**only one step, `index`, deletes anything.** Everything before it tags, measures,
judges, or just looks. Your original `.eml` files stay on disk throughout, so every
choice is reversible.

```mermaid
flowchart LR
    scope[scope<br/>pick folders] --> measure[measure<br/>chunk size]
    measure --> tag[tag<br/>regex flags · free]
    tag --> scan[scan<br/>no-LLM clusters · cheap]
    scan --> calibrate[calibrate<br/>sample gate · small LLM]
    calibrate --> summarize[summarize<br/>context · BIG LLM]
    summarize --> index[index<br/>embed + store · only this drops]
    index --> attach[attach<br/>store attachment files · CPU]
    scan -. recommends .-> persona{{which persona?}}
```

One cost dominates everything else: the **LLM pass over each email**. So the lever that
matters is how many emails you send to the LLM. Embedding, regex and clustering are all
cheap by comparison.

## Pick a persona, which is the one decision that matters

A **persona** is a recipe, meaning which verbs run in what order, for a given
budget-against-quality trade. You never wire verbs by hand. You pick a persona and
mailrag walks it.

| Persona | Use it when… | LLM cost | Quality |
|---------|--------------|----------|---------|
| **`llm-none`** | You want it fast and don't need summary context. | none | noisy |
| **`llm-verify`** | You want great quality *and* your corpus has a lot of **obvious** noise to skip cheaply.¹ | medium | high, safe |
| **`llm-all`** | You want the cleanest result and the LLM to look at every email. | high | best |

¹ `llm-verify` earns its place when `scan` shows a lot of *obvious* noise. Each LLM call is
dominated by reading the email body, so its saving over `llm-all` comes mainly from the
summaries it skips on the dropped bulk, plus the safety of LLM-confirmed drops, rather than
from a big cut in body processing. With little obvious noise, `llm-all` is simpler. Before
any drop, `prune` shows a sample of what it will blacklist and asks.

Not sure which? Run `scan` first. It is free, spends no LLM, and it tells you:

```text
$ ./mailrag scan --profile mybox.profile.json
scan: 186 threads / 188 emails, k=10, seed=11, baseline tag rate=0.022
┏━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ id ┃ threads ┃ score ┃ tag_lift ┃ top_sender (share)          ┃ tight ┃
┡━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ 0  │ 4       │ 0.77  │ 23.25    │ support@tripit.com (50%)    │ 0.80  │
│ 1  │ 27      │ 0.61  │ 1.72     │ no-reply@accounts.google.com│ 0.80  │
│ 9  │ 6       │ 0.61  │ 0.00     │ d.marsh@northwind.example   │ 0.82  │
│ …  │         │       │          │                             │       │
└────┴─────────┴───────┴──────────┴──────────────────────────────┴───────┘
recommended persona: llm-verify — 38% of threads sit in obvious noise pockets —
trim them with a cheap LLM check (llm-verify), or skip the LLM entirely (llm-none)
```

A `tag_lift` well above 1 alongside a dominant `no-reply@` or automated sender marks an
obvious noise pocket. The bigger that share, the more a budget persona saves you.

## What to expect from the wizard

`./mailrag wizard --profile mybox.profile.json` opens a full-screen TUI built with
[Textual](https://textual.textualize.io/), which walks the whole pipeline as a sequence
of screens. A breadcrumb at the top always shows where you are:

```text
✓ Welcome  ✓ Persona  ✓ Model  [ Scope ]  Review  Run
```

1. **Welcome.** What the wizard is about to do, in three lines, and the profile it
   works on: mail folder, junk rules (the rubric), search index (the collection), plus
   the recommended recipe if you ran `scan`. `enter` begins.

   ![Welcome screen — the profile facts and the scan-recommended persona](images/tui/welcome.svg)
2. **Persona.** The personas listed by plain names: *Quick, no AI* (`llm-none`),
   *Balanced* (`llm-verify`) and *Thorough, AI reads everything* (`llm-all`). The
   highlighted one is previewed on the right, every step in plain words with a
   colour-coded cost badge from `free` to `GPU, minutes to hours`, and the `scan` recommendation
   is starred and pre-highlighted. The step wording lives in `personas.yaml`.

   ![Persona picker — persona list on the left, the highlighted recipe with cost badges on the right](images/tui/persona.svg)
3. **Model.** Only for LLM personas, and skipped when you passed `--model`. It asks for
   the model name your local server lists (it says where to find it in LM Studio), and
   rejects blank input inline.

   ![Model screen — an input for the LLM model id served by your OpenAI-compatible endpoint](images/tui/model.svg)
4. **Scope.** The folder picker, as a navigable tree of your mailbox: top-level folders,
   their subfolders, plus "messages directly in …" rows. `space` includes or excludes,
   checking a folder covers all its subfolders and greys the children out, and a rule
   counter of ticked folders confirms what you have built. `c` continues, and you
   cannot proceed with nothing selected.

   ![Scope screen — the mailbox folder tree with a checked folder covering its greyed-out subfolders](images/tui/scope.svg)
5. **Review.** Everything on one screen before anything runs: persona, model, scope
   rules, rubric, limit, and the exact planned steps, with skipped optional steps
   marked. `enter` starts, `esc` goes back to change anything.

   ![Review screen — the collected configuration on the left, the exact planned steps on the right](images/tui/review.svg)
6. **Run.** The recipe as a live ladder (`○` pending, `▶` running, `✓` done) next to a
   streaming log that reports each step's result as a sentence, with an overall
   progress bar. Long steps don't freeze the terminal, because the pipeline runs in a
   worker while the UI stays live. When it finishes, the log says how to search the
   mail and how to connect agents. If it stops, rerunning the wizard carries on.

   ![Run screen — the recipe as a completed step ladder next to a streaming log, with a full progress bar](images/tui/run.svg)

`esc` steps back, `q` quits, and the footer always lists the active keys.

Two human checkpoints interrupt the run as modal dialogs, and keep you in control:

- **The calibrate gate.** Before any full LLM run you see a *sample* of what the rubric
  would flag, in two lists: *marked junk, but might be a real record* (suspected
  over-drops) and *kept, but might be junk* (suspected under-drops). Repeated
  notifications collapse into one row with a count. If it looks wrong,
  **re-tune** with `r` to pick another rubric and re-calibrate, then re-check. Only
  **proceed** with `p` once you trust it. This gate exists because a mis-tuned rubric
  once cost about 8 hours of LLM time on the wrong call.
- **Confirm-before-spend.** Right before the expensive summary pass, mailrag asks. So
  does `prune`, which shows a sample of what it would put on the skip list (the
  blacklist) and says what answering no means for the recipe you chose.

> The screenshots above are auto-generated from the real app, driven headlessly against a
> **synthetic** demo mailbox with no real mail, no Qdrant and no LLM, so they stay in sync
> with the code. Regenerate them with `python scripts/gen_tui_screenshots.py`.

Prefer the old line-by-line prompt flow? It is kept as `./mailrag wizard --classic`. For
non-interactive or scripted runs there is a headless equivalent,
`./mailrag run --profile … --persona <name> [--model <m>]`, which uses the same recipes and
the same handlers with no prompts.

## Quick start

```bash
# 1. create/scope a profile, measure it
./mailrag scope   --profile mybox.profile.json     # pick folders (interactive)
./mailrag measure --profile mybox.profile.json     # suggest chunk size

# 2. (optional, free) see where the noise is and get a recommendation
./mailrag scan    --profile mybox.profile.json

# 3. let the wizard walk the rest — or run a persona headlessly
./mailrag wizard  --profile mybox.profile.json
./mailrag run     --profile mybox.profile.json --persona llm-all --model <llm>
#    add --limit N to wizard/run for a fast end-to-end test on a small sample
#    (caps the scan/summarize/index steps) instead of a full multi-hour rebuild

# 4. ask questions
./mailrag ask "what did Alice say about the budget?" --profile mybox.profile.json
```

## Power use & custom personas

Every step is also a standalone verb (`./mailrag tag`, `./mailrag calibrate`, and so on),
so you can run the pipeline by hand. Personas live in
[`personas.yaml`](../personas.yaml). Add an entry there, or in a gitignored
`personas.local.yaml`, to define your own recipe, and both the wizard and `run` will read
it. See [`VERBS.md`](VERBS.md) for the full ladder and the cost of each verb.

## Attachments

Attachment text is searchable once you run `index`: it extracts the text of each kept
email's attachments (tesseract for scans) and indexes it beside the body, so a hit names
its file. The files themselves live in a separate store, one per collection, which is what
agents open them from. Every recipe ends with an `attach` step that fills the store, so
after the wizard or `mailrag run` there is nothing to do. By hand, the three commands are:

```bash
# ingest a profile's attachment bytes into the content-addressed store
./mailrag attachments build --profile <profile.json> [--limit N]

# list a thread's / message's attachments (prints full sha256s)
./mailrag attachments list --thread-id <id>        # or --message-id <id>
# list and get take --collection <name> to pick which collection's store to read

# fetch one attachment by sha256: extract + print its text, or write raw bytes
./mailrag attachments get <sha256> --text [--extractor llm|tesseract] [--force]
./mailrag attachments get <sha256> --out <path>    # raw bytes (always available)
```

Text extraction runs lazily on `get --text` and is cached. `--extractor` overrides the
`RAG_ATTACH_EXTRACTOR` env var for that call, and `--force` re-extracts even when a result
is already cached. `build` ingests the bytes, then OCRs the small images (under 100 kB by
default) once each to flag decoration such as signature logos. Its `--extractor` picks
the engine for that pass and defaults to tesseract. Documents are not read until `get`.

The default OCR backend for `get` is a **local vision LLM**, whatever `RAG_LLM_MODEL`
names at `RAG_LLM_API_BASE` (LM Studio by default), which keeps the work on-device. It
falls back automatically to local **tesseract** when the LLM is unavailable. Cloud OCR is opt-in and not yet implemented.

For setup details (optional Python deps, system packages, config vars), see
[`SETUP.md § 9`](SETUP.md#9-attachment-extraction).
