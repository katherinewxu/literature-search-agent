# Midpoint demo: recording script (about 4 minutes)

Open `demo/v2/index.html` in a browser. Every page is self-contained and works offline.

1. **Problem (20 s), on the index page.** Systematic reviews need every eligible study. The agent gets the
   review question and criteria, plus 40 paid tool calls: keyword search, references or citers of a paper,
   and reading an abstract. It decides what to include. We score recall against the studies the review
   actually included.
2. **Episode 1, closed pool, search and citations (90 s).** Click the first card. Press Play, or use the
   arrow keys, through about 10 steps. Point out:
   - each new query and the papers it returns (green dots are gold studies);
   - include decisions;
   - the budget meter;
   - the recall curve climbing.

   Jump to the end (drag the slider right). True recall is 0.89. Then the readout: the agent's own
   estimate, made without gold labels, is 0.44, with a 95% lower bound of 0.31; the yield curve says 0.84.
   The point: estimates rank searches but aren't accurate per episode, and an LLM checker shares the
   agent's mistakes.
3. **Episode 2, citations only (40 s).** Same review, no search tool. The agent chases references and
   citers, stalls near 0.3, and stops with 15 calls unused.
4. **Episode 3, live OpenAlex (50 s).** Same agent, all of OpenAlex. Queries are similar, but most results
   are off-topic: recall is 0.38 and it stops with 9 calls left. Show the table on the index page: the
   search engine alone finds 1%, so the LLM's value grows as the universe grows.
5. **Results and next steps (30 s).** Show Figures 1 and 3 from the report (`docs/figures/`). GRPO v2, a
   redesigned reward, missed its pre-stated targets: it trades recall for precision, and the trained model
   still never chases citations. Next: larger budgets, a better web agent, and a recall bound tight enough to
   decide when to stop.

Terminal segment (recommended, about 20 s, instant, no API key). The course expects a working prototype,
so show the code running. Tiny fixture data with the scripted agent, run from the repo root:

    PYTHONPATH=src python3.12 -m litsearch.cli run --data artifacts/fixture --review fixture_depression \
      --agent snowball --mode hybrid --budget 20 --output runs/demo.json
    PYTHONPATH=src python3.12 -m litsearch.cli demo build --trace runs/demo.json --output runs/demo.html
    open runs/demo.html
