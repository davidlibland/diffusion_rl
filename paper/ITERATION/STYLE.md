# Style target (from a measured corpus of Li-Bland's math papers)

Source: arXiv 0811.4470, 1204.2796, 1212.2097, 1401.7302, 1506.08870, 2411.17988.

## Measured features to imitate
- **Lineage-first openings.** Name who built the object and when, then locate
  the new work as a widening of that lineage. "Why care" arrives late.
- **Sentence length 19–27 words**, built as [short declarative main clause] +
  [appositive] + [relative clause]. Not staccato, not 45-word pileups.
- **"we" at ~7–12 per 1000 words; never "I"** (zero authorial "I" in ~134k
  words, including a solo thesis).
- **Hedging ~0.1 per 1000 words** — effectively zero. "believe" 0,
  "arguably" 0, "hope" 0 across six papers.
- **Results stated in full in the introduction**, as complete mathematical
  sentences or labelled Theorem A/B — not summarised.
- **Dedicated Notation block**, with explicit flagging of deviations from the
  standard reference.
- **Prior work inline, generous then decisive**: concede the overlap, then
  draw the distinction in one sentence. No Related Work section natively.
- **Remarks do interpretive work** ("One should interpret Axiom (AV-3) as
  saying that ..."). Footnotes carry credit, asides, jokes.
- **Computations pushed to appendices** with a one-line forward pointer.

## Conflicts with ML-conference convention, and the resolution taken
1. *Contributions bullets.* Keep them (reviewers read only these), but write
   each as a complete mathematical sentence, not a marketing fragment.
2. *Headline numbers in the abstract.* Foreign to him — none of the six papers
   contains a single percentage or benchmark number. Do it anyway, once, in
   one sentence. This is a deliberate import.
3. *Hedging floor.* A theorem earns certainty; an ablation on 30 seeds does
   not. Import hedges ONLY onto empirical claims, phrased as scope conditions
   ("on the tasks we study"), never as doubt ("we believe"). Theorem
   statements stay hedge-free.
4. *Related Work / Limitations.* He has never written either. Both are
   near-mandatory. Limitations is the natural home for the hedges stripped out
   of the body — which resolves conflict 3.
5. *Signposting.* Adopt the roadmap paragraph and per-section pointers;
   migrate Remarks/Examples to the appendix rather than deleting them.

Compatible without change: notation discipline, appendix-for-proofs, and the
inline generous-then-decisive treatment of competitors (which reads as
unusually fair-minded and is worth keeping over a flat citation dump).
