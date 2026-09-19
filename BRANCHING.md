# Branching strategy

This repository is designed to be adapted for specific PhD applications.
The `main` branch contains the generic scaffold; topic branches extend it
for a particular application area and opening.

---

## Branch structure

```
main                          ← generic scaffold, M0–M4 baseline
│
├── topic/spatial-transcriptomics   ← M3 focus, E6 = Visium dataset
├── topic/multi-omics               ← M4 focus, E6 = TCGA / CITE-seq
├── topic/sparse-genomics           ← M1/M2 focus, E6 = scRNA-seq
└── topic/<lab-specific>            ← customise per application
```

---

## What `main` contains

- Full engineering scaffold (CI, linting, docs, Makefile)
- M0 baseline model + simulator
- All model stubs (M1–M4) as clearly labelled TODOs
- All experiment stubs (E1–E6) with configs
- `docs/` skeleton with derivation placeholders
- Section 0 of `BRIEF.md` left as a placeholder — fill it per branch

---

## How to create a topic branch

```bash
git checkout main
git checkout -b topic/<name>
```

Then, on the topic branch:

1. **Fill in `BRIEF.md` Section 0** — your thesis connection, one sentence.
2. **Rewrite `README.md`** — open with the research question and your
   specific angle (spatial, multi-omics, sparse, etc.).
3. **Implement the relevant model phases** — e.g., for spatial:
   - Phase 1 (M0 + CAVI) — same as main
   - Phase 2 (MCMC) — same as main
   - Phase 4 (M3 + SVI) — spatial GP prior, the core contribution
   - Phase 6 (E6) — use a Visium dataset
4. **Choose E6 dataset** — one public dataset matching the application area.
5. **Update `mkdocs.yml` site_name** if desired.

---

## Merging back to main

Do **not** merge topic-specific work back to `main`. The main branch stays
generic. Use cherry-pick for any generic improvements (bug fixes, new
utilities) that would benefit all branches.

---

## Example topic branches and their differentiators

| Branch | Core model | E6 dataset | PhD application context |
|--------|-----------|------------|-------------------------|
| `topic/spatial-transcriptomics` | M3 spatial GP | 10x Visium | Spatial genomics labs |
| `topic/multi-omics` | M4 multi-view | TCGA multi-omics | Computational biology |
| `topic/sparse-genomics` | M1/M2 sparse | scRNA-seq | Single-cell methods |
| `topic/neuroimaging` | M3 spatial CAR | HCP fMRI parcels | Neuroimaging methods |

---

## Keeping branches in sync with main

When `main` receives scaffold improvements:

```bash
git checkout topic/<name>
git rebase main   # or merge, your preference
```

Resolve conflicts in model-specific files manually.

---

## What to commit on a topic branch

- `BRIEF.md` Section 0 filled in
- `README.md` with real findings (Phase 6)
- Implemented model phases (up to the relevant depth)
- E6 data loading + analysis (keep data files out of git — use `.gitignore`)
- `docs/results.md` and `docs/limitations.md` with actual content
