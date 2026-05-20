# scPRINT-2 — CLAUDE.md

> **Last updated:** 2026-04-10

## What is this project

scPRINT-2 is a next-generation single-cell foundation model (350M+ cells, 16 species) built on an additive study of 42 design choices. The paper is submitted (likely Nature Methods) and is currently under **active revision** — a rebuttal document (`rebuttal_nmth.docx` on Google Drive) is being finalized alongside local markdown working files.

## Current status (as of 2026-04-10)

- **Paper** : Edits in progress (`scprint2-paper.edited.md`). ~20 tracked changes applied (P1–P20+). **16 open comments** remain — mostly figure-dependent (panels out of order, missing captions, legend gaps). Local edits not yet uploaded to the GDoc.
- **Reply to reviewers** : Substantially restructured (`scprint2-reply-to-reviewers.edited.md`). **57 open comments** remain out of 100 total — many are "todo" placeholders (R#1 comments, tone softening from Laura, missing sentence fragments, unresolved reviewer questions).
- **Open comments** : 16 paper + 57 reply = **73 open items total**
- **DCA benchmark** : Integration into `denoising_V3.ipynb` on maestro **pending** (DCA itself done in isolated venv Python 3.9 + TF 2.12)
- **Figures** : Several figures need updating before corresponding comments can be closed

## Key files

| Fichier | Rôle |
|---------|------|
| `~/.openclaw/workspace/work/scprint2/scprint2-paper.origin.md` | Paper original — source de vérité (read-only) |
| `~/.openclaw/workspace/work/scprint2/scprint2-paper.edited.md` | Paper avec edits appliqués localement (1241 lignes) |
| `~/.openclaw/workspace/work/scprint2/scprint2-paper-open-comments.md` | 16 open comments paper (Laura, Jérémie, Gabriel, Ines) |
| `~/.openclaw/workspace/work/scprint2/scprint2-reply-to-reviewers.origin.md` | Reply original — source de vérité (read-only) |
| `~/.openclaw/workspace/work/scprint2/scprint2-reply-to-reviewers.edited.md` | Reply édité (733 lignes) |
| `~/.openclaw/workspace/work/scprint2/scprint2-reply-to-reviewers-open-comments.md` | 57 open comments reply |
| `~/.openclaw/workspace/work/scprint2/CHANGES-APPLIED-V2.md` | Log des edits P1–P20+ et R1–R5+ déjà appliqués |
| `~/.openclaw/workspace/work/scprint2/reply-improvements-plan.md` | Plan d'amélioration de la reply (99 lignes) |
| `~/.openclaw/workspace/work/scprint2/update_comments.sh` | Script pour re-syncer les commentaires depuis Drive |

## What's blocking

1. **Figures manquantes / à mettre à jour** — 16 commentaires paper pointent vers des problèmes de figures (panels hors ordre, légendes incomplètes, panel c absent de Fig 1). Bloque la fermeture de nombreux commentaires.
2. **57 open comments reply** — Beaucoup sont des "todo" (Jérémie #9–#20) sur les figures supp / GNN. Laura demande des reformulations de ton sur plusieurs réponses (#4, #6, #7).
3. **DCA → denoising_V3.ipynb** — L'intégration sur maestro est bloquée ; sans résultat, le benchmark denoising est incomplet.
4. **Upload GDoc** — Les edits locaux (P1–P20) ne sont pas encore dans `rebuttal_nmth.docx` sur Drive.
5. **Ines authorship** — Commentaire #19 de l'open comments paper : `@ine281101@gmail.com` — décision Laura en suspens sur la contribution au rebuttal.
6. **Citations manquantes** — Commentaire #15 paper : des citations `[]` vides dans le texte doivent être remplies.

## Next steps for Jérémie

1. **Générer/corriger les figures** pour débloquer les 16 commentaires figure-dépendants du paper (commencer par Fig 1 panel c et l'ordre des panels).
2. **Traiter les "todo" reply** (#9–#14, #16–#17, #20) — beaucoup sont des figures supp / GNN qui peuvent être fermées rapidement une fois les figures produites.
3. **Intégrer DCA dans `denoising_V3.ipynb`** sur maestro — venv Python 3.9 + TF 2.12 est prêt.
4. **Upload GDoc** — Copier les edits de `scprint2-paper.edited.md` dans `rebuttal_nmth.docx` sur Drive.
5. **Reformulations ton Laura** — Adresser les commentaires #4, #6, #7 reply (ton plus doux, contenu déjà bon) + #3 (phrase incomplète).

## Repo

- **Code** : `~/projects/scPRINT-2/` — code, analyses, notebooks
- **Branch active** : `main`
- **Remote rebuttal** : `ssh maestro`, `~/scPRINT` (remote `scprint2` → cantinilab/scPRINT-2)
- **Checkpoint** : utiliser `small-v2.ckpt` (⚠️ `18hebyht-final-small.ckpt` = stub corrompu — ne pas utiliser)
- **GitHub** : jkobject/scPRINT (v1) + cantinilab/scPRINT-2 (v2)

## People

| Personne | Rôle |
|----------|------|
| Laura Cantini | Directrice de thèse, co-auteure, beaucoup de commentaires de révision ouverts |
| Gabriel Peyré | Co-directeur, commentaires ponctuels |
| Ines Lalou | Co-auteure, active sur les commentaires reply |
| Geert Huizing | Collaborateur |
