# Public data assessment

Verified on 2026-09-16. Downloaded sources are pinned to repository commits and
recorded in `data/manifest.json`.

## Selected datasets

### CToMPersu mysplit

- Location: project-shared `data/CToMPersu/mysplit/`; not duplicated here.
- Scale: 5,341 train, 525 dev, and 525 test source rows; test contains 391
  unique public scenarios.
- Role: primary domain-matched response training, rollout, and final evaluation.
- Strength: exact task and existing public/private simulator boundary.
- Limitation: no gold turn-level belief/desire trajectory or concern-resolution
  status.

### PersuasionForGood

- Source: https://github.com/ohyj1002/persuasionforgood
- License: Apache-2.0.
- Downloaded: full 1,017-dialogue data, 300 annotated dialogues, annotation
  schemes, upstream README, and license.
- Observed files: `full_dialog.csv` has 20,932 sentence units;
  `full_info.csv` has 2,034 participant-role rows.
- Role: human persuasion language, persuasion strategies, and donation outcome.
- Restriction for v4: demographic, religion, ideology, income, personality, and
  other participant-profile fields must not be fed to the response generator.
  Use dialogue/strategy/outcome fields unless an explicitly approved fairness
  analysis requires protected attributes.
- Limitation: belief/desire state is inferred rather than recorded each turn.

### CaSiNo

- Source: https://github.com/kushalchawla/CaSiNo
- License: CC-BY-4.0; attribution is required in redistributed artifacts.
- Downloaded: 1,030 dialogues and the official 900/30/100 split.
- Role: auxiliary desire/preference supervision. Each participant has explicit
  high/medium/low issue priorities and free-text reasons, plus strategy labels.
- Best use: train/evaluate desire extraction and evidence grounding, then test
  transfer to persuasion.
- Limitation: two-party resource negotiation is not the same task as persuading
  one person toward a fixed public goal. It should not be the main response SFT
  source.

### PersuasionTrace human-target runs

- Source: https://github.com/jlcmoore/persuasiontrace
- License: MIT.
- Downloaded: 15 public human-target result files, upstream analysis README,
  main README, and license.
- Observed scale: 450 rounds, 432 with serial belief measurements, and 1,317
  intermediate belief values.
- Role: belief-trajectory evaluation, calibration, change-point detection, and
  Trigger stress tests.
- Best use: hold out by human target and proposition; evaluate whether inferred
  state changes align with serial belief reports.
- Limitation: compact research experiment with heterogeneous conditions; it is
  unsuitable as a drop-in CToMPersu response training set.

## Not downloaded

### ToMELP

The public repository at https://github.com/qian753/ToMELP currently contains
only a README and Apache-2.0 license. No benchmark examples, split files, schema,
or evaluation code are publicly available, so it cannot yet support training or
evaluation.

### DailyPersuasion

No authoritative public repository or dataset release matching the document's
claimed scale was verified. It remains a candidate name, not an available
training source.

## Leakage and split policy

- Split by dialogue participant and scenario/proposition where identifiers are
  available, never by individual utterance.
- Keep PersuasionTrace human targets disjoint across train/dev/test.
- Do not use CToMPersu private `preventive`, `generative`, or `persona` fields in
  the tracker, planner, Trigger, Weaver, response generator, or judge.
- Preserve upstream raw files. Any normalization belongs under `data/processed/`
  with a manifest and source hashes.
- Do not merge CaSiNo or PersuasionForGood labels into CToMPersu categories
  without an explicit mapping table and an `unknown` option.
