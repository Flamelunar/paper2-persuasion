# TrajWeaver-v1 mechanism ablation (fixed 100)

Manifest: `/home1/liujianjian/2-paper-Coling-v2/train/TrajWeaver-v1/ablations/manifests/eval100_seed20260910.json`  |  n=100  |  selection_sha256=`849bd4bd1111ca49e931b722862fd838c981555f5bf17e1aab61341a4e3af78d`

This is a diagnostic subset, not a replacement for the 525-row main result. Existing public-zero-shot and legacy-v1 rows are filtered from their completed canonical runs; they are not recomputed.

| Variant | Success (%) | Acceptance (%) | ACR (%) | UNF (%) | Groundedness (%) | GDR (%) ↓ | Mean tokens/turn | I understand (%) | Trigger INVOKE (%) | Memory active (%) | Mean memory tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| public-zero-shot | 41.00 | 71.20 | 26.00 | 82.25 | 58.25 | 10.00 | 58.86 | 14.92 | 0.00 | 0.00 | 0.00 |
| legacy-v1 | 37.00 | 69.00 | 28.00 | 75.75 | 60.50 | 5.00 | 47.73 | 54.95 | 73.96 | 100.00 | 16.00 |
| r0-no-memory | 35.00 | 71.40 | 27.00 | 79.25 | 56.50 | 9.00 | 63.91 | 15.00 | 0.00 | 0.00 | 0.00 |
| g8-static-goal | 29.00 | 62.20 | 25.00 | 71.75 | 62.25 | 17.00 | 43.94 | 5.91 | 74.29 | 100.00 | 8.00 |
| g8bd8-always | 36.00 | 69.40 | 18.00 | 76.75 | 59.25 | 7.00 | 48.01 | 52.97 | 100.00 | 100.00 | 16.00 |
| g8bd8-trigger-strict | 29.00 | 66.40 | 19.00 | 74.25 | 57.00 | 10.00 | 57.22 | 44.27 | 73.96 | 73.96 | 11.83 |

Delta columns below are scenario-paired bootstrap estimates against `r0-no-memory`; they are for mechanism screening only.

| Variant | ΔSuccess (pp) | ΔAcceptance (pp) | ΔACR (pp) | ΔUNF (pp) | ΔGroundedness (pp) | ΔGDR (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| public-zero-shot | +0.06 [-0.06, +0.18] | -0.20 [-4.60, +4.20] | -1.00 [-11.00, +9.00] | +3.00 [-0.50, +6.50] | +1.75 [-2.75, +6.00] | +1.00 [-7.00, +9.00] |
| legacy-v1 | +0.02 [-0.09, +0.13] | -2.40 [-6.80, +2.00] | +1.00 [-9.00, +12.00] | -3.50 [-7.25, +0.25] | +4.00 [-1.00, +9.00] | -4.00 [-11.00, +3.00] |
| r0-no-memory | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| g8-static-goal | -0.06 [-0.17, +0.04] | -9.20 [-15.00, -3.80] | -2.00 [-12.00, +8.00] | -7.50 [-11.25, -3.75] | +5.75 [+0.25, +11.00] | +8.00 [+0.00, +16.00] |
| g8bd8-always | +0.01 [-0.09, +0.11] | -2.00 [-7.00, +2.80] | -9.00 [-18.00, +0.00] | -2.50 [-6.00, +1.25] | +2.75 [-2.00, +7.75] | -2.00 [-9.00, +5.00] |
| g8bd8-trigger-strict | -0.06 [-0.15, +0.03] | -5.00 [-9.80, -0.40] | -8.00 [-18.00, +2.00] | -5.00 [-8.50, -1.75] | +0.50 [-4.00, +5.00] | +1.00 [-7.00, +9.00] |
