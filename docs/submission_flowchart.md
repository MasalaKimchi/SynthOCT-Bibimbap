# SynthOCT Hypothesis and Submission Flow

This project stays aligned with the official `SynthOCTChallenge/SynthOCT_Baseline` contract:

- `Part1_Generator.py` maps to `synthoct baseline final` and emits digital phantoms.
- `Part2_Scanner.exe` maps to `synthoct scan --mode real` on Windows or the hosted scanner API.
- `Part3_Processor.py` maps to `synthoct evaluate --maps --metrics` and the internal validation map stack.
- The deliverable remains a four-column scatterer table: `X | Y | Z | Energy`.

```mermaid
flowchart TD
    A["Official SynthOCT Inputs\nZenodo real OCT B-scans"] --> B["Hypothesis Seed\ninverse physics, not GAN/diffusion image synthesis"]
    B --> C["Feature Extraction\nintensity profile + OAC + speckle contrast + refined speckle"]
    C --> D["Phantom Parameterization\ninhomogeneous scatterer point process"]
    D --> E["Part1-Compatible Generator\nsrc/synthoct/baselines.py"]
    E --> F["Official Format Gate\nN x 4 text: X, Y, Z, Energy"]
    F --> G{"Bounds and Schema OK?"}
    G -- "no" --> E
    G -- "yes" --> H["Scanner Loop"]
    H --> J["Hosted API or Windows Part2_Scanner.exe\nofficial physics rendering"]
    J --> K
    K --> L["Methodic Verification\nMS-SSIM up, LPIPS down, physics maps as guardrails"]
    L --> M{"Promote Hypothesis?"}
    M -- "no" --> C
    M -- "yes" --> N["Current Final\nH56_h41_anti_anatomy"]
    N --> O["prepare-submission\nmanifest + phantom zip + code zip + validation CSV"]
    O --> P["api-evaluate-submission\nuses SYNTHOCT_API_KEY or local key file"]
    P --> Q["preliminary_upload_plan.csv\nranked synthetic/reference PNG pairs"]
    Q --> R["Chrome Portal Submission\nupload rendered synthetic PNG and matching real PNG"]
    O --> S["Final Submission\ncode/model package or repository link"]
```

## Promotion Rules

1. A hypothesis must output scanner-compatible scatterers, not a direct synthetic image.
2. Every phantom must pass local format validation before API rendering.
3. Ranking must use hosted API or Windows `Part2_Scanner.exe` renders.
4. OAC/SC/RSC are guardrails even when the leaderboard emphasizes MS-SSIM and LPIPS.
5. Runtime must stay below 600 seconds per B-scan on the challenge hardware target.

## Command Pipeline

```bash
synthoct prepare-submission \
  --zip 18095266.zip \
  --out outputs/submission_ready_h56_full \
  --scatterers-count 300000
```

```bash
synthoct api-evaluate-submission \
  --zip 18095266.zip \
  --submission-dir outputs/submission_ready_h56_full \
  --out outputs/api_preliminary_h56 \
  --api-key-file ~/.config/synthoct/api_key
```

```bash
synthoct prepare-upload-plan \
  --api-results outputs/api_preliminary_h56/api_metrics.csv \
  --out outputs/api_preliminary_h56/preliminary_upload_plan.csv
```
