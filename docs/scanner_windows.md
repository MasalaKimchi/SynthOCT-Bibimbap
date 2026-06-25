# Windows Scanner Validation

The official `Part2_Scanner.exe` is a Windows executable. This repository is Mac-first, so local development uses `synthoct scan --mode stub` or `--mode precomputed`.

For challenge-like validation:

1. Create the Python 3.12.12 environment from `environment.yml`.
2. Download `Part2_Scanner.exe` from the official baseline README.
3. Place `Part2_Scanner.exe` at the repository root or pass `--scanner-exe`.
4. Generate a phantom:

```bash
synthoct baseline heuristic --input data/DATASET_PNG/.../scan.png --out outputs/phantom.txt
```

5. Run the real scanner:

```bash
synthoct scan --phantom outputs/phantom.txt --out outputs/scan.png --mode real --scanner-exe Part2_Scanner.exe
```

6. Evaluate:

```bash
synthoct evaluate --ref data/DATASET_PNG/.../scan.png --pred outputs/scan.png --maps --metrics --out-csv outputs/metrics.csv
```

Keep the runtime below 600 seconds per B-scan.
