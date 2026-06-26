# Scanner Rendering Backends

The repo has one scanner abstraction:

```python
from synthoct.scanners import render_phantom

render_phantom("phantom.txt", "Configuration.ini", "synthetic.png", backend="api")
render_phantom("phantom.txt", "Configuration.ini", "synthetic.png", backend="windows")
```

The default CLI backend is the hosted SynthOCT API, which is the correct macOS path because `Part2_Scanner.exe` is a Windows executable.

```bash
synthoct scan \
  --phantom outputs/phantom.txt \
  --out outputs/api_scan.png \
  --api-key-file ~/.config/synthoct/api_key
```

API keys are read from `SYNTHOCT_API_KEY`, `SYNTHOCT_CHALLENGE_API_KEY`, or an untracked file passed with `--api-key-file`.

For official local Windows validation:

1. Create the Python 3.12.12 environment from `environment.yml`.
2. Download `Part2_Scanner.exe` from the official baseline README.
3. Place `Part2_Scanner.exe` at the repository root or pass `--scanner-exe`.
4. Generate a four-column phantom:

```bash
synthoct baseline heuristic --input data/DATASET_PNG/.../scan.png --out outputs/phantom.txt
```

5. Run the real scanner:

```bash
synthoct scan --phantom outputs/phantom.txt --out outputs/scan.png --mode windows --scanner-exe Part2_Scanner.exe
```

6. Evaluate:

```bash
synthoct evaluate --ref data/DATASET_PNG/.../scan.png --pred outputs/scan.png --maps --metrics --out-csv outputs/metrics.csv
```

Keep the runtime below 600 seconds per B-scan.
