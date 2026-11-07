# Feasibility decision — pending access and trained-model runs

Q1 (repeatable benefit over target-only): **unresolved**.
Q2 (predictable preferable action): **unresolved**.
Recommendation: do not select a cost-aware controller from fixture timings.

## Actual execution

2026-11-07, Codex Cloud Linux x86_64; AMD EPYC 9V74, five exposed logical CPUs,
cgroup CPU quota four cores, 16 GiB memory limit, no swap or GPU. GCC 14.2.0,
CMake 3.31.6, Ninja 1.11.1, Python 3.12.14, NumPy 2.3.5. Release CPU build with
GGML_NATIVE=OFF, GGML_METAL=OFF, GGML_CUDA=OFF. No machine is called an M3 here.

The C++ runtime passed 183 differential fixture cases and the core policy/commit
checks. A 168-request fixture experiment (24 complete blocks) passed exact output
ID comparison and generated analysis artifacts; resume was tested. These use
tiny, controlled/random F32 transformer weights generated locally. They are
correctness evidence only, not useful language-model generation or feasibility.

Real model metadata/download probe:

```bash
python scripts/download_models.py --model qwen25-05
```

Observed failure: `Tunnel connection failed: 403 Forbidden` for
`https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF?blobs=true`.
The configured cloud allowlist lacks Hugging Face. No trained weights or pilot
measurements were produced. Domains need enabling in environment network
settings; do not send credentials in chat.

## Separate environments

* **Codex Cloud Linux CPU:** source build, synthetic state/differential tests,
  resumable request/analysis pipeline executed.
* **GitHub Codespaces:** user created `orange-space-giggle-gwpgvr4rvg7fgv9`.
  Direct `gh codespace ssh -c ...` fails before SSH at an `api.github.com`
  metadata request with Forbidden. Neither build nor model tests have run there
  through this chat. Git repository fetch/push does work.
* **User's Mac:** no authorized shell/terminal connection is currently attached.
  Chip/RAM/power state have not been inspected. Metal configuration and commands
  are supplied, but no Metal execution or M3 result exists.

## Bounded investigation prepared

`configs/models-v1.json` justifies two candidate pairs using available RAM,
then one upstream ngram mechanism if both independent pairs are unprofitable.
`configs/pilot-v1.json`, `bench/run.py` and `bench/replay.py` implement the
separate Pilot A and B procedures. Run independently on each platform:

```bash
bash run.sh cpu --run-dir results/local/linux-pilot
bash run.sh metal --run-dir results/local/mac-metal-pilot
```

Results are local, ignored, resumable, and transferable. A prior from one
platform is not evidence for a different machine. The script stops at a
scientific gate and records its reason; controller implementation/held-out
validation remains future work until the gate is answered.
