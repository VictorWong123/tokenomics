# Contributing

Start with the [development guide](docs/development.md) for the source/language
map, native and full builds, test suites, sanitizers and formatting.
The [documentation index](docs/README.md) connects design, commands and evidence.

Keep changes focused on the owned C++ runtime or supporting tools. Preserve
upstream greedy verification and the clean pinned llama.cpp submodule. Python
handles offline calibration, orchestration and analysis; native C++ owns model
execution and online policy decisions.

Run the checks appropriate to the change. Full fixture and Python integration
coverage uses `bash scripts/bootstrap.sh cpu`; pure C++ policy tests are a
smaller, separately documented check. Trained measurement requires its exact
token-ID gate, frozen configuration and the owned Codespaces lifecycle.

Retain failures and uncertainty. Existing raw evidence, hash-bound configurations
and canonical receipts are immutable. New measurements need a new directory
after source, binary, model or setting changes. Local builds, models, environments
and experimental runs remain ignored; stage explicit paths and run
`.venv/bin/python scripts/check_staged.py` before committing.

Describe what changed, why and which checks actually ran. Update current user
documentation with behavior changes; keep historical session notes separate.
