## scripts/ops

Canonical entrypoints for multi-machine git/deploy operations.

During compatibility window, root-level scripts stay valid.
New automation and docs should prefer `scripts/ops/*`.

`render_deploy.sh` is the only entrypoint that talks to the Render API itself
(service settings, build logs, manual deploy and restart). It needs `RENDER_API_KEY`
in `.env`; everything else here works with git and the public prod URL only.
