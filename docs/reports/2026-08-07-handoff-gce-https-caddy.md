# Handoff: GCE HTTPS (Caddy) — 2026-08-07

## Repo
- worktree: `/private/tmp/protocol-task-gce-https-caddy-pc1`
- branch: `cursor/gce-https-caddy-pc1`
- base: `origin/main` @ `1520d61e`
- HEAD: see PR
- PR: open from this branch

## Done
- Caddy 2.11.4 installed on `protocol-app`, enabled, proxy to `:8000`
- Repo: `deploy/gcp-app/Caddyfile`, `setup_https_caddy.sh`, inventory + plan B5 notes
- `BUILD_VERSION` bumped (`gce-https-caddy`)

## Not done
- Public DNS still points `protocol.kravira.by` → Render (CNAME chain)
- Let's Encrypt not issued until **A** `protocol.kravira.by` → `34.118.21.47`
- Secret Manager still open

## Next command (after DNS flip)
```bash
cd /private/tmp/protocol-task-gce-https-caddy-pc1
dig +short protocol.kravira.by A   # must be 34.118.21.47
bash deploy/gcp-app/setup_https_caddy.sh --remote
curl -fsS https://protocol.kravira.by/health/live
curl -fsS https://protocol.kravira.by/api/version
```

## Do not touch in parallel
- `deploy/gcp-app/*` HTTPS/DNS, GCE firewall, Render custom domain for `protocol.kravira.by`
