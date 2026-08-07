# deploy/mac-bridge

Образ `protocol-mis-bridge` + контракт extract (эпоха E1 на Mac без Docker тоже ок).

```bash
docker build -f deploy/mac-bridge/Dockerfile -t protocol-mis-bridge .
PYTHONPATH=. python3 -m services.mis_bridge.extract_day --day 2026-08-06 --dry-run
```

Контракт: [extract-contract.md](extract-contract.md).
