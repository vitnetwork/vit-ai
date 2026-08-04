---
name: VIT ecosystem architecture
description: 4-service mesh topology, Render service IDs, GitHub repos, env groups
---

## Services

| Service | Render ID | URL | Repo | Env Group |
|---------|-----------|-----|------|-----------|
| vitnetwork (gateway) | srv-d8sipgjeo5us73eis7hg | https://vitnetwork-nls4.onrender.com | nemesistip-cloud/vit | evg-d97u2k67r5hc73dei09g ("vitnetwork") |
| vit-ai | srv-d97aur4s728c73d8jpk0 | https://vit-ai.onrender.com | vitnetwork/vit-ai | evg-d97s1028qa3s73fa9sv0 ("Vit") |
| vit-storage | srv-d97cr9faqgkc73ah2d20 | https://vit-storage-4trt.onrender.com | vitnetwork/vit-storage | evg-d97s1028qa3s73fa9sv0 |
| vit-chain | srv-d9etssn7f7vs73benfmg | https://vit-chain.onrender.com | vitnetwork/vit-chain | evm-d8sipgdckfvc73863sc0 |

## Environment structure
- Production environment: evm-d8sipgdckfvc73863sc0 (has Postgres dpg-d9dubfn41pts73du2kj0-a + Redis red-d8sitmm8bjmc738euoo0)
- Core service environment: evm-d97626uq1p3s738fnrqg (vit-ai + vit-storage)
- "vitnetwork" env group contains: DATABASE_URL, FOOTBALL_DATA_API_KEY, ODDS_API_KEY, PAYSTACK_SECRET_KEY, RESEND_API_KEY, TELEGRAM_BOT_TOKEN, DROPBOX_ACCESS_TOKEN, GDRIVE_SERVICE_ACCOUNT_JSON, ONEDRIVE creds, VIT_VALIDATOR_KEY, GENESIS_TREASURY_KEY

## Key facts
- All services: free plan, Oregon region, Docker runtime, auto-deploy on push to main
- vitnetwork health check: /ping; vit-ai health check: /health
- vitnetwork has a React+Vite frontend bundled into the Docker image (service URLs baked at build time as VITE_* ARGs)
- vitnetwork has 756 registered routes across ~80 route modules, all try/except wrapped
- vitnetwork Postgres DB is named "vit-postgres-v2", status: available

**Why:** Need this to orient quickly when working across services without re-querying Render.
