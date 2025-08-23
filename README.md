---
title: HF-Inferoxy AI Hub
emoji: 🚀
colorFrom: purple
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
hf_oauth: true
---

## 🚀 HF‑Inferoxy AI Hub

A focused, multi‑modal AI workspace. Chat, create images, transform images, generate short videos, and synthesize speech — all routed through HF‑Inferoxy for secure, quota‑aware token management and provider failover.

### Highlights
- Chat, Image, Image‑to‑Image, Video, and TTS in one app
- Works with any HF model exposed by your proxy
- Multi‑provider routing; default provider is `auto`
- Streaming chat and curated examples

### Quick Start (Hugging Face Space)
Add Space secrets:
- `PROXY_URL`: HF‑Inferoxy server URL (e.g., `https://proxy.example.com`)
- `PROXY_KEY`: API key for your proxy
- `ALLOWED_ORGS`: Comma/space‑separated org slugs allowed to use the Space

The app reads these at runtime — no extra setup required.

### How It Works
1. The app requests a valid token from HF‑Inferoxy for each call.
2. Requests are sent to the selected provider (or `auto`).
3. Status is reported back for rotation and telemetry.

### Using the App
- Chat: message → choose model/provider (`auto` by default) → tune temperature/top‑p/max tokens.
- Image: prompt → optional width/height (÷8), steps, guidance, seed, negative prompt.
- Image‑to‑Image: upload base image → describe the change → generate.
- Video: brief motion prompt → optional steps/guidance/seed.
- TTS: text → pick TTS model → adjust voice/style if supported.

### Configuration
- Model id only (e.g., `openai/gpt-oss-20b`, `stabilityai/stable-diffusion-xl-base-1.0`).
- Provider from dropdown. Default is `auto`.

### Providers
Compatible with providers configured in HF‑Inferoxy, including `auto` (default), `hf-inference`, `cerebras`, `cohere`, `groq`, `together`, `fal-ai`, `replicate`, `nebius`, `nscale`, and others.

### Security
- HF OAuth validates account/org membership (no inference scope).
- Inference uses proxy‑managed tokens. Secrets are Space secrets.
- RBAC, rotation, and quarantine handled by HF‑Inferoxy.

### Troubleshooting
- 401/403: verify secrets and org access.
- 402/quota: handled by proxy; retry later or switch provider.
- Image size: width/height must be divisible by 8.
- Slow/failures: try smaller models, fewer steps, or another provider.

### License
This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the [LICENSE](LICENSE) file for details.

### Links
- Live Space: [huggingface.co/spaces/nazdridoy/inferoxy-hub](https://huggingface.co/spaces/nazdridoy/inferoxy-hub)
- HF‑Inferoxy docs: [nazdridoy.github.io/hf-inferoxy](https://nazdridoy.github.io/hf-inferoxy/)
- Gradio docs: [gradio.app/docs](https://gradio.app/docs/)

— Built with HF‑Inferoxy for intelligent token management.


