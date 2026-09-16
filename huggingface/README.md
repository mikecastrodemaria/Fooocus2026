---
title: Fooocus2026
emoji: 🎨
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: gpl-3.0
short_description: Fooocus2026 (mikecastrodemaria fork of Fooocus), SDXL image generation
---

# Fooocus2026 on Hugging Face Spaces

This Space runs the Docker image built by the GitHub CI of
[mikecastrodemaria/Fooocus2026](https://github.com/mikecastrodemaria/Fooocus2026)
(`ghcr.io/mikecastrodemaria/fooocus2026`). Nothing is built here: the Space only pulls
the image and starts it on port 7860.

- **Hardware**: a GPU is required (SDXL). Create the Space on the free `CPU basic`, then
  Settings > Hardware > `Nvidia T4 small` (minimum) or `A10G small` (comfortable). Paid
  hardware needs a payment method (Settings > Billing). `ZeroGPU` does not apply: it is
  for Gradio-SDK Spaces with `@spaces.GPU` functions, not for a Docker Space.
- **CLI**: `hf repos create <user>/fooocus2026 --type space --sdk docker --flavor t4-small --sleep-time 900`
  then `hf upload <user>/fooocus2026 huggingface/ . --type space` from a checkout of the
  repository creates and fills the Space without the web form.
- **Sleep time**: set it (Settings > Sleep time) so the GPU stops being billed after a
  period of inactivity. The first request after a wake-up takes a few minutes.
- **Persistent storage** (Settings > Storage, optional): then set `DATADIR=/data`,
  `config_path=/data/config.txt` and `config_example_path=/data/config_modification_tutorial.txt`
  in Settings > Variables so the models (about 7 GB downloaded at the first start), the
  outputs and the config survive restarts. Without it they are downloaded again at each
  cold start.
- **Update**: the image tag `edge` follows the `main` branch. Settings > Factory rebuild
  pulls the latest image. Replace `edge` by a version (`2026.6.0`) in the Dockerfile to
  pin the Space to a release.

The Ollama features (Layout/Omost, Describe, Improve) and the Extra plugins need a
machine of your own; here they simply say so when clicked.
