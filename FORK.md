# crispz-sdxl: the crispz family on the Fooocus2026 SDXL engine

Status: lot 1 (plan and skeleton), branch `wip/crispz-sdxl`, version 0.1.0 unreleased.

## What it is

crispz-sdxl is a member of the crispz family (crispz-klein, crispz-flux3, crispz-qwen-edit,
crispz-krea2, crispz-studio) whose engine is SDXL as Fooocus2026 implements it: base and
refiner, styles, inpaint and outpaint, upscale and variation, Image Prompt (IP-Adapter),
FaceSwap, PyraCanny and CPDS, LoRAs, the Civitai browser. The SDXL checkpoints and LoRAs
become reachable with the family UI conventions and the CLI protocol v1, so
crispz-dataset-studio, the `czp` CLI and the other studios can drive them.

What makes an app "crispz" is the protocol: the hidden Gradio endpoints `cli_caps`,
`cli_gen`, `cli_edit`, `cli_faces`, one JSON string in, one JSON string out.

## What Fooocus2026 exposes without its UI (the map)

Fooocus2026 already speaks the protocol as the tool `fooocus2026` (custom-26, 2026-09).
Nothing had to be reverse engineered from `webui.py`:

| Need | Where | Notes |
|---|---|---|
| The generation task | `modules/async_worker.py`, `AsyncTask.__init__` (about 110 positional ctrls) and `worker()` | Stages: `process_prompt`, `apply_image_input` (uov / ip / inpaint / enhance tabs), `apply_vary`, `apply_upscale`, `apply_inpaint`, `apply_outpaint`, `apply_control_nets`, `process_task`, `process_enhance`, `save_and_log` |
| Building that task outside the UI | `modules/task_args.py`, `build(prompt, negative, performance, aspect, output_format, seed)` returns `(args, idx)` | `idx` records the position of every ctrl by name (`aspect`, `base_model`, `lora0_name`, `cn0_image`, `uov_method`, `inpaint_input_image`, `enh0_mask_prompt`...). An empty `AsyncTask(args=probe)` validates the layout |
| Queueing and waiting | `modules.async_worker.async_tasks.append(task)`, then `task.yields` until `['finish', ...]`, results in `task.results` (file paths) | One worker thread, one GPU, one queue, behind the user's renders |
| Spec to task mapping | `fooocus_protocol.py`: `validate_spec`, `build_task_args`, `execute`, `run` | gen (size, steps, LoRAs and `<lora:...>` tags, up to 4 refs as Image Prompt, `detail_faces` / `detail_hands` as Enhance tabs), upscale (factor 1 = Vary, 1.5 / 2 / fast), inpaint (white mask, local prompt) |
| Hidden endpoints of the running app | `webui.py` lines 3269 to 3282: `cli_caps`, `cli_gen` bound to `fooocus_protocol.remote_caps` / `remote_run` | Gradio 3.41, `POST /run/<name>`, out of the queue |
| CLI | `czp.bat` / `czp.sh`, `fooocus_protocol.main` | Routes to the running instance, else loads the backend in process |
| Model lists | `modules/config.py`: `model_filenames`, `lora_filenames`, `vae_filenames`, `embedding_filenames` after `update_files()` (architecture filter: SDXL and SD only) | Empty at import, `launch.py` scans; the CLI must call `update_files()` itself |
| Image Prompt slots | `modules.config.default_controlnet_image_count` (4) | `caps.supports.max_refs` |
| Pipeline entry points | `modules/default_pipeline.py`: `refresh_everything`, `refresh_base_model`, `refresh_refiner_model`, `refresh_loras`, `clip_encode`, `process_diffusion` | Used by the worker; the crispz layer does not call them directly |
| Flags and enums | `modules/flags.py`: `uov_list`, `ip_list`, `inpaint_engine_versions`, `Performance`, `Steps`, `sampler_list`, `scheduler_list`, `sdxl_aspect_ratios` | |
| Outputs and metadata | `modules/private_logger.py` `log()`: `path_outputs/YYYY-MM-DD/<timestamp>_<rand>.<ext>`, `log.html`, fooocus or a1111 scheme, AI provenance XMP (custom-23) | File names are not the family's |
| Faces | `modules/global_faceswap.py` (IP-Adapter face, custom-32), `modules/exact_faceswap.py` (inswapper through the crispz-studio plugin, custom-33), `extras/face_crop.py` | No detection or embedding API comparable to the family `cz_face.py` |
| Civitai | `modules/civitai_api.py`: `search_models`, `download_model_file`, `fetch_recommended_settings`, `fetch_lora_triggers` | Same scope as the family `cz_civitai.py` |
| Ollama | `modules/ollama_describe.py`, `modules/ollama_improve.py` | Same scope as the family `cz_ollama.py` |
| Boot and install | `launch.py` (torch 2.1.0 cu121, Python 3.10 / 3.11), `args_manager.py` (`--listen`, `--port`, `--preset`, `--output-path`), `boot_check_rtx5090.bat/.sh`, `Dockerfile` | |

Measured on the cloud container (Python 3.13, no torch, no gradio): `import modules.config`
takes about 1 s and writes `config.txt`; `modules.async_worker` cannot import (torch).

## Architecture decision

The Fooocus engine stays as it is under `modules/`, `ldm_patched/`, `extras/`. A crispz
layer sits beside `webui.py`, at the repo root like the other family apps:

| File | Role | Lot |
|---|---|---|
| `cz_core.py` | identity, `crispz` block of `config.txt`, output root, `[crispz]` log | 1 |
| `cz_imageio.py` | `out/YYYY-MM-DD/{date}_{tag}_seed{seed}_{w}x{h}.png`, `crispz` chunk, A1111 chunk, `.json` sidecar | 1 (fake engine), 2 (real outputs) |
| `cz_pipeline.py` | `generate(spec, op)` in the family call convention, over `fooocus_protocol.run()`; `FakeWorker` for CPU runs | 1, 2 |
| `cz_protocol.py` | `cli_caps`, `cli_gen`, `cli_edit`, `cli_faces` handlers, caps computed from what runs | 1, 2 |
| `cz_face.py` | insightface detection and embeddings (`detect_faces`, `detect_faces_full`) | 2 |
| `cz_ui.py` | family tabs: Text to Image / Edit or Image Prompt / Image to Image, refine, upscale / Inpaint, Outpaint / Faces | 3 |
| `app.py` | boots the crispz UI (and the Fooocus one if kept) and binds the endpoints | 3 |
| `install.*`, `run.*`, `update.*`, `boot_check.bat` | family scripts over `launch.py` | 4 |
| `tools/smoke_test.py`, `tests/` | CPU tests with the fake engine | 1, 5 |

Why reuse custom-26 instead of a new mapping: the ctrl order of `AsyncTask` is fragile and
is copied in exactly one place (`modules/task_args.py`); a second copy would drift. The fake
engine of lot 1 decodes that same snapshot by name, so the tests exercise the real mapping.

## Feature matrix

| Fooocus feature | crispz tab or protocol op | Lot |
|---|---|---|
| Text to image SDXL (base + refiner, performance presets, Fooocus V2 expansion, styles) | Text to Image tab; op `gen` | 1 (op), 3 (tab) |
| Image Prompt, IP-Adapter, 4 slots | `refs` of `gen`; Edit or Image Prompt tab | 1 (op), 3 (tab) |
| FaceSwap (IP-Adapter face, global face, exact swap) | Faces tab; op `faces` (detection and embeddings), face ref of `gen` | 2 (op), 3 (tab) |
| PyraCanny, CPDS | Edit or Image Prompt tab, advanced | 3; no protocol field in v1 |
| Vary subtle / strong | Image to Image tab; op `upscale` factor 1 | 1 (op), 3 (tab) |
| Upscale 1.5x / 2x / fast, upscaler models, face restoration (custom-10) | Image to Image, refine, upscale tab; op `upscale` | 1 (op), 3 (tab) |
| Inpaint | Inpaint, Outpaint tab; op `inpaint` | 1 (op), 3 (tab) |
| Outpaint | Inpaint, Outpaint tab | 3; no protocol field in v1 |
| Enhance (face, hand detail with SAM) | `detail_faces`, `detail_hands` of `gen` | 1 |
| Instruction edit | op `edit` | 2, mapping to decide (open question 6) |
| LoRAs (5 slots, prompt tags, trigger words) | `loras` of the spec, LoRA panel | 1 (op), 3 (tab) |
| Checkpoint and LoRA lists, architecture filter | `caps.models`, `caps.loras` | 1 |
| Metadata (fooocus / a1111 schemes), AI provenance (custom-23) | `crispz` chunk + sidecar, A1111 chunk kept | 2 |
| Civitai search, download, settings, triggers | Models panel of the crispz UI | 3 or 5 (open question 2) |
| Ollama describe / improve | prompt helpers of the tabs | 3 |
| Job Queue, X/Y/Z grid, batch input, Asset Browser, wildcards, tag autocomplete, presets, Omost | stay in the Fooocus UI | out of scope for 0.1.0 (open question 2) |
| RTX 5090 boot checks, Dockerfile | `install.*`, `run.*`, `boot_check.bat` | 4 |

## Lots

1. Plan and skeleton (this lot): `FORK.md`, `cz_core.py`, `cz_imageio.py`, `cz_pipeline.py`
   with the fake engine, `cz_protocol.py` with caps and `gen` (`upscale` and `inpaint` ride
   along since custom-26 maps them), `tests/test_cz_protocol.py`, `config-sample.txt`,
   `CHANGELOG.md`. `edit` and `faces` answer code 3.
2. Engine wrapper and protocol: real engine validated on the RTX 5090 (Mike), `edit` and
   `faces` ops (`cz_face.py`, insightface as in the family), real outputs renamed or written
   into the family layout with the `crispz` chunk, endpoints bound in the running app beside
   or instead of the custom-26 pair, `czp` answering as crispz-sdxl.
3. UI: `cz_ui.py` with the family tabs, `app.py`, Fooocus UI kept or replaced.
4. Install scripts and config: `install.bat/.sh`, `run.bat/.sh`, `update.bat/.sh`,
   `boot_check.bat`, `config-sample.txt` completed, Dockerfile check.
5. Tests, `tools/smoke_test.py`, README, CHANGELOG, Pinokio launcher repo.

Each lot stops for validation.

## Open questions for Mike

1. Fooocus UI: keep it beside the crispz tabs (proposal: yes, `crispz.keep_fooocus_ui`, so
   nothing of the fork is lost) or replace it in 0.1.0.
2. Out of scope for 0.1.0 in the crispz tabs: Job Queue, X/Y/Z, batch input, Asset Browser,
   wildcards, tag autocomplete, presets, Omost (proposal: out, they stay in the Fooocus UI).
   Civitai: in the crispz Models panel (lot 3) or later (lot 5)?
3. Refiner and styles: the protocol v1 has no field for them. Proposal: config defaults apply
   to the ops, the tabs expose them in an Advanced accordion; `styles` and `refiner` as
   tolerated extra spec fields later if crispz-dataset-studio needs them.
4. Port: 7865 (Fooocus, what `czp` and comics2crispz already use) or the family port of the
   other studios?
5. Default output folder: `out/YYYY-MM-DD/` (family) or `../outputs/` (`path_outputs`,
   Asset Browser and `log.html` depend on it). Proposal: Fooocus keeps `path_outputs`, the
   crispz layer renames or copies the final image into `out/` with the family name and chunk.
6. `edit` op on SDXL: Image Prompt + Vary strong (approximate), inpaint with a mask (exact
   but needs a mask), or `supports.edit` false in 0.1.0.
7. Identity on one port: `fooocus_protocol.probe_instance` accepts only `tool == fooocus2026`.
   If `cli_caps` answers `crispz-sdxl`, `czp` and comics2crispz must accept it too (proposal:
   `cli_caps` answers crispz-sdxl with `engine.tool = fooocus2026`, the probe accepts both).

## Measurements

| What | Value | Where |
|---|---|---|
| CPU tests, lot 1 | 32 tests (14 crispz + 18 custom-26) in 0.8 s, Python 3.13, no torch | `tests/test_cz_protocol.py`, `tests/test_protocol.py` |
| Fake render | 256x320 placeholder, under 50 ms | `cz_pipeline.FakeWorker.render` |
