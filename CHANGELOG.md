# Changelog — Fooocus2025 (mikecastrodemaria fork)

This fork is based on [lllyasviel/Fooocus](https://github.com/lllyasviel/Fooocus) **v2.5.5**.
Only fork-specific changes are listed here — upstream history is available via `git log`.

## [custom-11.1] — 2026-05-16 — Resilient sha256_from_cache (no more crash on missing LoRA/checkpoint)

### Fixed
- `modules/hash_cache.py::sha256_from_cache` used to call `sha256(filepath)`
  directly. If the file had been moved/deleted/reorganized after Fooocus
  indexed it (very common when the user shares `path_loras` with an A1111
  install that gets reshuffled), `open(filename, "rb")` raised
  `FileNotFoundError` and **the whole save_and_log path crashed AFTER the
  image was already saved**, leaving the user with a generated PNG but no
  metadata, no upscale, no continuation of the pipeline.
- The cache function now checks `os.path.isfile(filepath)` first; if absent
  or unreadable, it logs `[Cache] WARNING: file not found, skipping sha256`
  and caches an empty string. Downstream metadata gets a blank hash for that
  one file but everything else continues normally — upscale, enhance, save,
  log.html, all fine.

### Why
- Defensive metadata writing should never block a successful generation
  from completing. Image saving + post-processing must be more robust than
  the hash bookkeeping that decorates them.

## [custom-10.4] — 2026-05-16 — Surface "Upscale Denoising Strength" out of Developer Debug

### Changed
- The slider previously labelled **"Forced Overwrite of Denoising Strength of
  Upscale"** in the Developer Debug Mode section is now visible to every user,
  inside the custom-10 block (Enhance > Upscale or Variation, between the
  Upscaler Model dropdown and the Face Restoration dropdown). Renamed to a
  clearer **"🎚 Upscale Denoising Strength"** with a friendly info text.
- Same Python variable name (`overwrite_upscale_strength`) and same default
  (`-1` = auto / `0.382`) — no change to `ctrls` or `AsyncTask`. The widget
  just lives in a more discoverable place.

### Why
- The "denoising strength" value controls how aggressively the SDXL refinement
  pass redraws after the ESRGAN upscale. It's a major creative knob (low =
  preserve detail, high = creative redraw), yet it was buried under
  "developer debugging" and most users never found it.

## [custom-10.3] — 2026-05-16 — Substring match for residual_block_ + robust upscaler fallbacks

### Critical fix
- **`residual_block_` detection was wrong.** Original Fooocus keys look like
  `model.1.sub.N.residual_block_X.convY` — substring, not prefix. The
  `k.startswith('residual_block_')` check from custom-10 returned False for
  all keys of the bundled Fooocus default upscaler, sending it to the
  auto-detector which doesn't recognize that layer naming. Result: even the
  default upscaler was broken since custom-10. Fixed by switching to
  `'residual_block_' in k` (substring), mirroring the original `.replace()`
  call's behaviour.

### Added
- **Double fallback** in `_load_upscale_model`: legacy `residual_block_*` path
  first, then auto-detector, then legacy ESRGAN constructor as last resort.
  Recovers community ESRGAN variants whose keys confuse the auto-detector
  but still parse with a forced ESRGAN.
- **Safe-guard** around the bundled-default load: clear `RuntimeError` with
  actionable message ("file likely replaced — delete it, Fooocus will
  re-download the official one") instead of the cryptic `UnsupportedModel`.

## [custom-10.2] — 2026-05-16 — UX move + graceful upscaler fallback

### Changed
- **Moved the 4 custom-10 controls** (Upscaler Model, Face Restoration,
  Visibility, Apply Face Restoration order) **from the standalone Upscale
  or Variation tab into the Enhance > Upscale or Variation section**,
  inserted between "Order of Processing" and "Prompt". The controls are
  now wrapped in a `gr.Group` whose visibility is bound to
  `enhance_uov_method` — shown only when the user picks an Upscale mode
  there (`Upscale (1.5x)` / `Upscale (2x)` / `Upscale (Fast 2x)`).
- The wiring is unchanged: `apply_upscale()` in `modules/async_worker.py`
  is the **single shared upscale entry point** for both the standalone
  Input Image path AND the Enhance pipeline, so these controls govern
  both paths through one set of widgets. No duplicate UI required.

### Added
- **Graceful upscaler fallback** in `modules/upscaler.py::perform_upscale`.
  When a `.pth` / `.safetensors` doesn't match any architecture the bundled
  loader knows (e.g., a community ESRGAN variant with non-standard layer
  keys like `4x-ClearRealityV1`), we now log a clear `[Upscaler] WARNING`
  and fall back to the Fooocus bundled ESRGAN for that run, instead of
  crashing the whole Enhance/Upscale pipeline with `UnsupportedModel`.

### Removed
- The custom-10.1 visibility binding on `uov_method` (top tab) was rolled
  into the move; the standalone tab no longer hosts these controls.

## [custom-11] — 2026-05-16 — Security: dep cleanup + Gradio pin documented

### Removed
- **`pytorch_lightning==2.3.3`** dropped from `requirements_versions.txt`. The
  only reference in the codebase was a commented-out import in
  `ldm_patched/ldm/models/autoencoder.py` — pure dead dep. Closes Dependabot
  alerts #104 (Critical path traversal) and #40 (Direct).

### Changed
- **`tokenizers`** `0.19.1` → `0.21.0` (forced by the transformers bump — `transformers 4.50.0` requires `tokenizers>=0.21,<0.22`). Same upstream API for BertTokenizer, no breaking change for BLIP.
- **`transformers`** `4.42.4` → `4.50.0`. Fixes 4 deserialization CVEs
  (#97, #96, #95, #33, #32, #31). Only usage in the fork is BLIP loading
  `BertTokenizer` — API stable since transformers 4.x.
- **`torch`** `2.1.0` → `2.5.1` and **`torchvision`** `0.16.0` → `0.20.1`
  in `requirements_docker.txt`. Fixes the `torch.load(weights_only=True)`
  RCE (#52).
- **`gradio==3.41.2`** pin in `requirements_versions.txt` is now flagged with
  an explicit DO-NOT-BUMP comment. Every JS/DOM injection in this fork
  (Asset Browser icons, custom accordion headers, header reindex polling,
  custom-10 paths accordion) relies on the Gradio 3.x DOM layout. A bump to
  Gradio 4.x would break all custom-8 / custom-9 / custom-10 features and
  is not on the roadmap.

### Documentation
- New **🔒 Security note** section in `readme.md` (just above "Fork-specific
  config.txt keys") explaining the local-use threat model, why the remaining
  Gradio 3.41.x CVEs aren't exploitable in the intended deployment, and the
  rules to follow if exposing Fooocus on a LAN (no `--share`, no public bind,
  use a real reverse proxy with auth).

### Notes
- After this commit, the Dependabot dashboard drops from 127 to ~116 alerts.
  The remaining alerts are Gradio 3.x server CVEs that are knowingly accepted
  for compatibility — see the Security note for the threat model.
- After installing the new requirements, regenerate the venv or run
  `pip install -r requirements_versions.txt --upgrade` to apply the bumps.

## [custom-10] — 2026-05-16 — Multi-upscaler + face restoration (CodeFormer / GFPGAN)

### Added
- **Pluggable upscale models.** `modules/upscaler.py` refactored to use `ldm_patched.pfn.model_loading.load_state_dict()` instead of the hardcoded ESRGAN loader. ESRGAN, RealESRGAN (SRVGG), SwinIR, Swin2SR, HAT, DAT, SCUNet, OmniSR, SwiftSRGAN, SPSR are auto-detected from the state dict — drop a `.pth` / `.safetensors` in any configured folder and it shows up in the dropdown. LRU cache (`maxsize=4`) avoids re-loading 30+ MB per call when the user switches models in a session. `perform_upscale(img)` keeps its legacy signature (None → bundled Fooocus default) for back-compat.
- **Face restoration pipeline (CodeFormer / GFPGAN v1.4).** New `_apply_face_restoration()` inside `worker()`, blended with a visibility slider (0–1). Same auto-detecting loader handles the face restoration archs (CodeFormer / GFPGAN clean / RestoreFormer all recognized by `model_loading.load_state_dict`). Models auto-download on first use into `path_face_restore_models`.
- **UI controls in the Upscale or Variation tab** (4 new):
  - 🔍 **Upscaler Model** dropdown (populated from `list_upscale_models()`)
  - 👤 **Face Restoration** dropdown (Off / CodeFormer / GFPGAN v1.4)
  - **Visibility** slider (0.0–1.0, default 0.8)
  - **Apply Face Restoration** radio (Before / After upscale — After = A1111 default)
- **A1111-compatible paths.** Six new optional `config.txt` keys lets you point Fooocus at an existing Stable Diffusion WebUI install instead of duplicating models on disk:
  - `path_esrgan`, `path_realesrgan`, `path_swinir`, `path_dat`
  - `path_gfpgan`, `path_codeformer`
  - `path_face_restore_models` (default for auto-downloaded face models)
- **Accordion "🔍 Upscale & Face Restoration Paths"** in Advanced > Models. Edit the 6 paths directly from the WebUI and Save (writes to `config.txt`). Restart UI to refresh the dropdowns.

### Changed
- `modules/upscaler.py` — full rewrite (98 lines). Bundled Fooocus upscaler key naming `residual_block_*` is translated to `RDB*` on the fly so the same code path handles it.
- `modules/async_worker.py::AsyncTask` consumes 4 new args after `enhance_ctrls` (`upscaler_model_name`, `face_restore_model`, `face_restore_visibility`, `face_restore_order`). `apply_upscale()` now resolves the chosen model and wraps the call with optional pre/post face restoration.
- `modules/config.py` — `+116 lines`: 7 new path declarations (`_opt_path` helper writes empty default to `config_dict` so the serializer doesn't raise `KeyError`), `list_upscale_models()` / `list_face_restore_models()` helpers (dedupe across folders, `[folder] basename` prefix on collision), and two downloaders pulling CodeFormer (~376 MB) and GFPGAN v1.4 (~333 MB) from their official GitHub releases.

### Documentation
- `readme.md` — extended the Fork-specific `config.txt` keys table with the 7 new keys (`custom-10` row); example fragment shows pointing the three most common folders at an A1111 install.

### Notes
- The dropdown content is a snapshot taken at UI build time — Restart UI required after dropping new files in any of the configured folders.
- Empty path = silently ignored (no warning, no crash). Missing folder = silently ignored too.
- The Fooocus bundled upscaler remains the default; selecting "Fooocus Default (ESRGAN)" in the dropdown is identical to the pre-custom-10 behaviour.

## [custom-9] — 2026-05-16 — Architecture filtering + Asset Browser shortcut icons

### Added
- **Architecture-based model filtering** — checkpoints, LoRAs, and embeddings dropdowns now hide incompatible models automatically. Reads safetensors headers (~1 ms/file, no weight loading) to detect architecture:
  - **Checkpoints:** keeps SD 1.x and SDXL only; hides Flux, SD3, SVD, LLMs, and unrecognised formats.
  - **LoRAs:** blacklist approach — blocks Flux LoRAs, SD3 LoRAs, and full models misplaced in the LoRA folder; keeps everything else (kohya, diffusers, LyCORIS, LoHa, LoKr, Pony).
  - **Embeddings:** file-size guard (< 10 MB = always compatible) prevents false positives on small specialised embeddings (e.g. `badhands.safetensors`); larger files are checked for full-model signatures.
  - JSON cache (`model_arch_cache.json`) with mtime-based invalidation — delete the file to force a full rescan.
- **Asset Browser shortcut icons** — small clickable icons next to **Base Model**, **Refiner**, **LoRA**, and **Textual Inversion** headers in the Advanced tab. Each icon opens the corresponding Asset Browser tab (`#checkpoints`, `#loras`, `#embeddings`) in a new browser tab. Implemented as `javascript/ab_icons.js` injected via `<head>` (same pattern as all other Fooocus JS files).

### Changed
- **`modules/config.py`** — new functions: `_read_safetensors_keys()`, `_detect_checkpoint_arch()`, `_detect_lora_arch()`, `_detect_embedding_arch()`, `_filter_by_arch()`. `update_files()` now applies architecture filtering to checkpoints, LoRAs, and embeddings before populating dropdowns.
- **`modules/ui_gradio_extensions.py`** — injects `<meta name="ab-base-url">` tag and loads `ab_icons.js` via the standard `<head>` injection pipeline.
- **`webui.py`** — removed ~60 lines of inline JS icon injection code (replaced by `ab_icons.js`); removed the descriptive HTML block ("Outputs gallery + LoRAs / Checkpoints / Embeddings tabs…") from the Asset Browser accordion.
- **`.gitignore`** — comprehensive rewrite: organised by category (Python, model weights, models directory, outputs, runtime caches, user config, IDE/OS, environments, legacy, Claude artifacts). Now covers `model_arch_cache.json`, `hash_cache.txt`, `sorted_styles.json`, `civitai_cache/`, and all model weight formats.

### Fixed
- LoRA dropdown no longer shows full checkpoint models accidentally placed in the LoRA folder (e.g. `z-image`, `wan`).
- Pony-style LoRAs (LyCORIS/LoHa/LoKr) are no longer falsely filtered — the blacklist approach only rejects known-incompatible patterns.
- `badhands.safetensors` (small specialised embedding) is no longer falsely hidden thanks to the 10 MB size guard.

## [custom-8.6] — 2026-05-03 — Backfill PNG metadata during reindex (Copy All Params no longer empty)
### Fixed
- **`📋 All params (JSON)` button in the Outputs lightbox returned `{}` for all images that were backfilled by Reindex.** Root cause: `_build_image_entry()` was only populating `metadata` when called by the live `on_image_logged()` hook (which Fooocus passes the metadata tuple list to). The reindex path called it with `metadata=None` → empty dict → `JSON.stringify({})` → `{}`. Affected every image generated before the user enabled Asset Browser.
### Changed
- **`modules/gallery_writer.py::_extract_metadata_from_image_file(path)`** new helper that opens the PNG/JPEG/WebP via Pillow and calls Fooocus's existing `meta_parser.read_info_from_image()` to extract the metadata. Two cases:
  - Fooocus JSON scheme → returns the parsed dict directly (structured: `sampler`, `cfg_scale`, `steps`, `seed`, `prompt`, etc.).
  - A1111 string scheme → returns `{'parameters': '<the raw multiline string>'}` so the Copy All Params button has something to copy AND the lightbox can render it as a Parameters code block.
- **`_build_image_entry(metadata=None)`** now falls back to `_extract_metadata_from_image_file(image_path)` when no metadata was passed in. Existing call sites are unchanged: the live hook still passes its tuple list, the reindex passes nothing → backfill kicks in.
- **`captionImage()` in the SPA** now handles 3 metadata shapes:
  - Structured Fooocus dict → renders the existing `Sampler / CFG / Steps / Seed / …` rows.
  - Raw A1111 string → renders a `Parameters` row with a scrollable `<code>` block showing the full A1111 line; Copy button copies the raw A1111 string instead of `{}`.
  - Empty (image predates the hook AND has no PNG `parameters` chunk) → shows a hint pointing to the Reindex button.
- **`_build_image_entry(metadata=...)`** also now accepts a flat dict (in addition to the tuple list it already accepted), so callers that already have a dict don't need to convert.

## [custom-8.5] — 2026-05-03 — Drop eager preview-dimension probing (perf fix)
### Fixed
- **Model tabs no longer fetch every full-resolution preview just to read its dimensions.** The previous SPA `renderGrid` did `await Promise.all(filtered.map(it => probeDimensions(it.preview_full)))` to feed PhotoSwipe correct widths/heights. On a tab with thousands of items (5234 LoRAs in the user's case), this triggered thousands of HTTP fetches of preview files (10 KB placeholders to multi-MB sidecars) on every tab open / subfolder change. Could blow up to several GB of unnecessary downloads.
### Changed
- **`modules/model_indexer.py`** now probes preview dimensions once at index time via Pillow's lazy header read (`Image.open(...).size`) and stores `preview_width` / `preview_height` in each manifest item. Placeholders skip the probe entirely (always `PLACEHOLDER_FULL_SIZE` square = 1024×1024).
- **`gallery_template/index.html`**: model tabs read `it.preview_width` / `it.preview_height` directly from the manifest. Default 1024×1024 if missing (manifests built before this change won't have the fields — `Reindex everything now` rebuilds them). The eager `probeDimensions` call is removed; the function itself is kept because the Outputs tab still uses it as a fallback for old PNGs whose manifest lacks dimensions.
### Notes
- Grid thumbnails ARE still lazy-loaded via the browser-native `<img loading="lazy">` (unchanged).
- Lightbox slides are loaded on-demand by PhotoSwipe (unchanged).
- The fix only removes the EAGER probe per item at tab-open time.

## [custom-8.4] — 2026-05-03 — Fetch CivitAI preview from the lightbox
### Added
- **🌐 Fetch from CivitAI** button inside the PhotoSwipe lightbox caption sidebar. Visible only on models whose preview is a placeholder (no sidecar yet). One click → downloads the top-rated CivitAI image for that model and saves it as `<stem>.preview.png` next to the model file (A1111 / ComfyUI sidecar convention — visible in those tools too). After save, the model's manifest is rebuilt so the SPA reloads with the real preview.
- **`modules/model_indexer.py::fetch_civitai_preview_for(kind, rel_filename, api_key=None)`** — backend function. Resolves the model path, hashes it via the existing `civitai_api._get_full_sha256`, calls `civitai_api.get_model_version_by_hash` then `get_top_images`, downloads the first image via `urllib.request`, normalises to PNG via Pillow, writes the sidecar. Refuses to overwrite an existing sidecar (would silently destroy a hand-curated preview — user must delete first). Invalidates the cached thumbnail / placeholder for that model id so the next manifest rebuild picks up the new sidecar.
- **Hidden Gradio API endpoint** in `webui.py` — `gr.Button(visible=False).click(api_name='ab_fetch_preview')` exposes `_ab_fetch_civitai_preview(kind, rel_filename)` at `/run/ab_fetch_preview` (and `/api/ab_fetch_preview` as fallback). Pure backend bridge, zero UI.
- **SPA-side JS handler** `fetchCivitaiPreview(kind, relPath, btn)` — POSTs to the endpoint, shows live state on the button (`⏳ Fetching from CivitAI…` → `✓ Preview fetched — reloading…` → `location.reload()`, or `✗ Not found on CivitAI` / `✗ Network error` on failure). Delegated click handler so PhotoSwipe rebuilding the caption DOM between slides keeps it working.
- **Standalone-mode awareness**: the Fetch button is hidden when the SPA is opened via `file://` (no Fooocus to call). `HAS_FOOOCUS` constant set from `window.location.protocol`.
### Larger placeholders (companion fix)
- Placeholder PNGs are now generated at fixed **1024×1024** (`PLACEHOLDER_FULL_SIZE`) instead of the same 256×256 used for the grid thumbnail. This fixes the visual misalignment in the lightbox — image and caption now have comparable footprints. The 256×256 grid thumb is still derived from this PNG via `_make_preview_thumb`. Gradient is generated via numpy broadcasting (~245 ms per placeholder vs ~5-10 s with putpixel).
### Notes
- Refusing overwrite is intentional (data-loss avoidance). v2 idea: a "Force overwrite" toggle near the button or a "Delete current preview" button.
- Disk overhead per fetched preview: same as a typical CivitAI image, ~200-500 KB per model. For 100 missing models = ~20-50 MB max.

## [custom-8.3] — 2026-05-03 — Async reindex worker (no more browser timeouts)
### Changed
- **Reindex now runs in a daemon thread** — the click handler returns immediately so the browser never blocks waiting for the multi-minute job (previously caused `ConnectionResetError WinError 10054` when the HTTP request timed out around 5+ minutes for ~25k images).
- **Live status polling**: the Asset Browser accordion's status line is now refreshed every 2 s by `shared.gradio_root.load(every=2)`. While a reindex is running it shows `🔄 Outputs: [42/135] · current: 2024-06-08 · 8124 image(s) processed so far`, then `🔄 Outputs done: 135 days · scanning models…`, then `✓ Reindex complete: 135 day(s), 24831 image(s). · models: 5234 LoRAs, 153 ckpts, 603 embeds`.
- **Re-click protection**: clicking Reindex while one is already running shows `A reindex is already running. Watch its progress below.` instead of stacking workers.
### Added
- New API in `modules/gallery_writer.py`:
  - `reindex_outputs_async()` — spawns the worker, returns `(started, message)` immediately.
  - `reindex_status()` — pollable snapshot dict (`{ phase, days_total, days_done, days_seen, current_date, images_total, last_summary, ... }`).
  - `_reindex_worker()` — the body of the previous synchronous `reindex_outputs()`, refactored to update the state dict as it iterates.
  - The synchronous `reindex_outputs()` is kept as a thin wrapper for any external callers / tests (busy-waits with light sleep until the worker finishes).
### Notes
- The boot-time `model_indexer.maybe_start_boot_scan()` and the user-triggered async reindex use **separate locks** — that's why the previous console showed `[asset-browser] scan complete: {...}` interleaved between date dirs. Cosmetic only; both finish correctly.
- Polling cost: a single dict copy under a lightweight lock every 2 s. Negligible.

## [custom-8.2] — 2026-05-03 — Lightbox readability: full-res model previews + wider caption
### Changed
- **Lightbox now shows the full-resolution sidecar preview** (was the 256×256 thumbnail). On open, model_indexer now also caches a copy of the original sidecar to `outputs/_previews/<kind>/<hash>_full.<ext>` (PNG/JPG kept as-is). Manifest entries gain a `preview_full` field; the SPA prefers it over `preview` for the lightbox slide src and probes its real dimensions for correct PhotoSwipe centring/zoom. For models with no sidecar, `preview_full` falls back to the 256×256 placeholder PNG.
- **Wider Dynamic Caption sidebar**: width 320px → 440px, max-width 360px → 520px, font 13px → 14px, h2 16px (was 14px), padding 14/16 → 18/20, line-height 1.45 → 1.5. Long filenames like `Anime Summer Days Style SDXL_LoRA_Pony Diffusion V6 XL.safetensors` now wrap on word boundaries instead of mid-character, the metadata label column is wider (90→100px) for better alignment.
- **Generous lightbox padding**: switched from `padding: { left: 320, ... }` (which was reserving room on the wrong side) to a uniform `padding: 30px` all around. Dynamic Caption auto-positions next to the image and switches to 'below' mode when there isn't enough horizontal room — the previous left-bias was forcing the image off-centre.
- **Disk overhead notice**: the full-sidecar copy roughly doubles disk usage of `_previews/`. For typical setups (sidecar PNGs 0.5-2 MB each, ~100 LoRAs) that's ~50-200 MB max — acceptable. Cached by source mtime so re-indexing is idempotent.

## [custom-8.1] — 2026-05-03 — Blur NSFW + Hide days/subfolders
### Added
- **NSFW thumbnail blur** — new `asset_browser.blur_thumbnails` config (default `false`) + UI checkbox in the Advanced > 🖼️ Asset Browser accordion. When ON, the SPA blurs every thumbnail (`filter: blur(20px) saturate(0.7)`); hover or click reveals individually. The Browser header has a 🌫️ Blur toggle that lets each user override the config default temporarily (sticky in browser localStorage). Useful on shared screens / NSFW-adjacent collections.
- **Hide / show days and model subfolders** — pure SPA / localStorage feature (no Fooocus changes for the per-item state, keeps autonomy intact). Hovering on any sidebar item reveals a small `🚫 hide` button; clicking it removes the item from the list. The header has a `👁️ Hidden` badge with the count of hidden items; clicking the badge toggles "show hidden" mode where they reappear greyed-out with a `↩ unhide` button. Storage scope is per-tab (`outputs:days` / `loras:subfolders` / `checkpoints:subfolders` / `embeddings:subfolders`) — hiding e.g. a Flux subfolder in the LoRAs tab doesn't affect the same folder anywhere else. Hidden items are also filtered out of the grid in `__all__` view of model tabs unless show-hidden is on.
- **`outputs/_index/spa_settings.json`** — new JSON written by `gallery_writer.ensure_gallery_assets()` mirroring the SPA-relevant subset of `asset_browser` config (`blur_thumbnails`, `thumbnail_size`, `generated_at`). Keeps the SPA fully autonomous (no Gradio endpoint) while letting Fooocus push defaults to it.
### Changed
- `modules/config.py` — `_asset_browser_defaults` extended with `blur_thumbnails: false`; `write_asset_browser_settings()` validator includes it in `_bool_keys`.
- `webui.py` — Asset Browser accordion gains the 🌫️ Blur checkbox; `_ab_save_and_status` now persists 5 keys instead of 4 and refreshes `spa_settings.json` automatically when saving.
- `gallery_template/index.html` — header gains a right-side toggle cluster (🌫️ Blur, 👁️ Hidden); CSS adds `.blur-active` body class + per-item hide button + greyed `.hidden-row` style; new vanilla-JS `State` module (~80 lines) handles localStorage round-trip + delegated hide/unhide click handler + boot-time spa_settings.json fetch.

## [config extension] — 2026-05-03 (post-custom-8)
### Added
- **`path_civitai_cache`** in `config.txt` — was hardcoded `./civitai_cache`, now configurable. Default value preserves the historical behaviour, so existing installs see no change. Useful when the user wants the CivitAI response cache on a different drive (large LoRA collections + checkpoints can produce hundreds of MB of cache).
- **Extended `asset_browser` block** with 4 new tunable sub-keys:
  - `thumbnail_size` (default 256, clamp 64..1024) — square JPEG thumbnail dimension. Setting 128 cuts disk usage and grid load time by ~75%.
  - `thumbnail_quality` (default 85, clamp 40..100) — JPEG quality. 70 ~halves thumbnail file size at minor visual cost.
  - `dzi_threshold_mp` (default 4.0, clamp 0.5..64.0) — megapixel threshold for `generate_dzi_tiles='auto'` mode. (DZI generation itself still deferred.)
  - `placeholder_label_max` (default 24, clamp 8..64) — character limit for the filename overlay on auto-generated placeholder previews.
### Changed
- `modules/civitai_api.py::CIVITAI_CACHE_DIR` is now computed lazily from `modules.config.path_civitai_cache` (with `./civitai_cache` fallback for back-compat). The legacy module-level constant is kept for any external code that imports it.
- `modules/gallery_writer.py` and `modules/model_indexer.py` no longer hardcode 256/85 — they read `thumbnail_size` and `thumbnail_quality` from the config block via `asset_browser_setting()`. `model_indexer.py` also reads `placeholder_label_max` for the placeholder overlay truncation logic.
- `write_asset_browser_settings()` validator extended with per-key clamp ranges so a malformed UI value can never poison `config.txt`.

## [custom-8] — 2026-05-03 — Asset Browser (autonomous module)
### Added
- **🖼️ Asset Browser** — a standalone HTML/JSON gallery served from `outputs/index.html`, opened in a new browser tab from the **🖼️ Asset Browser** link next to **📚 History Log**. Built on **PhotoSwipe v5** + **Dynamic Caption plugin** + **Deep Zoom plugin** (all MIT, ESM modules bundled in `outputs/_assets/`, no CDN). Vanilla JS SPA — no React/Vue, ~50 KB gzip total payload.
- **4 tabs**, all with PhotoSwipe lightbox + per-type metadata sidebar (Dynamic Caption):
  - **📅 Outputs** — timeline sidebar (days, anti-chrono, "today" highlighted) + grid of thumbnails + lightbox with prompt / negative / sampler / cfg / steps / seed / resolution / model + 📋 Copy buttons (Prompt / Negative / All params JSON).
  - **🎨 LoRAs** — sidebar = subfolder facets, lightbox with triggers + size + 📋 Copy triggers / Copy filename + 🔗 Open on CivitAI link.
  - **📦 Models (Checkpoints)** — same UX + base_model + CivitAI consensus settings (sampler / cfg / steps / clip_skip / top_resolution) read from the existing `civitai_cache/` (no fresh API calls).
  - **🧩 Embeddings** — sidebar = subfolder facets, lightbox with trigger + negative-hint flag + 📋 Copy `(embedding:name:1.0)` token / Copy trigger only.
- **Toggle infrastructure (OFF by default — non-negotiable):** the feature is disabled by default. When `asset_browser.enabled = false` in `config.txt`, the per-image hook in `private_logger.py` returns in <1µs (single dict access) and the model indexer thread is never spawned. Indistinguable from not having the feature installed. UI accordion in **Advanced** with master enable + sub-toggles (`generate_thumbnails`, `generate_dzi_tiles` auto/always/never, `index_models_on_boot`) + 💾 Save Settings (writes config.txt) + 🔄 Reindex Now button.
- **New module `modules/gallery_writer.py`:**
  - `on_image_logged()` — silent hook called from `private_logger.log()`. Generates 256×256 JPEG centre-cropped thumbnail, appends entry to `outputs/<DATE>/manifest.json` (idempotent on retries), refreshes `outputs/_index/days.json`. Process-locked.
  - `reindex_outputs()` — walks every existing date subdir and rebuilds manifests + thumbs + bundled model scan. Wired to the Reindex Now button.
  - `ensure_gallery_assets()` — copies `gallery_template/index.html` + `_assets/*` to `outputs/`. Idempotent, runs on every image log so template upgrades land for free.
- **New module `modules/model_indexer.py`:**
  - `scan_loras()` / `scan_checkpoints()` / `scan_embeddings()` — walk the lists already populated by `config.update_files()`.
  - **Sidecar preview lookup in 5 patterns** (A1111/ComfyUI compatible): `<stem>.preview.png` → `.preview.jpg` → `.png` → `.jpg` → `_preview.png` → **placeholder** (hash-derived gradient PNG with filename overlay, cached by sha1 of rel_path).
  - **256×256 JPEG preview thumbnails** in `outputs/_previews/<kind>/<hash>.jpg`, invalidated by source mtime so model swaps are picked up.
  - **Cache-only CivitAI reads** via `load_cached_triggers` / `load_cached_settings` — never makes fresh API calls (would rate-limit on bulk scans). Triggers come from local safetensors metadata (existing `lora_metadata` module) merged with cached CivitAI trainedWords. Checkpoints get `base_model` + consensus settings from the cache.
  - `maybe_start_boot_scan()` — daemon thread spawned at startup only when toggle + `index_models_on_boot` are both true.
- **New `gallery_template/`** packaged in the repo:
  - `index.html` — the SPA (~25 KB).
  - `_assets/photoswipe.{esm.js,css}` (5.4.4)
  - `_assets/photoswipe-lightbox.esm.js`
  - `_assets/photoswipe-dynamic-caption-plugin.{esm.js,css}` (1.2.7)
  - `_assets/photoswipe-deep-zoom-plugin.esm.js` (1.1.2)
- **Cross-app communication = clipboard only.** All Copy buttons in the lightbox put data on the clipboard; user pastes into Fooocus by hand. No Gradio endpoint added — keeps the SPA fully autonomous (consultable via `file://` even when Fooocus is not running). Future v2 can add an iframe/postMessage bridge for direct picker integration with the LoRA/Checkpoint dropdowns.
- **URL hash navigation** — `#outputs` / `#loras` / `#checkpoints` / `#embeddings` for bookmarkable / shareable tab state.
### Changed
- `modules/private_logger.py` — +9 lines: silent try/except hook into `gallery_writer.on_image_logged()` after the final `print()`. Wrapped so any bug in the gallery writer can never break image generation.
- `launch.py` — +8 lines: `model_indexer.maybe_start_boot_scan()` before `from webui import *`, also wrapped in try/except.
- `webui.py` — new accordion in Advanced (right after the Documentation link); second link button next to History Log in the prompt area (greyed out when disabled).
- `modules/config.py` — new `asset_browser` config block (dict) + `asset_browser_enabled()` / `asset_browser_setting()` / `write_asset_browser_settings()` helpers.
### Deferred to follow-up
- **DZI tile generation** for very large images — toggle exists (`generate_dzi_tiles`) and Deep Zoom plugin is wired, but the Python-side tile generation is not in this commit. Without DZI tiles, the Deep Zoom plugin falls back to PhotoSwipe's normal pinch/scroll zoom on the full image — fine for outputs (typically <2 MP) and model preview thumbs.
- **Full-resolution model previews** in the lightbox — currently shows the 256×256 thumb. Sidecars > 256×256 will get pass-through in a follow-up.

## [version bump] — 2026-05-03
### Changed
- **Version bumped to `2026.2.0`** (`fooocus_version.py`) for the custom-8 feature ship.

## [version bump] — 2026-05-02
### Changed
- **Version bumped from `2.5.5` to `2026.1.0`** (`fooocus_version.py`). The fork has diverged enough from upstream v2.5.5 that the inherited version string was confusing users into thinking the fork was unmaintained. Switched to CalVer (`YYYY.MINOR.PATCH`). PNG metadata `Version` field and the WebUI title both now read `Fooocus v2026.1.0`.

## [custom-7] — 2026-05-02
### Added
- **Aspect Ratios switched from Radio block to Dropdown**, with **`Custom`** as the first entry. Selecting any preset behaves exactly like before; selecting `Custom` reveals the Custom Resolution panel and routes generation through it. Choices use `(display_label, value)` tuples so the dropdown shows clean text (`1152×896 ∣ 9:7`) while the value passed to downstream code keeps the original HTML-formatted form for back-compat with `meta_parser` and preset round-trip.
- **Custom Resolution panel** under the Aspect Ratios accordion (revealed by selecting `Custom` in the dropdown). Lets users pick any aspect ratio + size on the fly without editing `config.txt` or restarting Fooocus.
  - **Toggle**: driven by the dropdown — selecting any preset other than `Custom` hides the panel and disables the override (visible-only checkbox is gone; an internal hidden flag still flows to the worker).
  - **Inputs**: ratio `W`/`H` (integers) + a **mode selector** (`Max edge` / `~1 MP target` / `Min edge`) + a **size slider** (512–2048, step 64).
  - **Quick ratio chips**: `1:1` `3:2` `4:3` `16:9` `21:9` `√2 (A4)` populate the ratio fields in one click. **🔄 Swap** flips W ↔ H for landscape/portrait.
  - **Live computed display**: `→ 1344 × 768 · 1.03 MP · 7:4` updates on every input change. Warns when total pixels fall below 0.25 MP or above 2 MP (non-blocking).
  - **Auto-snap to /64**: results are always rounded to multiples of 64 on both axes (SDXL hard requirement).
  - **💾 Save as preset entry**: one-click button that appends the computed `W*H` to `available_aspect_ratios` in `config.txt`, so the resolution shows up in the standard radio block on next restart.
- **Preset round-trip**: `save_preset_to_file` now writes a `custom_resolution` block (`{enabled, ratio_w, ratio_h, mode, size}`) into preset JSONs. Old presets without the block default to `enabled: false` (full back-compat). The block is invisible to vanilla Fooocus (not in `possible_preset_keys` allowlist), so cross-loading is safe.
### Added — utility
- New helper `compute_custom_wh(ratio_w, ratio_h, mode, size)` in `modules/util.py` — single source of truth for the snapping math, used by both the WebUI display and the worker override.
### Changed
- `modules/async_worker.py::AsyncTask` consumes 5 new flags after `use_aspect_for_vary`. When `custom_res_enabled` is true (or the dropdown sentinel value `'Custom'` arrives, as a safety net), the computed `W×H` is stuffed back into `self.aspect_ratios_selection` so the existing parsers (main path at line 1135, Vary path at line 458) pick up the override transparently — no downstream changes needed. Vary + custom-7 + custom-6 stack: selecting `Custom` plus turning on `Use Aspect for Vary` forces Vary outputs to the custom W×H.

### Fixed
- **Embeddings: duplicate trigger words from Windows-suffixed filenames.** When an embedding file like `neg_realism512 (1).safetensors` was selected, clicking *Prompt* / *Negative* inserted both `(embedding:neg_realism512 (1):1.0)` AND a redundant `neg_realism512` extra trigger (the canonical name returned by CivitAI, which differed from the local stem only by the Windows-duplicate ` (N)` suffix). Fixed by normalising stems and CivitAI candidates with `\s*\(\d+\)\s*$` stripped before the dedup check (`webui.py::_normalize_emb_token`). Now only the embedding tag itself is inserted; the redundant canonical name is correctly recognised as a duplicate of the local stem.

## [custom-6] — 2026-04-18
### Added
- **"Use selected Aspect Ratio for Vary" checkbox** under the Aspect Ratios accordion in the Advanced tab. When enabled, Vary (Subtle) and Vary (Strong) resize the input image to the selected aspect ratio's dimensions (centre-crop, `resize_mode=1`) before encoding, instead of following the input image's native shape-ceil. Unchecked = original upstream behavior. Does not affect Upscale (which keeps its fixed 1.5x/2x factor).

### Changed
- `modules/async_worker.py::AsyncTask` now consumes a new `use_aspect_for_vary` flag. The control is appended to `ctrls` right after `mixing_image_prompt_and_inpaint` and popped in the matching order.

## [custom-5] — 2026-04-18
### Added
- **Checkpoint trigger words** in the CivitAI panel. The fetch result now also surfaces the checkpoint's `trainedWords` (score_9 for Pony, NSFW-style activation tokens for some merges, etc.) in a highlighted block above the settings table. A **📋 Copy checkpoint triggers to prompt** button appears under the panel when triggers are available.
- **💾 Save CivitAI consensus as preset** — button under the CivitAI Apply control that writes a new preset `.json` combining:
  - The current base model, refiner, styles, aspect ratio, prompts, LoRAs, and embeddings.
  - The CivitAI consensus sampler / scheduler / CFG / steps / clip-skip, overriding whatever was in the UI.
  - Default preset name auto-suggested as `civitai_<ModelName>`; editable.
  - Calls the existing `save_preset_to_file()` so new presets appear in the preset dropdown and can be re-loaded normally.

## [custom-4] — 2026-04-18
### Added
- **Textual Inversion / Embeddings panel** in the Advanced tab (5 slots, matching the LoRA count).
  - Each slot: dropdown of available embeddings from `models/embeddings/`, weight slider, enable checkbox.
  - Mirrors the LoRA trigger-words UX: activation token is auto-detected (filename + any extra tokens from safetensors metadata or CivitAI), shown in a read-only field below the row.
  - **📋 To prompt** / **📋 To negative** buttons per slot insert `(embedding:<name>:<weight>)` into the corresponding textbox.
  - **Insert ALL active to prompt / negative** buttons below the rows insert every enabled embedding's activation token at once.
### Changed
- **Negative prompt moved** to the main prompt area, directly below the positive prompt. Previously buried in the Advanced tab; now visible without enabling Advanced. Styled to match the positive prompt (shared CSS on `#positive_prompt`, `#negative_prompt` — same rounded corners and background). Small spacer between the two textboxes.
- **Preset Manager moved** out of the Developer → Debug Tools tab and into the Advanced → main aspect section (where the negative prompt used to be). Wrapped in its own `gr.Accordion` labelled "💾 Preset Manager" (collapsed by default).
- **Preset Manager now saves embeddings** alongside LoRAs. Saved as `default_embeddings` in the preset JSON with the same `[enabled, name, weight]` shape as `default_loras`. (Loading embeddings from a preset on preset-switch is not yet wired — follow-up.)
- **Collapsible sections** in the Advanced tab — CivitAI, LoRA, and Embeddings are now `gr.Accordion` panels. LoRA starts open (most-used), CivitAI and Embeddings start collapsed to reduce visual clutter.
### Added (continued)
- **Wildcards panel** in the Advanced tab (below Embeddings) with full editor:
  - Dropdown of all `.txt` files in the wildcards folder.
  - **Editable scrollable text area** with the file's full contents (not just a preview — edit in place).
  - **💾 Save** writes the edited contents back to the selected file.
  - **➕ Create new** — type a name (letters/digits/_/- only) and save the current contents as a new `.txt` in the wildcards folder; name is sanitised and collisions are rejected.
  - **📋 Insert __token__ to prompt** inserts the `__filename__` token into the positive prompt, deduped.
  - Filesystem changes trigger `update_files()` so the dropdown refreshes immediately.
- **⚠️ Restart UI button** next to Refresh All Files — re-execs the Python process (`os.execv`) so edits to `config.txt`, custom Python modules, or external model additions are picked up cleanly. User must refresh the browser tab after ~30 s (SDXL reload on RTX 5090). Clearly labelled as secondary to "Refresh All Files", which already handles new-file discovery without restart.
### Refactored
- Generalised `fetch_lora_triggers_combined` → `fetch_model_triggers_combined(filename, paths, kind, api_key)` in `modules/civitai_api.py`; `kind='lora'` and `kind='embedding'` share the same pipeline with separate cache namespaces (`<name>.lora.civitai.json` / `<name>.embedding.civitai.json`). Legacy `fetch_lora_triggers_combined` kept as a shim for back-compat.
- `modules/lora_metadata.py` extended with `extract_embedding_triggers_from_metadata()` and `get_embedding_triggers_from_file()` — for embeddings the filename stem is always the primary trigger; extra tokens from `sd_embedding_tokens` / `modelspec.trigger_phrase` are appended when present.
- `modules/config.py`: new `embedding_filenames` list, populated by `update_files()` scanning `path_embeddings` for `.safetensors`/`.pt`/`.bin`.

## [custom-3] — 2026-04-18
### Added
- **LoRA trigger words — local metadata + CivitAI, merged**
  - Each LoRA slot now shows a read-only field with the LoRA's trigger words, auto-fetched on selection.
  - **Two sources, combined for coverage:**
    1. **Local safetensors metadata** (new `modules/lora_metadata.py`) — reads the `__metadata__` header of the `.safetensors` file to extract `modelspec.trigger_phrase`, top tags from `ss_tag_frequency` (filtered by frequency), and `ss_output_name`. Instant, offline, works for LoRAs never uploaded to CivitAI.
    2. **CivitAI `trainedWords`** — hash-based lookup as before, cached to disk per LoRA.
  - Merged list is deduped, local-first (ground truth from training), CivitAI-only extras appended.
  - **📋 Copy to prompt** button per slot appends that LoRA's triggers to the positive prompt, de-duplicated against what's already there.
  - **📋 Copy ALL active LoRA triggers to prompt** button below the LoRA rows gathers triggers from every *enabled* LoRA at once.
  - New helpers: `fetch_lora_triggers()`, `fetch_lora_triggers_combined()`, `format_lora_triggers_display()` in `modules/civitai_api.py`; `read_safetensors_metadata()`, `extract_triggers_from_metadata()`, `get_lora_triggers_from_file()` in `modules/lora_metadata.py`. Miss results are cached so LoRAs not on CivitAI don't keep hitting the API.

## [custom-2] — 2026-04-11
### Added
- **CivitAI Model Settings integration**
  - New module `modules/civitai_api.py` — CivitAI client with response caching.
  - Fetches community-recommended generation settings (sampler, CFG, steps, clip skip) for the currently selected model, aggregated from top-rated images on CivitAI.
  - New panel in the UI shows the consensus analysis.
  - **Apply** button injects recommended settings into the Fooocus UI.
  - CivitAI API key configurable via UI; persisted to `config.txt`.
  - API responses cached locally in `civitai_cache/` (git-ignored).

## [custom-1] — 2026-04-11
### Added
- **Save Preset button** in the Advanced → Developer tab.
  - Located in Developer Debug Tools, after the Metadata Scheme.
  - Saves the current Advanced settings as a new preset `.json`, or overwrites an existing preset, directly from the UI.

## Base
- Upstream Fooocus **v2.5.5** (commit `ae05379`).
