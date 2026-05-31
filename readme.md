<div align=center>
<img src="https://github.com/lllyasviel/Fooocus/assets/19834515/483fb86d-c9a2-4c20-997c-46dafc124f25">
</div>

# Fooocus 2025 — Custom Fork

> Version **`2026.2.0`** · A personal fork of **[lllyasviel/Fooocus](https://github.com/lllyasviel/Fooocus) v2.5.5** with a series of quality-of-life features: a **Save Preset** button, **CivitAI Model Settings** integration (checkpoint triggers, consensus settings, save-as-preset), **LoRA trigger words** from local metadata + CivitAI, **Embeddings panel** with bulk-insert, **Wildcards editor**, **Vary-with-aspect-ratio** override, **Custom Resolution** (any ratio + size, snapped to /64), an **🖼️ Asset Browser** (PhotoSwipe-based standalone gallery for outputs + LoRAs/Checkpoints/Embeddings previews — opt-in, zero impact when disabled), **architecture filtering** (auto-hides Flux/SD3/LLMs from dropdowns — SD/SDXL only), and a real **Restart UI** button.

![Fooocus2025 fork — Models tab showing CivitAI / LoRA / Embeddings / Wildcards accordions and Restart UI, with wildcards in the prompt](docs/screenshots/overview.png)

*Above: the Models tab of the Advanced panel, showing the four fork-specific accordions (CivitAI, LoRA, Embeddings, Wildcards) and the Restart UI button. The generated images come from a prompt that uses two wildcards: `__artist-anime__` for the style and `__neg-weight__` in the negative — each expands to a random line from the matching `.txt` at generation time.*

**Original project:** [github.com/lllyasviel/Fooocus](https://github.com/lllyasviel/Fooocus) · **Original Windows package (v2.5.0):** [Fooocus_win64_2-5-0.7z](https://github.com/lllyasviel/Fooocus/releases/download/v2.5.0/Fooocus_win64_2-5-0.7z)

> 💡 **For the complete upstream documentation** (installation, models, troubleshooting, CLI flags, etc.), scroll down to the [**Original Fooocus README**](#-original-fooocus-readme) section further down this page. This top section only documents what's different in this fork.

---

## ✨ What's new in this fork

### 1. 💾 Save Preset button
**Where:** `Advanced` tab → `Developer` sub-tab → *Developer Debug Tools* (below the Metadata Scheme control).

**What it does:** Saves your current Advanced-tab settings (sampler, steps, CFG, refiner, styles, etc.) as a reusable preset `.json` file in the `presets/` folder — without having to hand-edit JSON.

**How to use:**
1. Tune your settings in the Advanced tab as usual.
2. Open `Advanced → Developer → Developer Debug Tools`.
3. Type a name in the preset field:
   - A **new name** → creates `presets/<name>.json`.
   - An **existing preset name** → overwrites that preset.
4. Click **Save Preset**. The preset appears in the normal preset dropdown on next launch.

---

### 2. 🏷️ LoRA trigger words (local metadata + CivitAI, merged)
**Where:** below each LoRA slot in the Advanced tab.

**What it does:** Auto-detects trigger words for the selected LoRA from **two sources** and merges them in a small read-only field. One click injects them into your positive prompt — most LoRAs only activate their training when these tokens are present in the prompt.

**Two sources, combined for coverage:**
1. **Local safetensors metadata** — reads the `__metadata__` header directly from the `.safetensors` file. Pulls `modelspec.trigger_phrase`, the top-N most-frequent tags from `ss_tag_frequency`, and `ss_output_name` as fallback. **Instant, offline**, works for LoRAs never uploaded to CivitAI.
2. **CivitAI `trainedWords`** — hash-based lookup of the LoRA on CivitAI (cached per file).

Local triggers appear first (ground truth from training); any CivitAI-only extras are appended, deduped.

**How to use:**
1. Pick a LoRA in any of the 5 slots.
2. Triggers auto-fetch and appear under the row. Local read happens instantly; CivitAI lookup runs in the background on first fetch (cached from then on).
3. Click **📋 Copy to prompt** next to that slot — triggers are appended to the main prompt, de-duplicated against what's already there.
4. Or click **📋 Copy ALL active LoRA triggers to prompt** below the rows to pull triggers from every *enabled* LoRA at once.

CivitAI responses are cached in `civitai_cache/<lora_name>.lora.civitai.json` (git-ignored). LoRAs not found on CivitAI are also cached. Local metadata is read fresh each time (it's a sub-millisecond file read).

---

### 3. 🧩 Textual Inversion / Embeddings panel
**Where:** Advanced tab, right below the LoRA section.

**What it does:** Gives embeddings a proper UI instead of the hidden `(embedding:filename:weight)` syntax that most users never discover. **5 slots** (matching LoRA), each with a dropdown of available embeddings from `models/embeddings/`, a weight slider, and two buttons to inject the activation token into either the positive or negative prompt (embeddings are often negative-prompt tools — `BadHands`, `FastNegativeV2`, `unaestheticXL`, etc.).

**How to use:**
1. Select an embedding in any slot. Its activation token appears in the read-only field below the row (auto-detected — the filename stem, plus any extra tokens from safetensors metadata or CivitAI).
2. Set a weight (default 1.0).
3. Click **📋 Prompt** or **📋 Negative**. What gets inserted:
   - The token `(embedding:<name>:<weight>)`.
   - The **canonical keyword** (the filename stem, with any Windows-duplicate ` (N)` suffix stripped — so `lazyneg (1).safetensors` produces `lazyneg`).
   - Any **extra CivitAI / metadata trigger words** that differ from the canonical keyword.
   - Everything deduped against what's already in the textbox.
4. Or fill multiple slots and use **📋 Insert ALL active embeddings to prompt / negative** to inject all *included* slots at once.

The **Include** checkbox per slot only filters the Insert-ALL bulk button — embeddings activate purely from their token being in the text, not from any enable flag. The per-slot buttons always work regardless of the checkbox.

> 🩹 *(custom-7) Fix:* embeddings whose filename ends with the Windows-duplicate marker ` (1)` / ` (2)` / … no longer produce a duplicate trigger word. The canonical name (without the suffix) is appended once; the suffixed variant returned by CivitAI / local metadata is recognised as the same token and dropped.

Uses the same local-metadata + CivitAI merged-triggers pipeline as the LoRA feature (`fetch_model_triggers_combined(kind='embedding')`). Cache: `civitai_cache/<name>.embedding.civitai.json`.

---

### 4. 🎲 Wildcards editor
**Where:** Advanced tab, below the Embeddings section (collapsible accordion).

**What it does:** Gives Fooocus's built-in wildcards a full UI with an inline editor. Browse every `.txt` in your wildcards folder, edit the entries directly, save changes, create new wildcard files — all without leaving Fooocus.

**How to use:**
1. Open Advanced → 🎲 Wildcards.
2. Pick a file from the dropdown (lists every `.txt` in your wildcards folder, e.g., `animal`, `artist-anime`, `adj-general`).
3. The contents load into a scrollable editable text area (one entry per line).
4. **Edit** entries directly in the text area, then click **💾 Save** to overwrite the file.
5. To create a new wildcard: type a name in the new-name field, put the lines you want in the editor, click **➕ Create new** — a new `.txt` is written to the wildcards folder and the dropdown refreshes.
6. Click **📋 Insert __token__ to prompt** — the corresponding `__<filename>__` token is appended to the positive prompt, de-duplicated.
7. Generate — Fooocus expands each occurrence of the token to a random line from the file at generation time.

Filenames are sanitised (letters/digits/underscore/dash only). Create rejects collisions.

---

### 5. 🎨 CivitAI Model Settings integration
**Where:** Advanced tab → CivitAI accordion (collapsible, next to Refiner).

**What it does:** Queries [CivitAI](https://civitai.com/) for the currently selected checkpoint, aggregates the generation settings used in the **top-rated community images** for that model, and shows a consensus view (most-used **sampler**, **CFG scale**, **steps**, **clip skip**). Also surfaces the checkpoint's **trigger words** (`trainedWords` — e.g. `score_9, score_8_up, …` for Pony models) and lets you save the whole consensus as a reusable preset.

**How to use:**
1. **One-time setup:** paste your CivitAI API key into the field in the CivitAI panel — it is saved to `config.txt` for future sessions. Get a key at [civitai.com/user/account](https://civitai.com/user/account) (API Keys section).
2. Select any checkpoint in the model dropdown.
3. Expand the CivitAI accordion and click **🔍 Fetch CivitAI Settings**. First fetch for a new file runs a hash → CivitAI lookup; subsequent fetches are instant from the local cache (`civitai_cache/<name>.civitai.json`).
4. The panel displays:
   - **Triggers block** (if CivitAI lists any) — the checkpoint's `trainedWords`.
   - **Consensus settings table** — sampler (CivitAI name + Fooocus-mapped name), CFG scale (median + range), steps (median + range), clip skip, and the top resolution.
5. Click **✅ Apply These Settings** to inject sampler/CFG/steps/clip-skip into the Advanced tab.
6. Click **📋 Copy checkpoint triggers to prompt** (appears only when triggers exist) to append them, deduped.
7. Click **💾 Save CivitAI consensus as preset** to write a new preset `.json` combining:
   - the CivitAI consensus sampler/scheduler/CFG/steps/clip-skip,
   - the current base model, prompts, styles, aspect ratio, LoRAs, and embeddings.
   - Default preset name auto-suggested as `civitai_<ModelName>`; editable.

**Why it's useful:** every SDXL checkpoint has its own "sweet spot" parameters, and the uploader's page rarely lists them clearly. This reads them directly from what actually worked for the highest-rated outputs on CivitAI — and composes with your LoRA/embedding setup to produce a ready-to-reuse preset in one click.

**Cache location (configurable):** CivitAI responses are cached at `./civitai_cache/` by default. You can move the cache anywhere by setting **`path_civitai_cache`** in `config.txt` — useful when you have hundreds of models and want the cache on a faster/larger drive:
```json
{
  "path_civitai_cache": "D:/Caches/civitai"
}
```

---

### 6. 📐 Use Aspect Ratio for Vary
**Where:** Advanced tab → Aspect Ratios accordion (below the aspect-ratio grid).

**What it does:** Forces Vary (Subtle) and Vary (Strong) outputs to use the selected Aspect Ratios dimensions instead of the input image's native shape. The input image is centre-cropped and resized to fit (not stretched). Upstream Fooocus always uses the input image's aspect for Vary/Upscale — this checkbox is the toggle for when you want to re-frame.

**How to use:**
1. Enable **☑ Advanced** mode.
2. Expand the **Aspect Ratios** accordion.
3. Pick the output aspect you want (e.g., `1152×896`).
4. Tick **Use selected Aspect Ratio for Vary (crop input to fit)**.
5. Load a source image into the Vary tab, pick Vary (Subtle) or Vary (Strong), Generate.

Unchecked → original upstream behaviour (preserves input's native aspect). Does **not** affect Upscale (which keeps its fixed 1.5x / 2x factor).

---

### 7. 📏 Aspect Ratios dropdown + Custom Resolution
**Where:** Advanced tab → Aspect Ratios accordion.

**What changed (UI):** the Aspect Ratios picker is now a **dropdown list** (instead of the upstream radio grid), with **`Custom`** as the first entry. Picking any preset works exactly like before; picking `Custom` reveals the resolution panel below.

**What Custom does:** Pick **any** aspect ratio + size on the fly without editing `config.txt` or restarting. Result is always snapped to multiples of 64 (the SDXL hard requirement). Useful for one-off ratios that aren't worth permanently adding to the dropdown — 16:9 banners, 4:5 IG portraits, 21:9 cinema, A4-ratio prints, etc.

**How to use:**
1. Expand the **Aspect Ratios** accordion.
2. Open the dropdown and pick **`Custom`** — the resolution panel unfolds below.
3. Either type a ratio in the **Ratio W** / **Ratio H** fields, or click one of the **quick chips** (3-column × 2-row grid):

   | | | |
   |---|---|---|
   | `1:1` | `3:2` | `4:3` |
   | `16:9` | `21:9` | `√2 (A4)` |

   Click **🔄 Swap** to flip W ↔ H (landscape ↔ portrait).
4. Pick a **Mode** — controls what the size slider means:
   - **Max edge** (default) — the longer side equals the slider value.
   - **~1 MP target** — total pixels stay near `size²` regardless of ratio (matches SDXL's training sweet-spot).
   - **Min edge** — the shorter side equals the slider value.
5. Adjust the **Size** slider (512–2048, step 64).
6. Read the live result line: `→ 1344 × 768 · 1.03 MP · 7:4`. A warning appears (non-blocking) if the result drops below 0.25 MP or exceeds 2 MP.
7. **Generate** — Vary, Upscale, Inpaint and the standard text-to-image path all pick up the override transparently.

**💾 Save as preset entry** — one-click button that appends the computed `W*H` to `available_aspect_ratios` in `config.txt`. After the next restart the resolution shows up directly in the Aspect Ratios dropdown, so you don't have to re-dial it every session.

**Preset round-trip:** `save_preset_to_file` writes a `custom_resolution: {enabled, ratio_w, ratio_h, mode, size}` block into preset JSONs. Old presets without the block default to OFF (full back-compat). The block is invisible to vanilla Fooocus, so cross-loading a Fooocus2025 preset on upstream Fooocus is safe.

**Compose with custom-6:** select **`Custom`** in the Aspect Ratios dropdown AND tick **Use Aspect Ratio for Vary** to force Vary outputs to your custom W × H — handy for re-framing source images to arbitrary print/social formats.

---

### 8. 🖼️ Asset Browser (autonomous gallery)
**Where:** Advanced tab → **🖼️ Asset Browser** accordion (master toggle is **OFF by default**). Once enabled, a **🖼️ Asset Browser** link appears next to **📚 History Log** in the prompt area; clicking it opens the gallery in a new browser tab.

**What it does:** A standalone HTML/JSON gallery served from `outputs/index.html` — built on **PhotoSwipe v5** + **Dynamic Caption plugin** + **Deep Zoom plugin** (all MIT, vanilla JS, no React/Vue, ~50 KB gzip total). Browses **outputs** and **model previews** (LoRAs / Checkpoints / Embeddings) in the same UI, with click-to-zoom lightbox + per-type metadata sidebar + clipboard copy buttons.

**Why it exists separately:** Designed as an **autonomous module** — Fooocus only knows it exists via 2 link buttons in the UI. The SPA is plain `outputs/index.html` reading JSON manifests, so it works even if Fooocus is not running. **Master toggle in `config.txt` (OFF by default)** keeps zero overhead on Fooocus when disabled (<1µs hook check). User-facing accordion in Advanced lets you enable + tune sub-features.

**4 tabs:**
- **📅 Outputs** — left timeline (one entry per generation day, "today" highlighted), grid of thumbnails for the selected day. Click → PhotoSwipe lightbox with metadata sidebar (prompt, negative, sampler, CFG, steps, seed, resolution, model, LoRAs…) + 📋 Copy Prompt / Copy Negative / Copy All Params (JSON) buttons.
- **🎨 LoRAs** — left subfolder facets, grid of preview images (auto-discovered sidecar `<model>.preview.png` etc., or hash-derived placeholder gradient if missing). Lightbox shows triggers + size + 📋 Copy triggers / 🔗 Open on CivitAI link.
- **📦 Models (Checkpoints)** — same UX + base model + CivitAI consensus settings (sampler / CFG / steps / clip skip) read from the cache populated by the existing **🎨 CivitAI Model Settings** panel. Lightbox has 📋 Copy consensus (JSON) for re-pasting elsewhere.
- **🧩 Embeddings** — same UX + auto-detected `(embedding:name:1.0)` token + negative-prompt heuristic flag. Lightbox has 📋 Copy embedding token / Copy trigger only.

**How to enable:**
1. **Advanced → 🖼️ Asset Browser** → tick **Enable Asset Browser** → click **💾 Save settings**.
2. Restart Fooocus (Restart UI button works fine).
3. Click **🔄 Reindex everything now** to backfill thumbnails + manifests for your existing outputs and your installed models. Console shows progress.
4. Click the new **🖼️ Asset Browser** link in the prompt area → opens in a new tab.

**UI sub-toggles** (all default ON when master is enabled, exposed in the Advanced accordion):
- *Generate thumbnails on save* — JPEG centre-crop, ~10 ms/image, makes the grid load instantly even with thousands of images.
- *Generate deep-zoom tiles for big images* — `auto` (>4 MP by default), `always`, `never`. Currently a no-op in v1 (the Deep Zoom plugin falls back to PhotoSwipe's native pinch/scroll zoom on the full image — fine for everything except gigapixel scans).
- *Index models on startup* — daemon thread, ~2-5 s for hundreds of LoRAs (uses cache, never makes fresh CivitAI API calls in bulk).
- *🌫 Blur thumbnails by default (NSFW privacy)* — when ON, all thumbnails are CSS-blurred with hover/click reveal. The Browser header has a per-browser override toggle (sticky in localStorage).

**Browser-side features (no Fooocus involvement):**
- **Hide / show days + subfolders** — hover over any item in the sidebar (a day in 📅 Outputs, a subfolder in 🎨 LoRAs / 📦 Models / 🧩 Embeddings) → small `🚫 hide` button appears → click to remove from the list. Header shows `👁️ Hidden N` badge with the count; clicking the badge toggles "show hidden" mode where they reappear greyed with `↩ unhide`. Persisted per-browser via localStorage. Useful to hide a Flux experiments subfolder, an NSFW day, or duplicate organisational folders.
- **🌐 Fetch from CivitAI** button — appears in the lightbox caption when a model has no sidecar preview (placeholder). One click downloads the top-rated CivitAI image for that model and saves it as `<stem>.preview.png` next to the model file (A1111/ComfyUI sidecar convention — visible in those tools too). Manifest is rebuilt automatically; the page reloads to show the new preview. Refuses to overwrite an existing sidecar (data-loss avoidance — user must delete first).
- **Async reindex with live progress** — clicking 🔄 Reindex everything now spawns a daemon thread; the button returns immediately with `Reindex started in background`, then the status line updates every 2 s with the current phase (`🔄 Outputs: [42/135] · current: 2024-06-08 · 8124 image(s) processed so far`, then `Now scanning models…`, then `✓ Reindex complete: 135 day(s), 24831 image(s). · models: 5234 LoRAs, 153 ckpts, 603 embeds`). No more browser HTTP timeouts on large collections.
- **Console silenced** — Pillow's DecompressionBombWarning (triggered by Fooocus's own Upscale 2x outputs >89 MP) and the Windows asyncio Win10054 connection-reset noise are both filtered out at startup. Real errors still bubble up.

**Advanced tunables (`config.txt` only — not exposed in the UI to avoid clutter):**

```json
{
  "asset_browser": {
    "thumbnail_size": 256,         // 64..1024 px square. Try 128 to ~quarter disk usage + speed up grid load.
    "thumbnail_quality": 85,       // 40..100 JPEG quality. 70 ~halves file size at minor visual cost.
    "dzi_threshold_mp": 4.0,       // 0.5..64.0 MP — when generate_dzi_tiles='auto' kicks in (DZI gen still deferred in v1).
    "placeholder_label_max": 24    // 8..64 chars before truncating the filename overlay on placeholder previews.
  }
}
```

Each value is clamped on save — bad values in `config.txt` fall back to the default rather than crashing.

**Browser navigation** — URL hash for tab state: `outputs/index.html#loras` / `#checkpoints` / `#embeddings` are all bookmarkable.

**Generated artifacts** — everything the Asset Browser writes lives under your `path_outputs` (configurable in `config.txt`):
- `outputs/index.html` + `outputs/_assets/` — the SPA + bundled PhotoSwipe.
- `outputs/_index/{days,loras,checkpoints,embeddings}.json` — the manifests the SPA reads.
- `outputs/_previews/<kind>/<hash>.jpg` — model preview thumbnails (sidecars + placeholders).
- `outputs/<DATE>/<image>_thumb.jpg` — output thumbnails next to each image.

All are gitignored under `outputs/` and can be wiped at any time — the next Reindex rebuilds them.

**Future v2 (not in this release):** open the SPA as an **iframe modal inside Fooocus** when the user clicks a LoRA/Checkpoint dropdown — picks a model visually instead of by filename. Already feasible without changes to the SPA itself (just a JS bridge in Fooocus).

---

### 9. 🔄 Restart UI button
**Where:** bottom of the Advanced tab, next to **Refresh All Files**.

**What it does:** Exits the Python process with code `42`. The included `.bat` launchers detect that exit code and relaunch automatically — a real restart (re-reads `config.txt`, re-imports modules, re-loads the model). Takes ~30 s on an RTX 5090; refresh the browser tab once the Gradio server is back.

**Setup for your launcher:** wrap your launch command in a `:fooocus_start` loop that checks `%ERRORLEVEL% EQU 42`:

```bat
:fooocus_start
.\python_embeded\Scripts\python.exe -s Fooocus\entry_with_update.py
if %ERRORLEVEL% EQU 42 goto fooocus_start
pause
```

Without this loop, the restart button still works — it just becomes a clean exit instead of a restart.

---

### 10. 🧬 Architecture filtering (SD/SDXL only)
**Where:** Automatic — no UI toggle needed.

**What it does:** Reads safetensors headers at startup (~1 ms per file, no weight loading) and hides incompatible models from **all three dropdowns** (checkpoints, LoRAs, embeddings). Only SD 1.x and SDXL models appear. Flux, SD3, SVD, LLMs, and any other architecture are silently hidden. Console shows a one-line summary: `[arch-filter] Kept 42, hidden 3 incompatible checkpoint(s): flux-dev.safetensors, sd3-medium.safetensors, ...`.

**Detection strategies:**
- **Checkpoints:** positive-match on `model.diffusion_model.input_blocks.*` (SD/SDXL signature). Everything else is hidden.
- **LoRAs:** blacklist approach — only blocks Flux LoRAs (`lora_transformer_double*`), SD3 LoRAs (`lora_transformer_joint*`), and full models accidentally placed in the LoRA folder. Keeps kohya, diffusers, LyCORIS, LoHa, LoKr, Pony, and everything else.
- **Embeddings:** files under 10 MB are always kept (covers all real embeddings). Larger files are checked for full-model signatures and hidden if found.

**Cache:** results are stored in `model_arch_cache.json` (gitignored) keyed by filename + mtime. Delete the file to force a full rescan.

---

### 11. 🖼️ Asset Browser shortcut icons
**Where:** Next to **Base Model**, **Refiner**, **LoRA**, and **Textual Inversion** labels in the Advanced tab.

**What it does:** Small clickable icons that open the corresponding Asset Browser tab in a new browser tab — `#checkpoints` for Base Model / Refiner, `#loras` for LoRA, `#embeddings` for Textual Inversion. Saves you from hunting for the Asset Browser link when you're already looking at the model selector.

**Requires:** Asset Browser enabled in `config.txt`.

---

## 🚀 Getting this fork

### Option A — I already have Fooocus installed
```bash
git clone https://github.com/mikecastrodemaria/Fooocus2025.git
# or, inside an existing clone:
git remote add fork2025 https://github.com/mikecastrodemaria/Fooocus2025.git
git fetch fork2025 && git checkout -b fork2025-main fork2025/main
```

### Option B — Fresh install on Windows
1. Download the original Windows package: **[Fooocus_win64_2-5-0.7z](https://github.com/lllyasviel/Fooocus/releases/download/v2.5.0/Fooocus_win64_2-5-0.7z)** (≈1.7 GB).
2. Extract it.
3. Inside the extracted `Fooocus/` subfolder, replace its contents with a clone of this fork:
   ```bash
   cd Fooocus_win64_2-5-0/Fooocus
   git init && git remote add origin https://github.com/mikecastrodemaria/Fooocus2025.git
   git fetch origin && git reset --hard origin/main
   ```
4. Run `run.bat` from the install root as usual.

### Hardware note (RTX 5090 users)
The install root of the original package can include launcher scripts tuned for an **NVIDIA RTX 5090** (`boot_check_rtx5090.bat`, `run_quality_rtx5090.bat`). These are *not* part of this git repo — use the stock `run.bat` / `run_realistic.bat` / `run_anime.bat` for any other hardware.

---

## 🔒 Security note — local-use deployment

This fork is designed for **local single-user use** on a workstation, behind your
own firewall. The Gradio 3.41.2 server it ships is pinned for compatibility
(every JS/DOM injection patch in this fork — Asset Browser, custom tabs, header
icons — assumes Gradio 3.x). Several known CVEs in Gradio 3.41.x require a
remote attacker who can reach the web server: they are **not exploitable in the
intended deployment** (localhost binding, no public exposure).

If you decide to expose Fooocus on a network — even a "trusted" LAN — observe:

- **Never use `--share`** on an untrusted network. The `--share` flag opens a
  public Gradio tunnel and re-introduces the full attack surface of the CVEs.
- **Do not bind Fooocus to a public interface.** Stay on `127.0.0.1`.
- If you must run on a LAN, put it behind an authenticated reverse proxy
  (Caddy / Nginx + basic auth) — Gradio 3.x's own `--auth` is the bare minimum
  and not a substitute for a proper proxy.
- Don't load `.safetensors` or `.pth` checkpoints from untrusted sources.
  Several CVEs in `transformers` / `torch` are deserialization issues triggered
  by loading malicious model files; the only realistic mitigation is the same
  rule as for any ML stack — only run models you trust.

The other deps (`transformers`, `torch`, `pytorch_lightning`) have been
upgraded to the latest fixed versions in custom-11. `pytorch_lightning` was
dropped entirely since the only reference in the codebase was a commented-out
import in `ldm_patched/ldm/models/autoencoder.py`.

---

## ⚙️ Fork-specific `config.txt` keys

All upstream keys still apply. The fork adds a few of its own. Most have a UI control (Advanced tab) but the ones below are useful enough to be tunable directly in `config.txt`:

| Key | Default | Range / Type | Used by | Purpose |
|---|---|---|---|---|
| `civitai_api_key` | `""` | string | custom-2 | API key persisted from the CivitAI panel. |
| `path_civitai_cache` | `"./civitai_cache"` | path string | custom-2 (custom-8 made it configurable) | CivitAI response cache directory. Move to a different drive if you have hundreds of cached models. |
| `asset_browser.enabled` | `false` | bool | custom-8 | Master toggle. **OFF by default** — when off the per-image hook returns in <1 µs, the indexer thread is never spawned. |
| `asset_browser.generate_thumbnails` | `true` | bool | custom-8 | Generate `*_thumb.jpg` next to each output (~10 ms/image). |
| `asset_browser.generate_dzi_tiles` | `"auto"` | `"auto"` / `"always"` / `"never"` | custom-8 | DZI tile generation mode (`auto` = above `dzi_threshold_mp`). DZI generation itself is still deferred in v1. |
| `asset_browser.index_models_on_boot` | `true` | bool | custom-8 | Daemon thread on startup, scans LoRAs / Checkpoints / Embeddings (~2-5 s, cached). |
| `asset_browser.thumbnail_size` | `256` | 64..1024 | custom-8 | Square thumbnail size in pixels. 128 cuts disk usage / load time ~75 %. |
| `asset_browser.thumbnail_quality` | `85` | 40..100 | custom-8 | JPEG quality for thumbnails. 70 ~halves file size. |
| `asset_browser.dzi_threshold_mp` | `4.0` | 0.5..64.0 | custom-8 | Megapixel threshold for `generate_dzi_tiles="auto"`. |
| `asset_browser.placeholder_label_max` | `24` | 8..64 | custom-8 | Filename length on placeholder previews before truncation. |
| `asset_browser.blur_thumbnails` | `false` | bool | custom-8.1 | NSFW privacy default — blur all thumbnails with hover/click reveal. Per-browser override available in the Browser header. |
| `path_face_restore_models` | `"../models/face_restore_models/"` | path string | custom-10 | Default folder for CodeFormer / GFPGAN models (auto-downloaded here on first use). |
| `path_esrgan` | `""` | path string (optional) | custom-10 | A1111-compatible: extra folder scanned for ESRGAN-family models. Empty = disabled. |
| `path_realesrgan` | `""` | path string (optional) | custom-10 | A1111-compatible: extra folder for RealESRGAN models (SRVGG arch). |
| `path_swinir` | `""` | path string (optional) | custom-10 | A1111-compatible: extra folder for SwinIR / Swin2SR models. |
| `path_dat` | `""` | path string (optional) | custom-10 | A1111-compatible: extra folder for DAT models. |
| `path_gfpgan` | `""` | path string (optional) | custom-10 | A1111-compatible: extra folder for GFPGAN models. |
| `path_codeformer` | `""` | path string (optional) | custom-10 | A1111-compatible: extra folder for CodeFormer models. |

Each value is clamped on save — bad values in `config.txt` fall back to the default rather than crashing.

Example fragment:
```json
{
  "path_civitai_cache": "D:/Caches/civitai",
  "path_esrgan": "D:/stable-diffusion-webui/models/ESRGAN",
  "path_realesrgan": "D:/stable-diffusion-webui/models/RealESRGAN",
  "path_gfpgan": "D:/stable-diffusion-webui/models/GFPGAN",
  "asset_browser": {
    "enabled": true,
    "thumbnail_size": 128,
    "thumbnail_quality": 70
  }
}
```

---

## 📁 Files touched by this fork
| File | Purpose |
|---|---|
| `fooocus_version.py` | Version bumped to `2026.2.0` (CalVer) |
| `modules/civitai_api.py` | **New** — CivitAI client, caching, consensus aggregation, model+embedding triggers |
| `modules/lora_metadata.py` | **New** — local safetensors metadata reader for LoRA/embedding triggers |
| `modules/util.py` | Adds `compute_custom_wh()` — ratio + size → snapped W×H (custom-7) |
| `modules/config.py` | Save Preset / preset round-trip (LoRAs, embeddings, custom resolution) + API key persistence + `asset_browser` config block (custom-8) |
| `modules/async_worker.py` | Reads `use_aspect_for_vary` (custom-6) and `custom_resolution` (custom-7) flags |
| `modules/private_logger.py` | Silent hook into `gallery_writer.on_image_logged()` (custom-8) |
| `modules/gallery_writer.py` | **New** — Asset Browser per-image hook, thumbnails, manifests, days.json (custom-8) |
| `modules/model_indexer.py` | **New** — Asset Browser model scanners (LoRAs / Checkpoints / Embeddings) + sidecar preview lookup + placeholder generation (custom-8) |
| `gallery_template/index.html` + `_assets/` | **New** — Asset Browser SPA + bundled PhotoSwipe v5 / Dynamic Caption / Deep Zoom (custom-8) |
| `launch.py` | Spawns Asset Browser model indexer in a daemon thread when enabled (custom-8) |
| `webui.py` | All fork UI: Save Preset, CivitAI / LoRA / Embeddings / Wildcards accordions, Aspect-for-Vary, Custom Resolution panel, Asset Browser accordion + link button, Restart UI |
| `CHANGELOG.md` | Per-release fork history |
| `.gitignore` | Excludes `civitai_cache/`, local presets, assistant artifacts |

See [`CHANGELOG.md`](CHANGELOG.md) for versioned history.

---
---

# 📖 Original Fooocus README

*Everything below this point is the unmodified upstream readme from [lllyasviel/Fooocus](https://github.com/lllyasviel/Fooocus) — installation, models, features table, troubleshooting, etc.*

[>>> Click Here to Install Fooocus <<<](#download)

Fooocus is an image generating software (based on [Gradio](https://www.gradio.app/) <a href='https://github.com/gradio-app/gradio'><img src='https://img.shields.io/github/stars/gradio-app/gradio'></a>).

Fooocus presents a rethinking of image generator designs. The software is offline, open source, and free, while at the same time, similar to many online image generators like Midjourney, the manual tweaking is not needed, and users only need to focus on the prompts and images. Fooocus has also simplified the installation: between pressing "download" and generating the first image, the number of needed mouse clicks is strictly limited to less than 3. Minimal GPU memory requirement is 4GB (Nvidia).

**Recently many fake websites exist on Google when you search “fooocus”. Do not trust those – here is the only official source of Fooocus.**

# Project Status: Limited Long-Term Support (LTS) with Bug Fixes Only

The Fooocus project, built entirely on the **Stable Diffusion XL** architecture, is now in a state of limited long-term support (LTS) with bug fixes only. As the existing functionalities are considered as nearly free of programmartic issues (Thanks to [mashb1t](https://github.com/mashb1t)'s huge efforts), future updates will focus exclusively on addressing any bugs that may arise. 

**There are no current plans to migrate to or incorporate newer model architectures.** However, this may change during time with the development of open-source community. For example, if the community converge to one single dominant method for image generation (which may really happen in half or one years given the current status), Fooocus may also migrate to that exact method.

For those interested in utilizing newer models such as **Flux**, we recommend exploring alternative platforms such as [WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) (also from us), [ComfyUI/SwarmUI](https://github.com/comfyanonymous/ComfyUI). Additionally, several [excellent forks of Fooocus](https://github.com/lllyasviel/Fooocus?tab=readme-ov-file#forks) are available for experimentation.

Again, recently many fake websites exist on Google when you search “fooocus”. Do **NOT** get Fooocus from those websites – this page is the only official source of Fooocus. We never have any website like such as “fooocus.com”, “fooocus.net”, “fooocus.co”, “fooocus.ai”, “fooocus.org”, “fooocus.pro”, “fooocus.one”. Those websites are ALL FAKE. **They have ABSOLUTLY no relationship to us. Fooocus is a 100% non-commercial offline open-source software.**

# Features

Below is a quick list using Midjourney's examples:

| Midjourney | Fooocus |
| - | - |
| High-quality text-to-image without needing much prompt engineering or parameter tuning. <br> (Unknown method) | High-quality text-to-image without needing much prompt engineering or parameter tuning. <br> (Fooocus has an offline GPT-2 based prompt processing engine and lots of sampling improvements so that results are always beautiful, no matter if your prompt is as short as “house in garden” or as long as 1000 words) |
| V1 V2 V3 V4 | Input Image -> Upscale or Variation -> Vary (Subtle) / Vary (Strong)|
| U1 U2 U3 U4 | Input Image -> Upscale or Variation -> Upscale (1.5x) / Upscale (2x) |
| Inpaint / Up / Down / Left / Right (Pan) | Input Image -> Inpaint or Outpaint -> Inpaint / Up / Down / Left / Right <br> (Fooocus uses its own inpaint algorithm and inpaint models so that results are more satisfying than all other software that uses standard SDXL inpaint method/model) |
| Image Prompt | Input Image -> Image Prompt <br> (Fooocus uses its own image prompt algorithm so that result quality and prompt understanding are more satisfying than all other software that uses standard SDXL methods like standard IP-Adapters or Revisions) |
| --style | Advanced -> Style |
| --stylize | Advanced -> Advanced -> Guidance |
| --niji | [Multiple launchers: "run.bat", "run_anime.bat", and "run_realistic.bat".](https://github.com/lllyasviel/Fooocus/discussions/679) <br> Fooocus support SDXL models on Civitai <br> (You can google search “Civitai” if you do not know about it) |
| --quality | Advanced -> Quality |
| --repeat | Advanced -> Image Number |
| Multi Prompts (::) | Just use multiple lines of prompts |
| Prompt Weights | You can use " I am (happy:1.5)". <br> Fooocus uses A1111's reweighting algorithm so that results are better than ComfyUI if users directly copy prompts from Civitai. (Because if prompts are written in ComfyUI's reweighting, users are less likely to copy prompt texts as they prefer dragging files) <br> To use embedding, you can use "(embedding:file_name:1.1)" |
| --no | Advanced -> Negative Prompt |
| --ar | Advanced -> Aspect Ratios |
| InsightFace | Input Image -> Image Prompt -> Advanced -> FaceSwap |
| Describe | Input Image -> Describe |

Below is a quick list using LeonardoAI's examples:

| LeonardoAI | Fooocus |
| - | - |
| Prompt Magic | Advanced -> Style -> Fooocus V2 |
| Advanced Sampler Parameters (like Contrast/Sharpness/etc) | Advanced -> Advanced -> Sampling Sharpness / etc |
| User-friendly ControlNets | Input Image -> Image Prompt -> Advanced |

Also, [click here to browse the advanced features.](https://github.com/lllyasviel/Fooocus/discussions/117)

# Download

### Windows

You can directly download Fooocus with:

**[>>> Click here to download <<<](https://github.com/lllyasviel/Fooocus/releases/download/v2.5.0/Fooocus_win64_2-5-0.7z)**

After you download the file, please uncompress it and then run the "run.bat".

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/c49269c4-c274-4893-b368-047c401cc58c)

The first time you launch the software, it will automatically download models:

1. It will download [default models](#models) to the folder "Fooocus\models\checkpoints" given different presets. You can download them in advance if you do not want automatic download.
2. Note that if you use inpaint, at the first time you inpaint an image, it will download [Fooocus's own inpaint control model from here](https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch) as the file "Fooocus\models\inpaint\inpaint_v26.fooocus.patch" (the size of this file is 1.28GB).

After Fooocus 2.1.60, you will also have `run_anime.bat` and `run_realistic.bat`. They are different model presets (and require different models, but they will be automatically downloaded). [Check here for more details](https://github.com/lllyasviel/Fooocus/discussions/679).

After Fooocus 2.3.0 you can also switch presets directly in the browser. Keep in mind to add these arguments if you want to change the default behavior:
* Use `--disable-preset-selection` to disable preset selection in the browser.
* Use `--always-download-new-model` to download missing models on preset switch. Default is fallback to `previous_default_models` defined in the corresponding preset, also see terminal output.

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/d386f817-4bd7-490c-ad89-c1e228c23447)

If you already have these files, you can copy them to the above locations to speed up installation.

Note that if you see **"MetadataIncompleteBuffer" or "PytorchStreamReader"**, then your model files are corrupted. Please download models again.

Below is a test on a relatively low-end laptop with **16GB System RAM** and **6GB VRAM** (Nvidia 3060 laptop). The speed on this machine is about 1.35 seconds per iteration. Pretty impressive – nowadays laptops with 3060 are usually at very acceptable price.

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/938737a5-b105-4f19-b051-81356cb7c495)

Besides, recently many other software report that Nvidia driver above 532 is sometimes 10x slower than Nvidia driver 531. If your generation time is very long, consider download [Nvidia Driver 531 Laptop](https://www.nvidia.com/download/driverResults.aspx/199991/en-us/) or [Nvidia Driver 531 Desktop](https://www.nvidia.com/download/driverResults.aspx/199990/en-us/).

Note that the minimal requirement is **4GB Nvidia GPU memory (4GB VRAM)** and **8GB system memory (8GB RAM)**. This requires using Microsoft’s Virtual Swap technique, which is automatically enabled by your Windows installation in most cases, so you often do not need to do anything about it. However, if you are not sure, or if you manually turned it off (would anyone really do that?), or **if you see any "RuntimeError: CPUAllocator"**, you can enable it here:

<details>
<summary>Click here to see the image instructions. </summary>

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/2a06b130-fe9b-4504-94f1-2763be4476e9)

**And make sure that you have at least 40GB free space on each drive if you still see "RuntimeError: CPUAllocator" !**

</details>

Please open an issue if you use similar devices but still cannot achieve acceptable performances.

Note that the [minimal requirement](#minimal-requirement) for different platforms is different.

See also the common problems and troubleshoots [here](troubleshoot.md).

### Colab

(Last tested - 2024 Aug 12 by [mashb1t](https://github.com/mashb1t))

| Colab | Info
| --- | --- |
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lllyasviel/Fooocus/blob/main/fooocus_colab.ipynb) | Fooocus Official

In Colab, you can modify the last line to `!python entry_with_update.py --share --always-high-vram` or `!python entry_with_update.py --share --always-high-vram --preset anime` or `!python entry_with_update.py --share --always-high-vram --preset realistic` for Fooocus Default/Anime/Realistic Edition.

You can also change the preset in the UI. Please be aware that this may lead to timeouts after 60 seconds. If this is the case, please wait until the download has finished, change the preset to initial and back to the one you've selected or reload the page.

Note that this Colab will disable refiner by default because Colab free's resources are relatively limited (and some "big" features like image prompt may cause free-tier Colab to disconnect). We make sure that basic text-to-image is always working on free-tier Colab.

Using `--always-high-vram` shifts resource allocation from RAM to VRAM and achieves the overall best balance between performance, flexibility and stability on the default T4 instance. Please find more information [here](https://github.com/lllyasviel/Fooocus/pull/1710#issuecomment-1989185346).

Thanks to [camenduru](https://github.com/camenduru) for the template!

### Linux (Using Anaconda)

If you want to use Anaconda/Miniconda, you can

    git clone https://github.com/lllyasviel/Fooocus.git
    cd Fooocus
    conda env create -f environment.yaml
    conda activate fooocus
    pip install -r requirements_versions.txt

Then download the models: download [default models](#models) to the folder "Fooocus\models\checkpoints". **Or let Fooocus automatically download the models** using the launcher:

    conda activate fooocus
    python entry_with_update.py

Or, if you want to open a remote port, use

    conda activate fooocus
    python entry_with_update.py --listen

Use `python entry_with_update.py --preset anime` or `python entry_with_update.py --preset realistic` for Fooocus Anime/Realistic Edition.

### Linux (Using Python Venv)

Your Linux needs to have **Python 3.10** installed, and let's say your Python can be called with the command **python3** with your venv system working; you can

    git clone https://github.com/lllyasviel/Fooocus.git
    cd Fooocus
    python3 -m venv fooocus_env
    source fooocus_env/bin/activate
    pip install -r requirements_versions.txt

See the above sections for model downloads. You can launch the software with:

    source fooocus_env/bin/activate
    python entry_with_update.py

Or, if you want to open a remote port, use

    source fooocus_env/bin/activate
    python entry_with_update.py --listen

Use `python entry_with_update.py --preset anime` or `python entry_with_update.py --preset realistic` for Fooocus Anime/Realistic Edition.

### Linux (Using native system Python)

If you know what you are doing, and your Linux already has **Python 3.10** installed, and your Python can be called with the command **python3** (and Pip with **pip3**), you can

    git clone https://github.com/lllyasviel/Fooocus.git
    cd Fooocus
    pip3 install -r requirements_versions.txt

See the above sections for model downloads. You can launch the software with:

    python3 entry_with_update.py

Or, if you want to open a remote port, use

    python3 entry_with_update.py --listen

Use `python entry_with_update.py --preset anime` or `python entry_with_update.py --preset realistic` for Fooocus Anime/Realistic Edition.

### Linux (AMD GPUs)

Note that the [minimal requirement](#minimal-requirement) for different platforms is different.

Same with the above instructions. You need to change torch to the AMD version

    pip uninstall torch torchvision torchaudio torchtext functorch xformers 
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

AMD is not intensively tested, however. The AMD support is in beta.

Use `python entry_with_update.py --preset anime` or `python entry_with_update.py --preset realistic` for Fooocus Anime/Realistic Edition.

### Windows (AMD GPUs)

Note that the [minimal requirement](#minimal-requirement) for different platforms is different.

Same with Windows. Download the software and edit the content of `run.bat` as:

    .\python_embeded\python.exe -m pip uninstall torch torchvision torchaudio torchtext functorch xformers -y
    .\python_embeded\python.exe -m pip install torch-directml
    .\python_embeded\python.exe -s Fooocus\entry_with_update.py --directml
    pause

Then run the `run.bat`.

AMD is not intensively tested, however. The AMD support is in beta.

For AMD, use `.\python_embeded\python.exe Fooocus\entry_with_update.py --directml --preset anime` or `.\python_embeded\python.exe Fooocus\entry_with_update.py --directml --preset realistic` for Fooocus Anime/Realistic Edition.

### Mac

Note that the [minimal requirement](#minimal-requirement) for different platforms is different.

Mac is not intensively tested. Below is an unofficial guideline for using Mac. You can discuss problems [here](https://github.com/lllyasviel/Fooocus/pull/129).

You can install Fooocus on Apple Mac silicon (M1 or M2) with macOS 'Catalina' or a newer version. Fooocus runs on Apple silicon computers via [PyTorch](https://pytorch.org/get-started/locally/) MPS device acceleration. Mac Silicon computers don't come with a dedicated graphics card, resulting in significantly longer image processing times compared to computers with dedicated graphics cards.

1. Install the conda package manager and pytorch nightly. Read the [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/) Apple Developer guide for instructions. Make sure pytorch recognizes your MPS device.
1. Open the macOS Terminal app and clone this repository with `git clone https://github.com/lllyasviel/Fooocus.git`.
1. Change to the new Fooocus directory, `cd Fooocus`.
1. Create a new conda environment, `conda env create -f environment.yaml`.
1. Activate your new conda environment, `conda activate fooocus`.
1. Install the packages required by Fooocus, `pip install -r requirements_versions.txt`.
1. Launch Fooocus by running `python entry_with_update.py`. (Some Mac M2 users may need `python entry_with_update.py --disable-offload-from-vram` to speed up model loading/unloading.) The first time you run Fooocus, it will automatically download the Stable Diffusion SDXL models and will take a significant amount of time, depending on your internet connection.

Use `python entry_with_update.py --preset anime` or `python entry_with_update.py --preset realistic` for Fooocus Anime/Realistic Edition.

### Docker

See [docker.md](docker.md)

### Download Previous Version

See the guidelines [here](https://github.com/lllyasviel/Fooocus/discussions/1405).

## Minimal Requirement

Below is the minimal requirement for running Fooocus locally. If your device capability is lower than this spec, you may not be able to use Fooocus locally. (Please let us know, in any case, if your device capability is lower but Fooocus still works.)

| Operating System  | GPU                          | Minimal GPU Memory           | Minimal System Memory     | [System Swap](troubleshoot.md) | Note                                                                       |
|-------------------|------------------------------|------------------------------|---------------------------|--------------------------------|----------------------------------------------------------------------------|
| Windows/Linux     | Nvidia RTX 4XXX              | 4GB                          | 8GB                       | Required                       | fastest                                                                    |
| Windows/Linux     | Nvidia RTX 3XXX              | 4GB                          | 8GB                       | Required                       | usually faster than RTX 2XXX                                               |
| Windows/Linux     | Nvidia RTX 2XXX              | 4GB                          | 8GB                       | Required                       | usually faster than GTX 1XXX                                               |
| Windows/Linux     | Nvidia GTX 1XXX              | 8GB (&ast; 6GB uncertain)    | 8GB                       | Required                       | only marginally faster than CPU                                            |
| Windows/Linux     | Nvidia GTX 9XX               | 8GB                          | 8GB                       | Required                       | faster or slower than CPU                                                  |
| Windows/Linux     | Nvidia GTX < 9XX             | Not supported                | /                         | /                              | /                                                                          |
| Windows           | AMD GPU                      | 8GB    (updated 2023 Dec 30) | 8GB                       | Required                       | via DirectML (&ast; ROCm is on hold), about 3x slower than Nvidia RTX 3XXX |
| Linux             | AMD GPU                      | 8GB                          | 8GB                       | Required                       | via ROCm, about 1.5x slower than Nvidia RTX 3XXX                           |
| Mac               | M1/M2 MPS                    | Shared                       | Shared                    | Shared                         | about 9x slower than Nvidia RTX 3XXX                                       |
| Windows/Linux/Mac | only use CPU                 | 0GB                          | 32GB                      | Required                       | about 17x slower than Nvidia RTX 3XXX                                      |

&ast; AMD GPU ROCm (on hold): The AMD is still working on supporting ROCm on Windows.

&ast; Nvidia GTX 1XXX 6GB uncertain: Some people report 6GB success on GTX 10XX, but some other people report failure cases.

*Note that Fooocus is only for extremely high quality image generating. We will not support smaller models to reduce the requirement and sacrifice result quality.*

## Troubleshoot

See the common problems [here](troubleshoot.md).

## Default Models
<a name="models"></a>

Given different goals, the default models and configs of Fooocus are different:

| Task      | Windows | Linux args | Main Model                  | Refiner | Config                                                                         |
|-----------| --- | --- |-----------------------------| --- |--------------------------------------------------------------------------------|
| General   | run.bat |  | juggernautXL_v8Rundiffusion | not used | [here](https://github.com/lllyasviel/Fooocus/blob/main/presets/default.json)   |
| Realistic | run_realistic.bat | --preset realistic | realisticStockPhoto_v20     | not used | [here](https://github.com/lllyasviel/Fooocus/blob/main/presets/realistic.json) |
| Anime     | run_anime.bat | --preset anime | animaPencilXL_v500          | not used | [here](https://github.com/lllyasviel/Fooocus/blob/main/presets/anime.json)     |

Note that the download is **automatic** - you do not need to do anything if the internet connection is okay. However, you can download them manually if you (or move them from somewhere else) have your own preparation.

## UI Access and Authentication
In addition to running on localhost, Fooocus can also expose its UI in two ways: 
* Local UI listener: use `--listen` (specify port e.g. with `--port 8888`). 
* API access: use `--share` (registers an endpoint at `.gradio.live`).

In both ways the access is unauthenticated by default. You can add basic authentication by creating a file called `auth.json` in the main directory, which contains a list of JSON objects with the keys `user` and `pass` (see example in [auth-example.json](./auth-example.json)).

## List of "Hidden" Tricks
<a name="tech_list"></a>

<details>
<summary>Click to see a list of tricks. Those are based on SDXL and are not very up-to-date with latest models.</summary>

1. GPT2-based [prompt expansion as a dynamic style "Fooocus V2".](https://github.com/lllyasviel/Fooocus/discussions/117#raw) (similar to Midjourney's hidden pre-processing and "raw" mode, or the LeonardoAI's Prompt Magic).
2. Native refiner swap inside one single k-sampler. The advantage is that the refiner model can now reuse the base model's momentum (or ODE's history parameters) collected from k-sampling to achieve more coherent sampling. In Automatic1111's high-res fix and ComfyUI's node system, the base model and refiner use two independent k-samplers, which means the momentum is largely wasted, and the sampling continuity is broken. Fooocus uses its own advanced k-diffusion sampling that ensures seamless, native, and continuous swap in a refiner setup. (Update Aug 13: Actually, I discussed this with Automatic1111 several days ago, and it seems that the “native refiner swap inside one single k-sampler” is [merged]( https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12371) into the dev branch of webui. Great!)
3. Negative ADM guidance. Because the highest resolution level of XL Base does not have cross attentions, the positive and negative signals for XL's highest resolution level cannot receive enough contrasts during the CFG sampling, causing the results to look a bit plastic or overly smooth in certain cases. Fortunately, since the XL's highest resolution level is still conditioned on image aspect ratios (ADM), we can modify the adm on the positive/negative side to compensate for the lack of CFG contrast in the highest resolution level. (Update Aug 16, the IOS App [Draw Things](https://apps.apple.com/us/app/draw-things-ai-generation/id6444050820) will support Negative ADM Guidance. Great!)
4. We implemented a carefully tuned variation of Section 5.1 of ["Improving Sample Quality of Diffusion Models Using Self-Attention Guidance"](https://arxiv.org/pdf/2210.00939.pdf). The weight is set to very low, but this is Fooocus's final guarantee to make sure that the XL will never yield an overly smooth or plastic appearance (examples [here](https://github.com/lllyasviel/Fooocus/discussions/117#sharpness)). This can almost eliminate all cases for which XL still occasionally produces overly smooth results, even with negative ADM guidance. (Update 2023 Aug 18, the Gaussian kernel of SAG is changed to an anisotropic kernel for better structure preservation and fewer artifacts.)
5. We modified the style templates a bit and added the "cinematic-default".
6. We tested the "sd_xl_offset_example-lora_1.0.safetensors" and it seems that when the lora weight is below 0.5, the results are always better than XL without lora.
7. The parameters of samplers are carefully tuned.
8. Because XL uses positional encoding for generation resolution, images generated by several fixed resolutions look a bit better than those from arbitrary resolutions (because the positional encoding is not very good at handling int numbers that are unseen during training). This suggests that the resolutions in UI may be hard coded for best results.
9. Separated prompts for two different text encoders seem unnecessary. Separated prompts for the base model and refiner may work, but the effects are random, and we refrain from implementing this.
10. The DPM family seems well-suited for XL since XL sometimes generates overly smooth texture, but the DPM family sometimes generates overly dense detail in texture. Their joint effect looks neutral and appealing to human perception.
11. A carefully designed system for balancing multiple styles as well as prompt expansion.
12. Using automatic1111's method to normalize prompt emphasizing. This significantly improves results when users directly copy prompts from civitai.
13. The joint swap system of the refiner now also supports img2img and upscale in a seamless way.
14. CFG Scale and TSNR correction (tuned for SDXL) when CFG is bigger than 10.
</details>

## Customization

After the first time you run Fooocus, a config file will be generated at `Fooocus\config.txt`. This file can be edited to change the model path or default parameters.

For example, an edited `Fooocus\config.txt` (this file will be generated after the first launch) may look like this:

```json
{
    "path_checkpoints": "D:\\Fooocus\\models\\checkpoints",
    "path_loras": "D:\\Fooocus\\models\\loras",
    "path_embeddings": "D:\\Fooocus\\models\\embeddings",
    "path_vae_approx": "D:\\Fooocus\\models\\vae_approx",
    "path_upscale_models": "D:\\Fooocus\\models\\upscale_models",
    "path_inpaint": "D:\\Fooocus\\models\\inpaint",
    "path_controlnet": "D:\\Fooocus\\models\\controlnet",
    "path_clip_vision": "D:\\Fooocus\\models\\clip_vision",
    "path_fooocus_expansion": "D:\\Fooocus\\models\\prompt_expansion\\fooocus_expansion",
    "path_outputs": "D:\\Fooocus\\outputs",
    "default_model": "realisticStockPhoto_v10.safetensors",
    "default_refiner": "",
    "default_loras": [["lora_filename_1.safetensors", 0.5], ["lora_filename_2.safetensors", 0.5]],
    "default_cfg_scale": 3.0,
    "default_sampler": "dpmpp_2m",
    "default_scheduler": "karras",
    "default_negative_prompt": "low quality",
    "default_positive_prompt": "",
    "default_styles": [
        "Fooocus V2",
        "Fooocus Photograph",
        "Fooocus Negative"
    ]
}
```

Many other keys, formats, and examples are in `Fooocus\config_modification_tutorial.txt` (this file will be generated after the first launch).

Consider twice before you really change the config. If you find yourself breaking things, just delete `Fooocus\config.txt`. Fooocus will go back to default.

A safer way is just to try "run_anime.bat" or "run_realistic.bat" - they should already be good enough for different tasks.

~Note that `user_path_config.txt` is deprecated and will be removed soon.~ (Edit: it is already removed.)

### All CMD Flags

```
entry_with_update.py  [-h] [--listen [IP]] [--port PORT]
                      [--disable-header-check [ORIGIN]]
                      [--web-upload-size WEB_UPLOAD_SIZE]
                      [--hf-mirror HF_MIRROR]
                      [--external-working-path PATH [PATH ...]]
                      [--output-path OUTPUT_PATH]
                      [--temp-path TEMP_PATH] [--cache-path CACHE_PATH]
                      [--in-browser] [--disable-in-browser]
                      [--gpu-device-id DEVICE_ID]
                      [--async-cuda-allocation | --disable-async-cuda-allocation]
                      [--disable-attention-upcast]
                      [--all-in-fp32 | --all-in-fp16]
                      [--unet-in-bf16 | --unet-in-fp16 | --unet-in-fp8-e4m3fn | --unet-in-fp8-e5m2]
                      [--vae-in-fp16 | --vae-in-fp32 | --vae-in-bf16]
                      [--vae-in-cpu]
                      [--clip-in-fp8-e4m3fn | --clip-in-fp8-e5m2 | --clip-in-fp16 | --clip-in-fp32]
                      [--directml [DIRECTML_DEVICE]]
                      [--disable-ipex-hijack]
                      [--preview-option [none,auto,fast,taesd]]
                      [--attention-split | --attention-quad | --attention-pytorch]
                      [--disable-xformers]
                      [--always-gpu | --always-high-vram | --always-normal-vram | --always-low-vram | --always-no-vram | --always-cpu [CPU_NUM_THREADS]]
                      [--always-offload-from-vram]
                      [--pytorch-deterministic] [--disable-server-log]
                      [--debug-mode] [--is-windows-embedded-python]
                      [--disable-server-info] [--multi-user] [--share]
                      [--preset PRESET] [--disable-preset-selection]
                      [--language LANGUAGE]
                      [--disable-offload-from-vram] [--theme THEME]
                      [--disable-image-log] [--disable-analytics]
                      [--disable-metadata] [--disable-preset-download]
                      [--disable-enhance-output-sorting]
                      [--enable-auto-describe-image]
                      [--always-download-new-model]
                      [--rebuild-hash-cache [CPU_NUM_THREADS]]
```

## Inline Prompt Features

### Wildcards

Example prompt: `__color__ flower`

Processed for positive and negative prompt.

Selects a random wildcard from a predefined list of options, in this case the `wildcards/color.txt` file. 
The wildcard will be replaced with a random color (randomness based on seed). 
You can also disable randomness and process a wildcard file from top to bottom by enabling the checkbox `Read wildcards in order` in Developer Debug Mode.

Wildcards can be nested and combined, and multiple wildcards can be used in the same prompt (example see `wildcards/color_flower.txt`).

### Array Processing

Example prompt: `[[red, green, blue]] flower`

Processed only for positive prompt.

Processes the array from left to right, generating a separate image for each element in the array. In this case 3 images would be generated, one for each color.
Increase the image number to 3 to generate all 3 variants.

Arrays can not be nested, but multiple arrays can be used in the same prompt.
Does support inline LoRAs as array elements!

### Inline LoRAs

Example prompt: `flower <lora:sunflowers:1.2>`

Processed only for positive prompt.

Applies a LoRA to the prompt. The LoRA file must be located in the `models/loras` directory.

## Advanced Features

[Click here to browse the advanced features.](https://github.com/lllyasviel/Fooocus/discussions/117)

## Forks

Below are some Forks to Fooocus:

| Fooocus' forks |
| - |
| [fenneishi/Fooocus-Control](https://github.com/fenneishi/Fooocus-Control) </br>[runew0lf/RuinedFooocus](https://github.com/runew0lf/RuinedFooocus) </br> [MoonRide303/Fooocus-MRE](https://github.com/MoonRide303/Fooocus-MRE) </br> [mashb1t/Fooocus](https://github.com/mashb1t/Fooocus) </br> and so on ... |

## Thanks

Many thanks to [twri](https://github.com/twri) and [3Diva](https://github.com/3Diva) and [Marc K3nt3L](https://github.com/K3nt3L) for creating additional SDXL styles available in Fooocus. 

The project starts from a mixture of [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) and [ComfyUI](https://github.com/comfyanonymous/ComfyUI) codebases.

Also, thanks [daswer123](https://github.com/daswer123) for contributing the Canvas Zoom!

## Update Log

The log is [here](update_log.md).

## Localization/Translation/I18N

You can put json files in the `language` folder to translate the user interface.

For example, below is the content of `Fooocus/language/example.json`:

```json
{
  "Generate": "生成",
  "Input Image": "入力画像",
  "Advanced": "고급",
  "SAI 3D Model": "SAI 3D Modèle"
}
```

If you add `--language example` arg, Fooocus will read `Fooocus/language/example.json` to translate the UI.

For example, you can edit the ending line of Windows `run.bat` as

    .\python_embeded\python.exe -s Fooocus\entry_with_update.py --language example

Or `run_anime.bat` as

    .\python_embeded\python.exe -s Fooocus\entry_with_update.py --language example --preset anime

Or `run_realistic.bat` as

    .\python_embeded\python.exe -s Fooocus\entry_with_update.py --language example --preset realistic

For practical translation, you may create your own file like `Fooocus/language/jp.json` or `Fooocus/language/cn.json` and then use flag `--language jp` or `--language cn`. Apparently, these files do not exist now. **We need your help to create these files!**

Note that if no `--language` is given and at the same time `Fooocus/language/default.json` exists, Fooocus will always load `Fooocus/language/default.json` for translation. By default, the file `Fooocus/language/default.json` does not exist.
