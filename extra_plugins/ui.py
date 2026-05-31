"""UI Gradio de l'onglet Extra (cible Gradio 3.41).

Structure :
  - sous-onglet "Gestionnaire" : install depuis URL GitHub, liste, suppression.
  - un sous-onglet par plugin installe : controles generes depuis le manifeste,
    image d'entree, bouton Run, image de sortie, statut + VRAM.

Les plugins installes sont enumeres au demarrage de Fooocus. Apres une nouvelle
install, relancer Fooocus pour voir apparaitre son onglet (comportement type
custom-nodes ComfyUI). L'install elle-meme se fait a chaud.
"""
import os
import time
import tempfile
import threading

import gradio as gr

from . import registry, runner, installer, manifest as manifest_mod, settings
from . import INSTALL_ROOT, OUTPUT_DIR, offload_host_models


# Helpers exposes a webui.py pour l'etat persiste de la case "Extra Plugins".
def enabled_default():
    return settings.get_enabled()


def save_enabled(value):
    settings.set_enabled(value)


def _build_param_controls(m, saved_params=None):
    """Cree les composants Gradio pour les params d'un manifeste.

    Renvoie (components, keys) alignes par index.
    """
    components, keys = [], []
    saved_params = saved_params or {}
    for p in m.get("params", []):
        t = p["type"]
        label = p.get("label", p["key"])
        # valeur sauvee prioritaire sur le defaut du manifeste
        default = saved_params.get(p["key"], p.get("default"))
        if t == "dropdown":
            choices = list(p.get("choices", []))
            # un modele ESRGAN sauve n'est pas dans les choix statiques : l'ajouter
            if default and default not in choices:
                choices = [default] + choices
            comp = gr.Dropdown(label=label, choices=choices,
                               value=default if default is not None else (choices[0] if choices else None),
                               allow_custom_value=True)
        elif t == "slider":
            comp = gr.Slider(label=label, minimum=p.get("min", 0),
                             maximum=p.get("max", 1), step=p.get("step", 1),
                             value=default)
        elif t == "number":
            comp = gr.Number(label=label, value=default)
        else:  # text
            comp = gr.Textbox(label=label, value=default or "")
        components.append(comp)
        keys.append(p["key"])
    return components, keys


def _make_run_handler(plugin, keys):
    """Closure : recoit (image, esrgan_dir, *param_values) et lance crispz."""
    pdir = plugin["dir"]
    m = plugin["manifest"]

    def _run(image, esrgan_dir, *vals):
        if image is None:
            return None, "Load an image first."
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # write the current image to a temp file
        in_path = os.path.join(tempfile.mkdtemp(), "extra_input.png")
        image.save(in_path)
        out_dir = os.path.join(OUTPUT_DIR, plugin["id"])
        os.makedirs(out_dir, exist_ok=True)

        param_values = dict(zip(keys, vals))
        # persist the chosen config (ESRGAN folder + params) for next launches
        settings.set_plugin(plugin["id"], esrgan_dir=esrgan_dir or "",
                            params=param_values)
        cmd = runner.build_upscale_command(
            m, pdir, input_path=in_path, output_dir=out_dir,
            param_values=param_values,
            esrgan_dir=esrgan_dir or None, report_vram=True)

        # offload the host model before the heavy call
        offloaded = offload_host_models()
        t0 = time.time()
        try:
            res = runner.run_upscale(cmd, pdir)
        except Exception as e:
            return None, f"Launch error: {e}"
        dt = time.time() - t0

        if not res["ok"]:
            tail = (res["stderr"] or "").strip().splitlines()[-8:]
            return None, "Failed (code %s)\n%s" % (res["returncode"], "\n".join(tail))

        status = "OK in %.1fs%s" % (dt, " | host offloaded" if offloaded else "")
        if res["vram"]:
            status += " | " + res["vram"]
        out_img = res["outputs"][-1]
        return out_img, status + "\n" + out_img

    return _run


def _newest_output_image():
    """Chemin de l'image generee la plus recente dans le dossier de sortie Fooocus."""
    try:
        import modules.config as cfg
        root = cfg.path_outputs
    except Exception:
        return None
    exts = (".png", ".jpg", ".jpeg", ".webp")
    newest, newest_t = None, -1.0
    for dirpath, _, files in os.walk(root or ""):
        for f in files:
            if f.lower().endswith(exts):
                p = os.path.join(dirpath, f)
                try:
                    t = os.path.getmtime(p)
                except OSError:
                    continue
                if t > newest_t:
                    newest, newest_t = p, t
    return newest


def _path_from_gallery_select(value):
    """Normalise la valeur d'un SelectData de gr.Gallery en chemin de fichier."""
    v = value
    if isinstance(v, dict):
        return v.get("name") or v.get("path") or v.get("url")
    if isinstance(v, (list, tuple)) and v:
        first = v[0]
        return first.get("name") if isinstance(first, dict) else first
    return v if isinstance(v, str) else None


def _load_grabbed(picked):
    """Image a injecter : selection galerie si dispo, sinon derniere generee."""
    path = picked or _newest_output_image()
    if not path or not os.path.isfile(path):
        return None
    try:
        from PIL import Image as _PILImage
        return _PILImage.open(path).convert("RGB")
    except Exception:
        return None


def _build_plugin_tab(plugin, picked_state=None):
    m = plugin["manifest"]
    with gr.Tab(label=plugin["name"]):
        saved = settings.get_plugin(plugin["id"])
        gr.Markdown("**%s** v%s — %s" % (
            plugin["name"], plugin.get("version", "?"),
            m.get("description", "")))
        with gr.Row():
            with gr.Column():
                # Images on the left: input + result
                in_image = gr.Image(label="Input image", type="pil")
                grab_btn = gr.Button(
                    "⬇ Get a generated image (gallery selection, else the latest)",
                    size="sm")
                out_image = gr.Image(label="Result", type="filepath")
            with gr.Column():
                # Settings on the right, Upscale button on top
                run_btn = gr.Button("Upscale", variant="primary")
                esrgan_dir = gr.Textbox(
                    label="ESRGAN folder (optional, plugin default otherwise)",
                    value=saved.get("esrgan_dir", ""))
                # Refresh button placed right after the ESRGAN folder field
                model_param_idx = next(
                    (i for i, p in enumerate(m.get("params", []))
                     if p.get("choices_cmd")), None)
                refresh = None
                if model_param_idx is not None:
                    refresh = gr.Button("Refresh models", size="sm")
                comps, keys = _build_param_controls(m, saved.get("params"))
                if refresh is not None:
                    def _refresh(edir, _pid=plugin["id"], _pdir=plugin["dir"], _m=m):
                        settings.set_plugin(_pid, esrgan_dir=edir or "")
                        models = runner.list_models(_m, _pdir, esrgan_dir=edir or None)
                        return gr.update(choices=models,
                                         value=models[0] if models else None)
                    refresh.click(_refresh, inputs=[esrgan_dir],
                                  outputs=[comps[model_param_idx]])
                status = gr.Textbox(label="Status", lines=4, interactive=False)

        run_btn.click(_make_run_handler(plugin, keys),
                      inputs=[in_image, esrgan_dir] + comps,
                      outputs=[out_image, status])

        _pstate = picked_state if picked_state is not None else gr.State(None)
        grab_btn.click(_load_grabbed, inputs=[_pstate], outputs=[in_image])


def _build_manager_tab():
    with gr.Tab(label="Gestionnaire"):
        gr.Markdown(
            "Installe un plugin Extra depuis un depot GitHub (avec un "
            "`fooocus_extra.json`). Le plugin tourne dans son propre venv. "
            "Apres une install, relance Fooocus pour voir son onglet.")
        with gr.Row():
            url = gr.Textbox(label="URL GitHub",
                             placeholder="https://github.com/utilisateur/plugin")
            strategy = gr.Dropdown(
                label="Environnement",
                choices=["fresh_venv", "reuse_python"], value="fresh_venv")
        base_python = gr.Textbox(
            label="Python de base (pour reuse_python, ex: py -3.10 ou chemin)",
            value="")
        with gr.Row():
            install_btn = gr.Button("Installer", variant="primary")
            force = gr.Checkbox(label="Forcer (reinstaller si present)", value=False)
        log = gr.Textbox(label="Journal d'installation", lines=14, interactive=False)
        installed = gr.Markdown(_installed_md())

        restart_notice = gr.HTML(value="", visible=False)
        with gr.Row():
            restart_btn = gr.Button(value="\U000026A0 Restart UI", variant="stop",
                                    min_width=130, scale=1)
            gr.Markdown("L'onglet d'un plugin n'apparait qu'au demarrage. Apres une "
                        "install, clique Restart UI (relance via run.bat / run.sh).")

        def _install(u, strat, bp, frc):
            lines = []
            def _log(s):
                lines.append(str(s))
            if not u.strip():
                return ("Donne une URL GitHub.", _installed_md(),
                        gr.update(visible=False))
            ok = True
            try:
                installer.install_from_github(
                    u.strip(), INSTALL_ROOT, strategy=strat,
                    base_python=bp.strip() or None, log=_log, force=frc)
            except Exception as e:
                ok = False
                lines.append("ERREUR: %s" % e)
            if ok:
                lines.append("")
                lines.append(">>> Installe. Clique '⚠ Restart UI' pour charger "
                             "le plugin : son onglet apparaitra apres le redemarrage.")
                notice = ('<div style="padding:8px;border:1px solid #4ecdc4;'
                          'border-radius:6px;color:#4ecdc4;">✅ Plugin installe. '
                          'Un redemarrage de l\'UI est requis pour afficher son onglet. '
                          'Clique <b>⚠ Restart UI</b> ci-dessus.</div>')
                return "\n".join(lines), _installed_md(), gr.update(value=notice, visible=True)
            return "\n".join(lines), _installed_md(), gr.update(visible=False)

        install_btn.click(_install, inputs=[url, strategy, base_python, force],
                          outputs=[log, installed, restart_notice])

        def _restart_ui():
            # Sortie code 42 : la boucle de run.bat / run.sh relance le process.
            def _do_exit():
                time.sleep(0.4)  # laisse la reponse Gradio partir avant de tuer
                os._exit(42)
            threading.Thread(target=_do_exit, daemon=True).start()
            return gr.update(
                value='<div style="padding:8px;border:1px solid #ffa500;'
                      'border-radius:6px;color:#ffa500;">⚠ Redemarrage… '
                      'attends ~30 s puis rafraichis la page. Si elle ne revient pas, '
                      'ton lanceur n\'implemente pas la boucle de restart : relance '
                      'le .bat / .sh a la main.</div>',
                visible=True)

        restart_btn.click(_restart_ui, outputs=[restart_notice])


def _installed_md():
    plugins = registry.list_plugins(INSTALL_ROOT)
    if not plugins:
        return "_Aucun plugin installe._"
    rows = ["Plugins installes :"]
    for p in plugins:
        rows.append("- **%s** v%s (`%s`)" % (p["name"], p["version"], p["id"]))
    return "\n".join(rows)


def build_extra_panel(output_gallery=None):
    """Point d'entree appele depuis webui.py pour construire l'onglet Extra.

    output_gallery : la gr.Gallery de sortie de Fooocus. Si fournie, l'image
    selectionnee dedans peut etre recuperee vers l'entree d'un plugin.
    """
    with gr.Tabs():
        picked_state = gr.State(None)
        if output_gallery is not None:
            def _on_gallery_select(evt: gr.SelectData):
                return _path_from_gallery_select(evt.value)
            output_gallery.select(_on_gallery_select, None, picked_state)
        # Plugins d'abord, Gestionnaire en dernier.
        for plugin in registry.list_plugins(INSTALL_ROOT):
            try:
                _build_plugin_tab(plugin, picked_state=picked_state)
            except Exception as e:
                with gr.Tab(label="%s (erreur)" % plugin.get("id", "?")):
                    gr.Markdown("Erreur de construction: %s" % e)
        _build_manager_tab()
