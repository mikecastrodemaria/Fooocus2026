"""UI Gradio de l'onglet Extra (cible Gradio 3.41).

Structure :
  - sous-onglet "Gestionnaire" : install depuis URL GitHub, liste, mises a jour.
  - un sous-onglet par plugin installe : controles generes depuis le manifeste,
    image d'entree, bouton Run, image de sortie, statut + VRAM.

Les plugins installes sont enumeres au demarrage de Fooocus. Apres une nouvelle
install, relancer Fooocus pour voir apparaitre son onglet (comportement type
custom-nodes ComfyUI). L'install elle-meme se fait a chaud.

custom-20 : mode serveur (modele garde chaud entre deux appels) quand le manifeste
declare un bloc server, avec repli CLI annonce ; mise a jour d'un plugin installe
depuis le Gestionnaire.
"""
import os
import time
import tempfile
import threading

import gradio as gr

from . import registry, runner, installer, manifest as manifest_mod, settings
from . import server as server_mod
from . import INSTALL_ROOT, OUTPUT_DIR, offload_host_models


_RESTART_NOTICE = ('<div style="padding:8px;border:1px solid #4ecdc4;'
                   'border-radius:6px;color:#4ecdc4;">✅ {msg} '
                   'Un redemarrage de l\'UI est requis pour reconstruire l\'onglet. '
                   'Clique <b>⚠ Restart UI</b> ci-dessus.</div>')


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
    """Closure : recoit (image, esrgan_dir, server_mode, *param_values) et lance le plugin."""
    pdir = plugin["dir"]
    m = plugin["manifest"]
    has_server = server_mod.server_spec(m) is not None

    def _run(image, esrgan_dir, server_mode, *vals):
        if image is None:
            return None, "Load an image first."
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # write the current image to a temp file
        in_path = os.path.join(tempfile.mkdtemp(), "extra_input.png")
        image.save(in_path)
        out_dir = os.path.join(OUTPUT_DIR, plugin["id"])
        os.makedirs(out_dir, exist_ok=True)

        param_values = dict(zip(keys, vals))
        use_server = has_server and bool(server_mode)
        # persist the chosen config (ESRGAN folder + params + mode) for next launches
        settings.set_plugin(plugin["id"], esrgan_dir=esrgan_dir or "",
                            params=param_values,
                            server_mode=bool(server_mode) if has_server else None)

        # offload the host model before the heavy call
        offloaded = offload_host_models()
        host_note = " | host offloaded" if offloaded else ""
        fallback_note = ""
        t0 = time.time()

        if use_server:
            try:
                url = server_mod.ensure(plugin, esrgan_dir=esrgan_dir or None, log_dir=out_dir)
                payload = runner.build_server_payload(m, in_path, out_dir, param_values)
                res = server_mod.upscale(plugin["id"], url, payload)
                dt = time.time() - t0
                state = "warm model" if res.get("was_warm") else "model loaded on this call"
                timing = ""
                if res.get("esrgan_s") is not None or res.get("refine_s") is not None:
                    timing = " (esrgan %.1fs + refine %.1fs)" % (
                        float(res.get("esrgan_s") or 0), float(res.get("refine_s") or 0))
                status = "OK in %.1fs via server %s, %s%s%s" % (dt, url, state, timing, host_note)
                return res["output"], status + "\n" + res["output"]
            except server_mod.ServerError as e:
                # degradation annoncee, jamais silencieuse : on le dit et on passe en CLI
                fallback_note = "Server mode unavailable, CLI fallback: %s\n" % e
                t0 = time.time()

        cmd = runner.build_upscale_command(
            m, pdir, input_path=in_path, output_dir=out_dir,
            param_values=param_values,
            esrgan_dir=esrgan_dir or None, report_vram=True)
        try:
            res = runner.run_upscale(cmd, pdir)
        except Exception as e:
            return None, fallback_note + f"Launch error: {e}"
        dt = time.time() - t0

        if not res["ok"]:
            tail = (res["stderr"] or "").strip().splitlines()[-8:]
            return None, fallback_note + "Failed (code %s)\n%s" % (res["returncode"], "\n".join(tail))

        status = "OK in %.1fs%s" % (dt, host_note)
        if res["vram"]:
            status += " | " + res["vram"]
        out_img = res["outputs"][-1]
        return out_img, fallback_note + status + "\n" + out_img

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
                # custom-20 : mode serveur, seulement si le manifeste le declare
                server_mode = gr.State(False)
                stop_srv_btn = None
                if server_mod.server_spec(m) is not None:
                    with gr.Row():
                        server_mode = gr.Checkbox(
                            label="Server mode (keep the model warm between runs)",
                            value=bool(saved.get("server_mode", True)),
                            info="Loads the model once and reuses it. Its VRAM is released "
                                 "as soon as Fooocus starts a generation.")
                        stop_srv_btn = gr.Button("⏹ Stop server", size="sm")
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
                      inputs=[in_image, esrgan_dir, server_mode] + comps,
                      outputs=[out_image, status])

        if stop_srv_btn is not None:
            def _stop_server(_pid=plugin["id"]):
                return "Server stopped." if server_mod.stop(_pid) else "No server was running."
            stop_srv_btn.click(_stop_server, outputs=[status], queue=False)

        _pstate = picked_state if picked_state is not None else gr.State(None)
        grab_btn.click(_load_grabbed, inputs=[_pstate], outputs=[in_image])


def _plugin_ids():
    return [p["id"] for p in registry.list_plugins(INSTALL_ROOT)]


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
                pdir = installer.install_from_github(
                    u.strip(), INSTALL_ROOT, strategy=strat,
                    base_python=bp.strip() or None, log=_log, force=frc)
                # custom-20 : la mise a jour rejouera la meme strategie d'environnement
                try:
                    data = manifest_mod.load(pdir)
                    settings.set_plugin(data["id"], install={
                        "strategy": strat, "base_python": bp.strip()})
                except Exception:
                    pass
            except Exception as e:
                ok = False
                lines.append("ERREUR: %s" % e)
            if ok:
                lines.append("")
                lines.append(">>> Installe. Clique '⚠ Restart UI' pour charger "
                             "le plugin : son onglet apparaitra apres le redemarrage.")
                notice = _RESTART_NOTICE.format(msg="Plugin installe.")
                return "\n".join(lines), _installed_md(), gr.update(value=notice, visible=True)
            return "\n".join(lines), _installed_md(), gr.update(visible=False)

        install_btn.click(_install, inputs=[url, strategy, base_python, force],
                          outputs=[log, installed, restart_notice])

        def _restart_ui():
            # Sortie code 42 : la boucle de run.bat / run.sh relance le process.
            def _do_exit():
                time.sleep(0.4)  # laisse la reponse Gradio partir avant de tuer
                server_mod.stop_all()
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

        # --- custom-20 : mises a jour ------------------------------------------
        gr.Markdown(
            "### Mises a jour\n"
            "Verifie le depot du plugin et applique ses nouveaux commits en avance "
            "rapide, jamais par-dessus un fichier modifie dans le plugin. Les "
            "dependances ne sont reinstallees que si leur fichier `requirements` a change. "
            "Le serveur du plugin est arrete avant la mise a jour.")
        ids = _plugin_ids()
        with gr.Row():
            upd_plugin = gr.Dropdown(label="Plugin installe", choices=ids,
                                     value=ids[0] if ids else None)
            upd_check_btn = gr.Button("\U0001F50D Verifier")
            upd_btn = gr.Button("⬆ Mettre a jour", variant="primary")
        upd_force = gr.Checkbox(
            label="Reinstaller les dependances meme si requirements n'a pas change",
            value=False)
        upd_log = gr.Textbox(label="Journal de mise a jour", lines=12, interactive=False)

        def _check_update(pid):
            p = registry.get_plugin(INSTALL_ROOT, pid) if pid else None
            if not p:
                return "Choisis un plugin installe."
            return "\n".join(installer.format_status(installer.update_status(p["dir"])))

        def _apply_update(pid, force_deps):
            p = registry.get_plugin(INSTALL_ROOT, pid) if pid else None
            if not p:
                return "Choisis un plugin installe.", _installed_md(), gr.update()
            lines = []
            if server_mod.stop(pid):
                lines.append("Serveur du plugin arrete avant la mise a jour.")
            inst = settings.get_plugin(pid).get("install") or {}
            try:
                res = installer.update_plugin(
                    p["dir"], strategy=inst.get("strategy") or "fresh_venv",
                    base_python=inst.get("base_python") or None,
                    log=lambda s: lines.append(str(s)), force_deps=bool(force_deps))
            except Exception as e:
                lines.append("ERREUR : %s" % e)
                return "\n".join(lines), _installed_md(), gr.update()
            if res["manifest_changed"]:
                lines.append("")
                lines.append(">>> Le manifeste a change : clique '⚠ Restart UI' pour "
                             "reconstruire l'onglet du plugin.")
                notice = _RESTART_NOTICE.format(msg="Plugin mis a jour, manifeste modifie.")
                return "\n".join(lines), _installed_md(), gr.update(value=notice, visible=True)
            return "\n".join(lines), _installed_md(), gr.update()

        upd_check_btn.click(_check_update, inputs=[upd_plugin], outputs=[upd_log])
        upd_btn.click(_apply_update, inputs=[upd_plugin, upd_force],
                      outputs=[upd_log, installed, restart_notice])


def _installed_md():
    plugins = registry.list_plugins(INSTALL_ROOT)
    if not plugins:
        return "_Aucun plugin installe._"
    rows = ["Plugins installes :"]
    for p in plugins:
        commit = installer.current_commit(p["dir"])
        rows.append("- **%s** v%s (`%s`%s)" % (
            p["name"], p["version"], p["id"], (" @ `%s`" % commit) if commit else ""))
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
