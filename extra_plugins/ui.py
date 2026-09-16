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

custom-24 : un plugin peut declarer plusieurs actions (bloc `actions` du manifeste),
chacune avec ses params, ses flags et ses images supplementaires (ex. le visage
source d'un face swap). Sans bloc actions, l'onglet est identique a avant.
"""
import os
import time
import tempfile
import threading

import gradio as gr

from . import registry, runner, installer, manifest as manifest_mod, settings
from . import server as server_mod
from . import envcheck
from . import INSTALL_ROOT, OUTPUT_DIR, offload_host_models


_RESTART_NOTICE = ('<div style="padding:8px;border:1px solid #4ecdc4;'
                   'border-radius:6px;color:#4ecdc4;">✅ {msg} '
                   'A UI restart is required to rebuild the tab. '
                   'Click <b>⚠ Restart UI</b> above.</div>')


# Helpers exposes a webui.py pour l'etat persiste de la case "Extra Plugins".
def enabled_default():
    return settings.get_enabled()


def save_enabled(value):
    settings.set_enabled(value)


def _build_param_controls(m, saved_params=None, only=None):
    """Cree les composants Gradio pour les params d'un manifeste (ceux de `only`
    quand il est donne, dans l'ordre du manifeste).

    Renvoie (components, keys) alignes par index.
    """
    components, keys = [], []
    saved_params = saved_params or {}
    wanted = None if only is None else set(only)
    for p in m.get("params", []):
        if wanted is not None and p["key"] not in wanted:
            continue
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


def _make_run_handler(plugin, action, keys, show_models):
    """Closure : recoit (image, esrgan_dir, server_mode, *param_values, *images_action)."""
    pdir = plugin["dir"]
    pid = plugin["id"]
    m = plugin["manifest"]
    has_server = action["server"] and server_mod.server_spec(m) is not None
    image_params = action["image_params"]

    def _run(image, esrgan_dir, server_mode, *vals):
        if image is None:
            return None, "Load an image first."
        param_vals, extra_images = vals[:len(keys)], vals[len(keys):]
        missing = [ip.get("label") or ip["key"]
                   for ip, img in zip(image_params, extra_images) if img is None]
        if missing:
            return None, "Load an image in: " + ", ".join(missing) + "."
        # custom-30 : un venv dont le torch est incomplet (install interrompue) ne peut
        # pas demarrer ; on le dit avec la commande de reparation, sans lancer le run
        problem = envcheck.preflight(pdir, m)
        if problem:
            return None, problem
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # write the current image(s) to temp files
        tmp_dir = tempfile.mkdtemp()
        in_path = os.path.join(tmp_dir, "extra_input.png")
        image.save(in_path)
        image_args = []
        for ip, img in zip(image_params, extra_images):
            p = os.path.join(tmp_dir, "%s.png" % ip["key"])
            img.save(p)
            image_args.append((ip["arg"], p))
        out_dir = os.path.join(OUTPUT_DIR, pid)
        os.makedirs(out_dir, exist_ok=True)

        param_values = dict(zip(keys, param_vals))
        use_server = has_server and bool(server_mode)
        # persist the chosen config for next launches ; params fusionnes : chaque action
        # ne voit que les siens, elle ne doit pas effacer ceux des autres
        previous = settings.get_plugin(pid).get("params") or {}
        settings.set_plugin(pid,
                            esrgan_dir=(esrgan_dir or "") if show_models else None,
                            params={**previous, **param_values},
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
                res = server_mod.upscale(pid, url, payload)
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
            esrgan_dir=(esrgan_dir or None) if show_models else None, report_vram=True,
            extra_args=action["args"], image_args=image_args)
        try:
            res = runner.run_upscale(cmd, pdir)
        except Exception as e:
            return None, fallback_note + f"Launch error: {e}"
        dt = time.time() - t0

        if not res["ok"]:
            tail = (res["stderr"] or "").strip().splitlines()[-8:]
            text = fallback_note + "Failed (code %s)\n%s" % (res["returncode"], "\n".join(tail))
            hint = envcheck.explain_failure(text, m, pdir)   # custom-30 : torch DLL / import
            return None, text + ("\n" + hint if hint else "")

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


def _build_action(plugin, action, saved, picked_state):
    """Controles d'une action : image(s) a gauche, reglages a droite."""
    m = plugin["manifest"]
    model_param = next((p for p in m.get("params", [])
                        if p.get("choices_cmd") and p["key"] in action["params"]), None)
    show_models = model_param is not None
    with gr.Row():
        with gr.Column():
            # Images on the left: input + extra images of the action + result
            in_image = gr.Image(label="Input image", type="pil")
            grab_btn = gr.Button(
                "⬇ Get a generated image (gallery selection, else the latest)",
                size="sm")
            extra_images = [gr.Image(label=ip.get("label") or ip["key"], type="pil")
                            for ip in action["image_params"]]
            out_image = gr.Image(label="Result", type="filepath")
        with gr.Column():
            # Settings on the right, run button on top
            run_btn = gr.Button(action["label"], variant="primary")
            if action.get("note"):
                gr.Markdown("*%s*" % action["note"])
            # custom-20 : mode serveur, seulement si l'action et le manifeste le permettent
            server_mode = gr.State(False)
            stop_srv_btn = None
            if action["server"] and server_mod.server_spec(m) is not None:
                with gr.Row():
                    server_mode = gr.Checkbox(
                        label="Server mode (keep the model warm between runs)",
                        value=bool(saved.get("server_mode", True)),
                        info="Loads the model once and reuses it. Its VRAM is released "
                             "as soon as Fooocus starts a generation.")
                    stop_srv_btn = gr.Button("⏹ Stop server", size="sm")
            esrgan_dir = gr.State("")
            refresh = None
            if show_models:
                esrgan_dir = gr.Textbox(
                    label="ESRGAN folder (optional, plugin default otherwise)",
                    value=saved.get("esrgan_dir", ""))
                # Refresh button placed right after the ESRGAN folder field
                refresh = gr.Button("Refresh models", size="sm")
            comps, keys = _build_param_controls(m, saved.get("params"), only=action["params"])
            status = gr.Textbox(label="Status", lines=4, interactive=False)
            if refresh is not None:
                model_idx = keys.index(model_param["key"])

                def _refresh(edir, _pid=plugin["id"], _pdir=plugin["dir"], _m=m):
                    settings.set_plugin(_pid, esrgan_dir=edir or "")
                    # custom-30 : un torch incomplet videra la liste en silence ; on l'explique
                    problem = envcheck.preflight(_pdir, _m)
                    if problem:
                        return gr.update(choices=[], value=None), problem
                    models = runner.list_models(_m, _pdir, esrgan_dir=edir or None)
                    note = ("%d model(s) found." % len(models) if models
                            else "No model found (check the ESRGAN folder).")
                    return gr.update(choices=models,
                                     value=models[0] if models else None), note
                refresh.click(_refresh, inputs=[esrgan_dir], outputs=[comps[model_idx], status])

    run_btn.click(_make_run_handler(plugin, action, keys, show_models),
                  inputs=[in_image, esrgan_dir, server_mode] + comps + extra_images,
                  outputs=[out_image, status])

    if stop_srv_btn is not None:
        def _stop_server(_pid=plugin["id"]):
            return "Server stopped." if server_mod.stop(_pid) else "No server was running."
        stop_srv_btn.click(_stop_server, outputs=[status], queue=False)

    _pstate = picked_state if picked_state is not None else gr.State(None)
    grab_btn.click(_load_grabbed, inputs=[_pstate], outputs=[in_image])


def _build_plugin_tab(plugin, picked_state=None):
    m = plugin["manifest"]
    acts = manifest_mod.actions(m)
    with gr.Tab(label=plugin["name"]):
        saved = settings.get_plugin(plugin["id"])
        gr.Markdown("**%s** v%s — %s" % (
            plugin["name"], plugin.get("version", "?"),
            m.get("description", "")))
        if len(acts) == 1:
            _build_action(plugin, acts[0], saved, picked_state)
        else:
            with gr.Tabs():
                for act in acts:
                    with gr.Tab(label=act["label"]):
                        _build_action(plugin, act, saved, picked_state)


def _plugin_ids():
    return [p["id"] for p in registry.list_plugins(INSTALL_ROOT)]


def _build_manager_tab():
    with gr.Tab(label="Manager"):
        gr.Markdown(
            "Install an Extra plugin from a GitHub repository (with a "
            "`fooocus_extra.json`). The plugin runs in its own venv. "
            "After an install, restart Fooocus to see its tab.")
        with gr.Row():
            url = gr.Textbox(label="GitHub URL",
                             placeholder="https://github.com/user/plugin")
            strategy = gr.Dropdown(
                label="Environment",
                choices=["fresh_venv", "reuse_python"], value="fresh_venv")
        base_python = gr.Textbox(
            label="Base Python (for reuse_python, e.g. py -3.10 or a path)",
            value="")
        with gr.Row():
            install_btn = gr.Button("Install", variant="primary")
            force = gr.Checkbox(label="Force (reinstall if present)", value=False)
        log = gr.Textbox(label="Install log", lines=14, interactive=False)
        installed = gr.Markdown(_installed_md())

        restart_notice = gr.HTML(value="", visible=False)
        with gr.Row():
            restart_btn = gr.Button(value="\U000026A0 Restart UI", variant="stop",
                                    min_width=130, scale=1)
            gr.Markdown("A plugin tab only appears at startup. After an "
                        "install, click Restart UI (relaunches via run.bat / run.sh).")

        def _install(u, strat, bp, frc):
            lines = []
            def _log(s):
                lines.append(str(s))
            if not u.strip():
                return ("Enter a GitHub URL.", _installed_md(),
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
                lines.append("ERROR: %s" % e)
            if ok:
                lines.append("")
                lines.append(">>> Installed. Click '⚠ Restart UI' to load "
                             "the plugin: its tab will appear after the restart.")
                notice = _RESTART_NOTICE.format(msg="Plugin installed.")
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
                      'border-radius:6px;color:#ffa500;">⚠ Restarting… '
                      'wait ~30 s, then refresh the page. If it does not come back, '
                      'your launcher does not implement the restart loop: relaunch '
                      'the .bat / .sh by hand.</div>',
                visible=True)

        restart_btn.click(_restart_ui, outputs=[restart_notice])

        # --- custom-20 : mises a jour ------------------------------------------
        gr.Markdown(
            "### Updates\n"
            "Checks the plugin repository and applies its new commits as a "
            "fast-forward, never over a file modified inside the plugin. "
            "Dependencies are reinstalled only if their `requirements` file changed. "
            "The plugin server is stopped before the update.")
        ids = _plugin_ids()
        with gr.Row():
            upd_plugin = gr.Dropdown(label="Installed plugin", choices=ids,
                                     value=ids[0] if ids else None)
            upd_check_btn = gr.Button("\U0001F50D Check")
            upd_btn = gr.Button("⬆ Update", variant="primary")
        upd_force = gr.Checkbox(
            label="Reinstall dependencies even if requirements did not change",
            value=False)
        upd_log = gr.Textbox(label="Update log", lines=12, interactive=False)

        def _check_update(pid):
            p = registry.get_plugin(INSTALL_ROOT, pid) if pid else None
            if not p:
                return "Pick an installed plugin."
            return "\n".join(installer.format_status(installer.update_status(p["dir"])))

        def _apply_update(pid, force_deps):
            p = registry.get_plugin(INSTALL_ROOT, pid) if pid else None
            if not p:
                return "Pick an installed plugin.", _installed_md(), gr.update()
            lines = []
            if server_mod.stop(pid):
                lines.append("Plugin server stopped before the update.")
            inst = settings.get_plugin(pid).get("install") or {}
            try:
                res = installer.update_plugin(
                    p["dir"], strategy=inst.get("strategy") or "fresh_venv",
                    base_python=inst.get("base_python") or None,
                    log=lambda s: lines.append(str(s)), force_deps=bool(force_deps))
            except Exception as e:
                lines.append("ERROR: %s" % e)
                return "\n".join(lines), _installed_md(), gr.update()
            if res["manifest_changed"]:
                lines.append("")
                lines.append(">>> The manifest changed: click '⚠ Restart UI' to "
                             "rebuild the plugin tab.")
                notice = _RESTART_NOTICE.format(msg="Plugin updated, manifest changed.")
                return "\n".join(lines), _installed_md(), gr.update(value=notice, visible=True)
            return "\n".join(lines), _installed_md(), gr.update()

        upd_check_btn.click(_check_update, inputs=[upd_plugin], outputs=[upd_log])
        upd_btn.click(_apply_update, inputs=[upd_plugin, upd_force],
                      outputs=[upd_log, installed, restart_notice])


def _installed_md():
    plugins = registry.list_plugins(INSTALL_ROOT)
    if not plugins:
        return "_No plugin installed._"
    rows = ["Installed plugins:"]
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
                with gr.Tab(label="%s (error)" % plugin.get("id", "?")):
                    gr.Markdown("Build error: %s" % e)
        _build_manager_tab()
