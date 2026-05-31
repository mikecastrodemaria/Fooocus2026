"""Construction et execution des commandes CLI d'un plugin Extra.

Contrat (manifeste entry.output.mode == 'print_output') :
  - crispz ecrit l'image dans output_dir et imprime le chemin absolu sur stdout.
  - stderr porte les logs et la ligne '[VRAM] ...'.
  - code de sortie 0 = succes.

Module pur (subprocess). Aucune dependance a Fooocus.
"""
import os
import subprocess
import sys


def _is_windows():
    return os.name == "nt"


def venv_python(plugin_dir):
    if _is_windows():
        return os.path.join(plugin_dir, ".venv", "Scripts", "python.exe")
    return os.path.join(plugin_dir, ".venv", "bin", "python")


def venv_pip(plugin_dir):
    if _is_windows():
        return os.path.join(plugin_dir, ".venv", "Scripts", "pip.exe")
    return os.path.join(plugin_dir, ".venv", "bin", "pip")


def _fmt_value(v):
    """Formate une valeur de parametre pour la ligne de commande.

    Gradio (Number, parfois Slider) renvoie des floats : -1.0, 760.0, 12.0.
    crispz attend des int pour seed/tile/overlap/steps/refine-*. On convertit
    tout float entier en int. Les vrais floats (factor 2.5, denoise 0.3) restent.
    """
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _subst(token, ctx):
    """Remplace les placeholders {xxx} d'un token de commande."""
    if not isinstance(token, str):
        return token
    for k, v in ctx.items():
        token = token.replace("{" + k + "}", str(v))
    return token


def _base_command(manifest, plugin_dir):
    ctx = {
        "venv_python": venv_python(plugin_dir),
        "plugin_dir": plugin_dir,
    }
    return [_subst(tok, ctx) for tok in manifest["entry"]["command"]]


def build_upscale_command(manifest, plugin_dir, input_path, output_dir,
                          param_values, esrgan_dir=None, report_vram=False):
    """Construit la commande CLI complete pour un upscale.

    param_values : dict key -> valeur (cf params du manifeste).
    """
    entry = manifest["entry"]
    cmd = _base_command(manifest, plugin_dir)

    # Entree
    cmd += [entry["input_arg"], input_path]

    # Sortie (contrat print_output)
    out = entry["output"]
    cmd += list(out.get("save_mode_flag", []))
    save_dir_flag = out.get("save_dir_flag", [])
    cmd += [_subst(t, {"output_dir": output_dir}) for t in save_dir_flag]
    cmd += list(out.get("format_flag", []))
    cmd += [out["flag"]]  # --print-output

    # Dossier modeles ESRGAN
    if esrgan_dir and entry.get("models_arg"):
        cmd += [_subst(t, {"esrgan_dir": esrgan_dir}) for t in entry["models_arg"]]

    # Parametres UI -> flags
    by_key = {p["key"]: p for p in manifest.get("params", [])}
    for key, val in param_values.items():
        spec = by_key.get(key)
        if not spec or val is None:
            continue
        # choices_cmd (ex: model) : pas de valeur vide
        if isinstance(val, str) and val == "" and spec.get("type") != "text":
            continue
        cmd += [spec["arg"], _fmt_value(val)]

    if report_vram and entry.get("vram_report_flag"):
        cmd += [entry["vram_report_flag"]]

    return cmd


def list_models(manifest, plugin_dir, esrgan_dir=None, timeout=120):
    """Renvoie la liste des modeles ESRGAN via la commande choices_cmd du param model."""
    model_param = next((p for p in manifest.get("params", [])
                        if p.get("choices_cmd")), None)
    if not model_param:
        return []
    cmd = _base_command(manifest, plugin_dir) + list(model_param["choices_cmd"])
    entry = manifest["entry"]
    if esrgan_dir and entry.get("models_arg"):
        cmd += [_subst(t, {"esrgan_dir": esrgan_dir}) for t in entry["models_arg"]]
    try:
        proc = subprocess.run(cmd, cwd=plugin_dir, capture_output=True,
                              text=True, timeout=timeout)
    except Exception as e:
        return []
    if proc.returncode != 0:
        return []
    return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]


def run_upscale(cmd, plugin_dir, timeout=1800):
    """Lance la commande, renvoie un dict resultat.

    {ok, returncode, outputs (chemins absolus), vram (ligne brute ou None),
     stdout, stderr}
    """
    proc = subprocess.run(cmd, cwd=plugin_dir, capture_output=True,
                          text=True, timeout=timeout)
    outputs = []
    for ln in proc.stdout.splitlines():
        ln = ln.strip()
        if ln and os.path.isabs(ln):
            outputs.append(ln)
    vram = None
    for ln in proc.stderr.splitlines():
        if ln.startswith("[VRAM]"):
            vram = ln.strip()
    return {
        "ok": proc.returncode == 0 and bool(outputs),
        "returncode": proc.returncode,
        "outputs": outputs,
        "vram": vram,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
