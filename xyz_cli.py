"""custom-15.3 — xyz_cli.py : grille X/Y/Z en ligne de commande, sans navigateur.

Usage (depuis la racine du depot, avec le python de l'install) :

  ..\\python_embeded\\Scripts\\python.exe -s xyz_cli.py ^
      --prompt "a castle at dawn, masterpiece" ^
      --x "CFG:3,5,7" --y "Steps:20,40" ^
      --seed 12345 --dry-run

  Axes : --x/--y/--z au format "NomAxe:val1,val2,..." parmi ceux de
  modules/xyz_grid.py (CFG, Steps, Sampler, Scheduler, Sharpness,
  Checkpoint, LoRA 1 weight, Preset, Prompt S/R). Noms d'axes
  insensibles a la casse.

  --preset applique d'abord un preset au snapshot de base.
  --dry-run construit et valide tous les jobs sans rien generer.

  Les flags Fooocus inconnus de ce script (--always-gpu, etc.) sont
  transmis tels quels a args_manager. La planche finit dans
  outputs/xyz_grids/ comme depuis l'UI.

Garde-fou : chaque snapshot est valide en construisant un AsyncTask a
blanc; si l'ordre des ctrls change un jour dans async_worker.py, le
script refuse de demarrer au lieu de generer n'importe quoi.
"""
import argparse
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def parse_cli():
    p = argparse.ArgumentParser(add_help=True, description='Fooocus2026 X/Y/Z grid from the command line')
    p.add_argument('--prompt', required=True)
    p.add_argument('--negative', default=None)
    p.add_argument('--preset', default=None, help='preset applied to the base snapshot (partial name accepted)')
    p.add_argument('--x', required=True, help='"AxisName:val1,val2,..."')
    p.add_argument('--y', default=None)
    p.add_argument('--z', default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--performance', default=None, help='Quality / Speed / ...')
    p.add_argument('--aspect', default=None, help='e.g. 1152×896 (default: config)')
    p.add_argument('--output-format', default=None, choices=[None, 'png', 'jpeg', 'webp'])
    p.add_argument('--dry-run', action='store_true')
    ours, passthrough = p.parse_known_args()
    # args_manager (importe plus tard) ne doit voir que les flags Fooocus
    sys.argv = [sys.argv[0]] + passthrough
    return ours


def resolve_axis(name, xyz):
    low = str(name).strip().lower()
    for k in xyz.AXES:
        if k.lower() == low:
            return k
    raise SystemExit(f'[xyz-cli] unknown axis "{name}". Choices: {", ".join(xyz.AXES)}')


def parse_axis_arg(raw, xyz):
    if ':' not in raw:
        raise SystemExit(f'[xyz-cli] expected format "AxisName:val1,val2" (got: "{raw}")')
    name, values = raw.split(':', 1)
    axis = resolve_axis(name, xyz)
    try:
        return axis, xyz.parse_values(axis, values)
    except ValueError as e:
        # custom-18 : valeur invalide (LoRA introuvable, poids illisible) -> message
        # net, pas une traceback. Le nom de l'axe est deja dans le message quand il
        # apporte quelque chose, on ne le repete pas.
        raise SystemExit(f'[xyz-cli] {e}')


def build_base_args(a):
    """Snapshot ctrls complet, STRICTEMENT dans l'ordre de AsyncTask.__init__
    (modules/async_worker.py). custom-26 : construit par modules/task_args.py, partage
    avec le protocole CLI (fooocus_protocol.py), pour que cet ordre fragile ne soit
    recopie qu'a un seul endroit. Le dry-run AsyncTask ci-dessous echoue bruyamment
    en cas de derive."""
    import modules.task_args as task_args
    args, _idx = task_args.build(prompt=a.prompt, negative=a.negative, performance=a.performance,
                                 aspect=a.aspect, output_format=a.output_format, seed=a.seed)
    return args


def validate(args_list, worker, label):
    """Dry-run AsyncTask : consommation exacte ou refus."""
    probe = list(args_list)
    task = worker.AsyncTask(args=probe)
    if len(probe) != 0:
        raise SystemExit(f'[xyz-cli] ERROR: {len(probe)} ctrls not consumed for "{label}". '
                         'The AsyncTask order changed: update build_base_args().')
    return task


def main():
    a = parse_cli()
    print('[xyz-cli] Starting the Fooocus backend (please wait, torch is waking up)...')
    import modules.async_worker as worker
    import modules.config  # noqa: F401  (charge la config + listes de modeles)
    import modules.xyz_grid as xyz

    # custom-18 : l'import seul laisse model_filenames/lora_filenames VIDES (c'est
    # launch.py qui scanne, et le CLI ne passe pas par la). Sans ce scan, les axes
    # Checkpoint et LoRA acceptaient n'importe quelle valeur sans la resoudre :
    # la serie partait avec un nom de fichier inexistant.
    modules.config.update_files()

    base = build_base_args(a)
    if a.preset:
        xyz._apply_preset(base, a.preset)
        print(f'[xyz-cli] Preset applied: {a.preset}')

    spec = [parse_axis_arg(a.x, xyz)]
    if a.y:
        spec.append(parse_axis_arg(a.y, xyz))
    if a.z:
        spec.append(parse_axis_arg(a.z, xyz))

    try:
        jobs, group = xyz.expand(base, spec)
    except ValueError as e:
        # custom-18 : l'axe Checkpoint resout ses valeurs ici (dans _apply), pas au
        # parsing -> sans ce filet, un modele introuvable sortait en traceback.
        raise SystemExit(f'[xyz-cli] {e}')
    print(f'[xyz-cli] {len(jobs)} cells to generate '
          f'({group["nx"]}x{group["ny"]}x{group["nz"]}), seed {base[8]}.')

    # validation integrale AVANT de generer quoi que ce soit
    for args_i, label_i, _ in jobs:
        validate(args_i, worker, label_i)
    print('[xyz-cli] Snapshots valid (ctrls order OK).')
    if a.dry_run:
        for _, label_i, _ in jobs:
            print(f'[xyz-cli]   {label_i}')
        print('[xyz-cli] Dry run finished, nothing was generated.')
        return

    xyz.register_group(group)
    prev_model = None
    grids = None
    try:
        for args_i, label_i, meta_i in jobs:
            next_model = args_i[12]
            if prev_model is not None and next_model != prev_model:
                import ldm_patched.modules.model_management as mm
                print(f'[xyz-cli] Checkpoint change: purging VRAM.')
                mm.unload_all_models()
                mm.soft_empty_cache()
            prev_model = next_model
            print(f'[xyz-cli] >>> {label_i}')
            task = worker.AsyncTask(args=list(args_i))
            worker.async_tasks.append(task)
            finished = False
            while not finished:
                time.sleep(0.05)
                while task.yields:
                    flag, product = task.yields.pop(0)
                    if flag == 'finish':
                        finished = True
            first = next((r for r in task.results if isinstance(r, str)), None)
            grids = xyz.on_job_done(meta_i, first)
    except KeyboardInterrupt:
        print('\n[xyz-cli] Interrupted. Cells already generated stay in outputs/.')
        return
    if grids:
        print('[xyz-cli] Done. Sheet(s):')
        for gpath in grids:
            print(f'[xyz-cli]   {gpath}')
    else:
        print('[xyz-cli] Done without a sheet (missing cells?). See outputs/.')


if __name__ == '__main__':
    main()
