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
    p = argparse.ArgumentParser(add_help=True, description='Grille X/Y/Z Fooocus2026 en CLI')
    p.add_argument('--prompt', required=True)
    p.add_argument('--negative', default=None)
    p.add_argument('--preset', default=None, help='preset applique au snapshot de base (nom partiel accepte)')
    p.add_argument('--x', required=True, help='"NomAxe:val1,val2,..."')
    p.add_argument('--y', default=None)
    p.add_argument('--z', default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--performance', default=None, help='Quality / Speed / ...')
    p.add_argument('--aspect', default=None, help='ex: 1152×896 (defaut: config)')
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
    raise SystemExit(f'[xyz-cli] axe inconnu "{name}". Choix: {", ".join(xyz.AXES)}')


def parse_axis_arg(raw, xyz):
    if ':' not in raw:
        raise SystemExit(f'[xyz-cli] format attendu "NomAxe:val1,val2" (recu: "{raw}")')
    name, values = raw.split(':', 1)
    axis = resolve_axis(name, xyz)
    return axis, xyz.parse_values(axis, values)


def build_base_args(a):
    """Snapshot ctrls complet, STRICTEMENT dans l'ordre de AsyncTask.__init__
    (modules/async_worker.py). Toute modification la-bas doit etre reportee ici;
    le dry-run AsyncTask ci-dessous echoue bruyamment en cas de derive."""
    import modules.config as cfg
    import modules.flags as flags

    def g(attr, fallback):
        return getattr(cfg, attr, fallback)

    n_loras = int(g('default_max_lora_number', 5))
    loras = []
    for entry in (g('default_loras', []) or [])[:n_loras]:
        if isinstance(entry, (list, tuple)) and len(entry) >= 3:
            loras += [bool(entry[0]), str(entry[1]), float(entry[2])]
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            loras += [True, str(entry[0]), float(entry[1])]
    while len(loras) < 3 * n_loras:
        loras += [False, 'None', 1.0]

    cn_stop, cn_weight = flags.default_parameters[flags.default_ip]
    args = [
        False,                                        # generate_image_grid
        a.prompt,                                     # 1 prompt
        a.negative if a.negative is not None else g('default_prompt_negative', ''),
        list(g('default_styles', [])),                # 3 styles
        a.performance or g('default_performance', 'Quality'),
        a.aspect or g('default_aspect_ratio', '1152×896'),
        1,                                            # 6 image_number (1 par case)
        a.output_format or g('default_output_format', 'png'),
        a.seed if a.seed is not None else random.randint(0, 2 ** 32),
        False,                                        # read_wildcards_in_order
        g('default_sample_sharpness', 2.0),           # 10
        g('default_cfg_scale', 4.0),                  # 11
        g('default_base_model_name', 'model.safetensors'),  # 12
        g('default_refiner_model_name', 'None'),
        g('default_refiner_switch', 0.5),
    ] + loras + [
        False,                                        # input_image_checkbox
        'uov',                                        # current_tab
        flags.disabled,                               # uov_method
        None,                                         # uov_input_image
        [],                                           # outpaint_selections
        None,                                         # inpaint_input_image
        '',                                           # inpaint_additional_prompt
        None,                                         # inpaint_mask_image_upload
        False, False, False,                          # disable_preview / intermediate / seed_increment
        bool(g('default_black_out_nsfw', False)),
        1.5, 0.8, 0.3,                                # adm +/-/end
        g('default_cfg_tsnr', 7.0),
        g('default_clip_skip', 2),
        g('default_sampler', 'dpmpp_2m_sde_gpu'),
        g('default_scheduler', 'karras'),
        g('default_vae', getattr(flags, 'default_vae', 'Default (model)')),
        g('default_overwrite_step', -1),
        g('default_overwrite_switch', -1),
        -1, -1, -1,                                   # overwrite width / height / vary_strength
        g('default_overwrite_upscale', -1),
        False, False,                                 # mixing image prompt vary / inpaint
        False,                                        # use_aspect_for_vary (custom-6)
        False, 4, 3, 'Ratio + taille max', 1024,      # custom-7 (desactive)
        False, False, 64, 128,                        # cn preprocessor debug/skip + canny
        flags.refiner_swap_method,
        0.25,                                         # controlnet_softness
        False, 1.01, 1.02, 0.99, 0.95,                # freeu
        False, False,                                 # inpaint debug / disable_initial_latent
        g('default_inpaint_engine_version', 'v2.6'),
        1.0, 0.618, False, False, 0,                  # inpaint strength/field/advanced/invert/erode
        False,                                        # save_final_enhanced_image_only
        bool(g('default_save_metadata_to_images', False)),
        g('default_metadata_scheme', 'fooocus'),
    ]
    for _ in range(int(g('default_controlnet_image_count', 4))):
        args += [None, cn_stop, cn_weight, flags.default_ip]
    args += [
        False, 0, False,                              # dino debug / erode / enhance masks debug
        None,                                         # enhance_input_image
        False,                                        # enhance_checkbox
        flags.disabled,                               # enhance_uov_method
        flags.enhancement_uov_before,
        flags.enhancement_uov_prompt_type_original,
    ]
    for _ in range(int(g('default_enhance_tabs', 3))):
        args += [False, '', '', '', 'sam', 'full', 'vit_b', 0.25, 0.3, 0,
                 False, g('default_inpaint_engine_version', 'v2.6'), 1.0, 0.618, 0, False]
    args += [
        'Fooocus Default (ESRGAN)',                   # custom-10 upscaler sentinel
        'Off', 0.8, 'After upscale',                  # face restore model / visibility / order
    ]
    return args


def validate(args_list, worker, label):
    """Dry-run AsyncTask : consommation exacte ou refus."""
    probe = list(args_list)
    task = worker.AsyncTask(args=probe)
    if len(probe) != 0:
        raise SystemExit(f'[xyz-cli] ERREUR: {len(probe)} ctrls non consommes pour "{label}". '
                         'L\'ordre de AsyncTask a change: mettre a jour build_base_args().')
    return task


def main():
    a = parse_cli()
    print('[xyz-cli] Demarrage du backend Fooocus (patiente, torch se reveille)...')
    import modules.async_worker as worker
    import modules.config  # noqa: F401  (charge la config + listes de modeles)
    import modules.xyz_grid as xyz

    base = build_base_args(a)
    if a.preset:
        xyz._apply_preset(base, a.preset)
        print(f'[xyz-cli] Preset applique: {a.preset}')

    spec = [parse_axis_arg(a.x, xyz)]
    if a.y:
        spec.append(parse_axis_arg(a.y, xyz))
    if a.z:
        spec.append(parse_axis_arg(a.z, xyz))

    jobs, group = xyz.expand(base, spec)
    print(f'[xyz-cli] {len(jobs)} cases a generer '
          f'({group["nx"]}x{group["ny"]}x{group["nz"]}), seed {base[8]}.')

    # validation integrale AVANT de generer quoi que ce soit
    for args_i, label_i, _ in jobs:
        validate(args_i, worker, label_i)
    print('[xyz-cli] Snapshots valides (ordre des ctrls OK).')
    if a.dry_run:
        for _, label_i, _ in jobs:
            print(f'[xyz-cli]   {label_i}')
        print('[xyz-cli] Dry-run termine, rien n\'a ete genere.')
        return

    xyz.register_group(group)
    prev_model = None
    grids = None
    try:
        for args_i, label_i, meta_i in jobs:
            next_model = args_i[12]
            if prev_model is not None and next_model != prev_model:
                import ldm_patched.modules.model_management as mm
                print(f'[xyz-cli] Changement de checkpoint: purge VRAM.')
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
        print('\n[xyz-cli] Interrompu. Les cases deja generees restent dans outputs/.')
        return
    if grids:
        print('[xyz-cli] Termine. Planche(s):')
        for gpath in grids:
            print(f'[xyz-cli]   {gpath}')
    else:
        print('[xyz-cli] Termine sans planche (cases manquantes ?). Voir outputs/.')


if __name__ == '__main__':
    main()
