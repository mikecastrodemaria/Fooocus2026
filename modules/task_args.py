"""custom-26 — Snapshot de ctrls d'un AsyncTask construit hors de l'UI.

Extrait de xyz_cli.build_base_args (custom-15.3) pour etre partage avec le protocole
CLI de la famille crispz (fooocus_protocol.py) : l'ordre des ctrls de
AsyncTask.__init__ (modules/async_worker.py) n'est plus recopie qu'a UN endroit hors
de webui.py. Toute modification de l'ordre la-bas doit etre reportee ici ; les
appelants valident le snapshot en construisant un AsyncTask a blanc, qui refuse un
vecteur mal aligne.

build() renvoie (args, idx) : idx memorise, PENDANT la construction, la position de
chaque ctrl utile (prompt, uov_method, cn0_image, enh0_mask_prompt...). Aucune position
n'est codee en dur chez les appelants.
"""


def build(prompt, negative=None, performance=None, aspect=None, output_format=None, seed=None,
          image_number=1):
    """Snapshot complet aux valeurs de config (comme l'UI au demarrage). seed None =
    tire au hasard ; les autres None = defaut de config."""
    import random
    import modules.config as cfg
    import modules.flags as flags

    def g(attr, fallback):
        return getattr(cfg, attr, fallback)

    args, idx = [], {}

    def put(name, value):
        idx[name] = len(args)
        args.append(value)

    put('generate_image_grid', False)
    put('prompt', prompt)
    put('negative', negative if negative is not None else g('default_prompt_negative', ''))
    put('styles', list(g('default_styles', [])))
    put('performance', performance or g('default_performance', 'Quality'))
    put('aspect', aspect or g('default_aspect_ratio', '1152×896'))
    put('image_number', image_number)
    put('output_format', output_format or g('default_output_format', 'png'))
    put('seed', seed if seed is not None else random.randint(0, 2 ** 32))
    put('read_wildcards_in_order', False)
    put('sharpness', g('default_sample_sharpness', 2.0))
    put('cfg', g('default_cfg_scale', 4.0))
    put('base_model', g('default_base_model_name', 'model.safetensors'))
    put('refiner', g('default_refiner_model_name', 'None'))
    put('refiner_switch', g('default_refiner_switch', 0.5))

    n_loras = int(g('default_max_lora_number', 5))
    loras = []
    for entry in (g('default_loras', []) or [])[:n_loras]:
        if isinstance(entry, (list, tuple)) and len(entry) >= 3:
            loras.append((bool(entry[0]), str(entry[1]), float(entry[2])))
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            loras.append((True, str(entry[0]), float(entry[1])))
    while len(loras) < n_loras:
        loras.append((False, 'None', 1.0))
    for i, (enabled, name, weight) in enumerate(loras):
        put(f'lora{i}_enabled', enabled)
        put(f'lora{i}_name', name)
        put(f'lora{i}_weight', weight)

    put('input_image_checkbox', False)
    put('current_tab', 'uov')
    put('uov_method', flags.disabled)
    put('uov_input_image', None)
    put('outpaint_selections', [])
    put('inpaint_input_image', None)
    put('inpaint_additional_prompt', '')
    put('inpaint_mask_image_upload', None)
    put('disable_preview', False)
    put('disable_intermediate_results', False)
    put('disable_seed_increment', False)
    put('black_out_nsfw', bool(g('default_black_out_nsfw', False)))
    put('adm_scaler_positive', 1.5)
    put('adm_scaler_negative', 0.8)
    put('adm_scaler_end', 0.3)
    put('adaptive_cfg', g('default_cfg_tsnr', 7.0))
    put('clip_skip', g('default_clip_skip', 2))
    put('sampler', g('default_sampler', 'dpmpp_2m_sde_gpu'))
    put('scheduler', g('default_scheduler', 'karras'))
    put('vae', g('default_vae', getattr(flags, 'default_vae', 'Default (model)')))
    put('overwrite_step', g('default_overwrite_step', -1))
    put('overwrite_switch', g('default_overwrite_switch', -1))
    put('overwrite_width', -1)
    put('overwrite_height', -1)
    put('overwrite_vary_strength', -1)
    put('overwrite_upscale_strength', g('default_overwrite_upscale', -1))
    put('mixing_image_prompt_and_vary_upscale', False)
    put('mixing_image_prompt_and_inpaint', False)
    put('use_aspect_for_vary', False)             # custom-6
    put('custom_res_enabled', False)              # custom-7 (desactive)
    put('custom_ratio_w', 4)
    put('custom_ratio_h', 3)
    put('custom_res_mode', 'Ratio + taille max')
    put('custom_res_size', 1024)
    put('debugging_cn_preprocessor', False)
    put('skipping_cn_preprocessor', False)
    put('canny_low_threshold', 64)
    put('canny_high_threshold', 128)
    put('refiner_swap_method', flags.refiner_swap_method)
    put('controlnet_softness', 0.25)
    put('freeu_enabled', False)
    put('freeu_b1', 1.01)
    put('freeu_b2', 1.02)
    put('freeu_s1', 0.99)
    put('freeu_s2', 0.95)
    put('debugging_inpaint_preprocessor', False)
    put('inpaint_disable_initial_latent', False)
    put('inpaint_engine', g('default_inpaint_engine_version', 'v2.6'))
    put('inpaint_strength', 1.0)
    put('inpaint_respective_field', 0.618)
    put('inpaint_advanced_masking_checkbox', False)
    put('invert_mask_checkbox', False)
    put('inpaint_erode_or_dilate', 0)
    put('save_final_enhanced_image_only', False)
    put('save_metadata_to_images', bool(g('default_save_metadata_to_images', False)))
    put('metadata_scheme', g('default_metadata_scheme', 'fooocus'))

    cn_stop, cn_weight = flags.default_parameters[flags.default_ip]
    for i in range(int(g('default_controlnet_image_count', 4))):
        put(f'cn{i}_image', None)
        put(f'cn{i}_stop', cn_stop)
        put(f'cn{i}_weight', cn_weight)
        put(f'cn{i}_type', flags.default_ip)

    put('debugging_dino', False)
    put('dino_erode_or_dilate', 0)
    put('debugging_enhance_masks_checkbox', False)
    put('enhance_input_image', None)
    put('enhance_checkbox', False)
    put('enhance_uov_method', flags.disabled)
    put('enhance_uov_processing_order', flags.enhancement_uov_before)
    put('enhance_uov_prompt_type', flags.enhancement_uov_prompt_type_original)
    engine = g('default_inpaint_engine_version', 'v2.6')
    for i in range(int(g('default_enhance_tabs', 3))):
        put(f'enh{i}_enabled', False)
        put(f'enh{i}_mask_prompt', '')
        put(f'enh{i}_prompt', '')
        put(f'enh{i}_negative', '')
        put(f'enh{i}_mask_model', 'sam')
        put(f'enh{i}_cloth_category', 'full')
        put(f'enh{i}_sam_model', 'vit_b')
        put(f'enh{i}_text_threshold', 0.25)
        put(f'enh{i}_box_threshold', 0.3)
        put(f'enh{i}_sam_max_detections', 0)
        put(f'enh{i}_disable_initial_latent', False)
        put(f'enh{i}_engine', engine)
        put(f'enh{i}_strength', 1.0)
        put(f'enh{i}_respective_field', 0.618)
        put(f'enh{i}_erode_or_dilate', 0)
        put(f'enh{i}_invert', False)

    put('upscaler_model', 'Fooocus Default (ESRGAN)')  # custom-10 : modele Fooocus par defaut
    put('face_restore_model', 'Off')
    put('face_restore_visibility', 0.8)
    put('face_restore_order', 'After upscale')
    return args, idx


def lora_slot_count(idx):
    """Nombre de slots LoRA du snapshot (depuis les positions memorisees)."""
    return sum(1 for name in idx if name.startswith('lora') and name.endswith('_name'))
