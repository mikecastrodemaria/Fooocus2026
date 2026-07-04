import gradio as gr
import random
import os
import sys
import json
import time
import shared
import modules.config
import fooocus_version
import modules.html
import modules.async_worker as worker
import modules.constants as constants
import modules.flags as flags
import modules.gradio_hijack as grh
import modules.style_sorter as style_sorter
import modules.meta_parser
import modules.civitai_api
import args_manager
import copy
import launch
from extras.inpaint_mask import SAMOptions
from extra_plugins import ui as extra_ui

from modules.sdxl_styles import legal_style_names
from modules.private_logger import get_current_html_path
from modules.ui_gradio_extensions import reload_javascript
from modules.auth import auth_enabled, check_auth
from modules.util import is_json

def get_task(*args):
    args = list(args)
    args.pop(0)

    return worker.AsyncTask(args=args)

def execute_task_streaming(task: worker.AsyncTask):
    """custom-14: corps de generation partage par Generate et Run queue."""
    import ldm_patched.modules.model_management as model_management

    with model_management.interrupt_processing_mutex:
        model_management.interrupt_processing = False
    # outputs=[progress_html, progress_window, progress_gallery, gallery]

    if len(task.args) == 0:
        return

    execution_start_time = time.perf_counter()
    finished = False

    yield gr.update(visible=True, value=modules.html.make_progress_html(1, 'Waiting for task to start ...')), \
        gr.update(visible=True, value=None), \
        gr.update(visible=False, value=None), \
        gr.update(visible=False)

    worker.async_tasks.append(task)

    while not finished:
        time.sleep(0.01)
        if len(task.yields) > 0:
            flag, product = task.yields.pop(0)
            if flag == 'preview':

                # help bad internet connection by skipping duplicated preview
                if len(task.yields) > 0:  # if we have the next item
                    if task.yields[0][0] == 'preview':   # if the next item is also a preview
                        # print('Skipped one preview for better internet connection.')
                        continue

                percentage, title, image = product
                yield gr.update(visible=True, value=modules.html.make_progress_html(percentage, title)), \
                    gr.update(visible=True, value=image) if image is not None else gr.update(), \
                    gr.update(), \
                    gr.update(visible=False)
            if flag == 'results':
                yield gr.update(visible=True), \
                    gr.update(visible=True), \
                    gr.update(visible=True, value=product), \
                    gr.update(visible=False)
            if flag == 'finish':
                if not args_manager.args.disable_enhance_output_sorting:
                    product = sort_enhance_images(product, task)

                yield gr.update(visible=False), \
                    gr.update(visible=False), \
                    gr.update(visible=False), \
                    gr.update(visible=True, value=product)
                finished = True

                # delete Fooocus temp images, only keep gradio temp images
                if args_manager.args.disable_image_log:
                    for filepath in product:
                        if isinstance(filepath, str) and os.path.exists(filepath):
                            os.remove(filepath)

    execution_time = time.perf_counter() - execution_start_time
    print(f'Total time: {execution_time:.2f} seconds')
    return


def generate_clicked(task: worker.AsyncTask):
    # inchange : un seul job, comportement historique
    yield from execute_task_streaming(task)


def run_queue_clicked():
    """custom-14 : depile et execute les jobs sequentiellement.
    Stop = interrompt le job courant et met la file en pause (rien n'est perdu).
    """
    from modules import job_queue as jq
    if len(jq.queue) == 0:
        yield gr.update(visible=False), gr.update(visible=False), \
            gr.update(visible=False), gr.update(visible=True)
        return
    jq.queue.paused = False
    while not jq.queue.paused:
        job = jq.queue.pop_next()
        if job is None:
            break
        task = worker.AsyncTask(args=list(job.args))
        jq.queue.current_task = task
        print(f'[JobQueue] Job lance : {job.label} ({len(jq.queue)} restant(s))')
        yield from execute_task_streaming(task)
        jq.queue.current_task = None
        if task.last_stop == 'stop':
            jq.queue.paused = True
            print(f'[JobQueue] Stop : file en pause, {len(jq.queue)} job(s) en attente.')
            break
    jq.queue.current_task = None


def sort_enhance_images(images, task):
    if not task.should_enhance or len(images) <= task.images_to_enhance_count:
        return images

    sorted_images = []
    walk_index = task.images_to_enhance_count

    for index, enhanced_img in enumerate(images[:task.images_to_enhance_count]):
        sorted_images.append(enhanced_img)
        if index not in task.enhance_stats:
            continue
        target_index = walk_index + task.enhance_stats[index]
        if walk_index < len(images) and target_index <= len(images):
            sorted_images += images[walk_index:target_index]
        walk_index += task.enhance_stats[index]

    return sorted_images


def inpaint_mode_change(mode, inpaint_engine_version):
    assert mode in modules.flags.inpaint_options

    # inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
    # inpaint_disable_initial_latent, inpaint_engine,
    # inpaint_strength, inpaint_respective_field

    if mode == modules.flags.inpaint_option_detail:
        return [
            gr.update(visible=True), gr.update(visible=False, value=[]),
            gr.Dataset.update(visible=True, samples=modules.config.example_inpaint_prompts),
            False, 'None', 0.5, 0.0
        ]

    if inpaint_engine_version == 'empty':
        inpaint_engine_version = modules.config.default_inpaint_engine_version

    if mode == modules.flags.inpaint_option_modify:
        return [
            gr.update(visible=True), gr.update(visible=False, value=[]),
            gr.Dataset.update(visible=False, samples=modules.config.example_inpaint_prompts),
            True, inpaint_engine_version, 1.0, 0.0
        ]

    return [
        gr.update(visible=False, value=''), gr.update(visible=True),
        gr.Dataset.update(visible=False, samples=modules.config.example_inpaint_prompts),
        False, inpaint_engine_version, 1.0, 0.618
    ]


reload_javascript()

title = f'Fooocus {fooocus_version.version}'

if isinstance(args_manager.args.preset, str):
    title += ' ' + args_manager.args.preset

shared.gradio_root = gr.Blocks(title=title).queue()

with shared.gradio_root:
    currentTask = gr.State(worker.AsyncTask(args=[]))
    inpaint_engine_state = gr.State('empty')
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                progress_window = grh.Image(label='Preview', show_label=True, visible=False, height=768,
                                            elem_classes=['main_view'])
                progress_gallery = gr.Gallery(label='Finished Images', show_label=True, object_fit='contain',
                                              height=768, visible=False, elem_classes=['main_view', 'image_gallery'])
            progress_html = gr.HTML(value=modules.html.make_progress_html(32, 'Progress 32%'), visible=False,
                                    elem_id='progress-bar', elem_classes='progress-bar')
            gallery = gr.Gallery(label='Gallery', show_label=False, object_fit='contain', visible=True, height=768,
                                 elem_classes=['resizable_area', 'main_view', 'final_gallery', 'image_gallery'],
                                 elem_id='final_gallery')
            with gr.Row():
                with gr.Column(scale=17):
                    prompt = gr.Textbox(show_label=False, placeholder="Type prompt here or paste parameters.", elem_id='positive_prompt',
                                        autofocus=True, lines=3)

                    default_prompt = modules.config.default_prompt
                    if isinstance(default_prompt, str) and default_prompt != '':
                        shared.gradio_root.load(lambda: default_prompt, outputs=prompt)

                    gr.HTML('<div class="prompt_spacer" style="height:8px;"></div>')

                    negative_prompt = gr.Textbox(
                        show_label=False,
                        placeholder="Negative prompt \u2014 describe what you do not want to see.",
                        elem_id='negative_prompt', lines=2,
                        value=modules.config.default_prompt_negative)

                with gr.Column(scale=3, min_width=0):
                    generate_button = gr.Button(label="Generate", value="Generate", elem_classes='type_row', elem_id='generate_button', visible=True)
                    reset_button = gr.Button(label="Reconnect", value="Reconnect", elem_classes='type_row', elem_id='reset_button', visible=False)
                    load_parameter_button = gr.Button(label="Load Parameters", value="Load Parameters", elem_classes='type_row', elem_id='load_parameter_button', visible=False)
                    skip_button = gr.Button(label="Skip", value="Skip", elem_classes='type_row_half', elem_id='skip_button', visible=False)
                    stop_button = gr.Button(label="Stop", value="Stop", elem_classes='type_row_half', elem_id='stop_button', visible=False)

                    def stop_clicked(currentTask):
                        import ldm_patched.modules.model_management as model_management
                        from modules import job_queue as jq
                        # custom-14 : pendant Run queue, viser le job en cours
                        task = jq.queue.current_task or currentTask
                        task.last_stop = 'stop'
                        jq.queue.paused = True  # pause la file, les jobs restants attendent
                        if (task.processing):
                            model_management.interrupt_current_processing()
                        return currentTask

                    def skip_clicked(currentTask):
                        import ldm_patched.modules.model_management as model_management
                        from modules import job_queue as jq
                        task = jq.queue.current_task or currentTask
                        task.last_stop = 'skip'
                        if (task.processing):
                            model_management.interrupt_current_processing()
                        return currentTask

                    stop_button.click(stop_clicked, inputs=currentTask, outputs=currentTask, queue=False, show_progress=False, _js='cancelGenerateForever')
                    skip_button.click(skip_clicked, inputs=currentTask, outputs=currentTask, queue=False, show_progress=False)

                    # custom-14 : ajout a la file d'attente (visible si job_queue.enabled)
                    if modules.config.job_queue_enabled():
                        queue_add_button = gr.Button(label="+ Queue", value="+ Queue",
                                                     elem_classes='type_row_half', elem_id='queue_add_button',
                                                     visible=True)
            with gr.Row(elem_classes='advanced_check_row'):
                input_image_checkbox = gr.Checkbox(label='Input Image', value=modules.config.default_image_prompt_checkbox, container=False, elem_classes='min_check')
                enhance_checkbox = gr.Checkbox(label='Enhance', value=modules.config.default_enhance_checkbox, container=False, elem_classes='min_check')
                advanced_checkbox = gr.Checkbox(label='Advanced', value=modules.config.default_advanced_checkbox, container=False, elem_classes='min_check')
                if modules.config.job_queue_enabled():
                    queue_checkbox = gr.Checkbox(label='Job Queue', value=False, container=False, elem_classes='min_check')
            # custom-14: panneau Job Queue, bascule par la case du dessus
            if modules.config.job_queue_enabled():
                with gr.Row(visible=False) as job_queue_panel:
                    with gr.Column():
                        gr.HTML('<div style="font-size:12px;color:#888;margin-bottom:6px;">'
                                'Empilez des generations avec le bouton <b>+ Queue</b> sous Generate '
                                '(chaque job fige les reglages du moment), puis lancez la serie. '
                                'Stop interrompt le job courant et met la file en pause.</div>')
                        queue_status = gr.HTML(value='File vide.')
                        queue_display = gr.Radio(label='Jobs en attente', choices=[], value=None, interactive=True)
                        with gr.Row():
                            queue_run_button = gr.Button(value='\u25B6 Run queue', variant='primary', scale=2)
                            queue_up_button = gr.Button(value='\U0001F53C Up', scale=1)
                            queue_down_button = gr.Button(value='\U0001F53D Down', scale=1)
                            queue_remove_button = gr.Button(value='\u274C Remove', scale=1)
                            queue_clear_button = gr.Button(value='\U0001F5D1 Clear', scale=1)
            with gr.Row(visible=modules.config.default_image_prompt_checkbox) as image_input_panel:
                with gr.Tabs(selected=modules.config.default_selected_image_input_tab_id):
                    with gr.Tab(label='Upscale or Variation', id='uov_tab') as uov_tab:
                        with gr.Row():
                            with gr.Column():
                                uov_input_image = grh.Image(label='Image', source='upload', type='numpy', show_label=False)
                            with gr.Column():
                                uov_method = gr.Radio(label='Upscale or Variation:', choices=flags.uov_list, value=modules.config.default_uov_method)
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/390" target="_blank">\U0001F4D4 Documentation</a>')

                    with gr.Tab(label='Image Prompt', id='ip_tab') as ip_tab:
                        with gr.Row():
                            ip_images = []
                            ip_types = []
                            ip_stops = []
                            ip_weights = []
                            ip_ctrls = []
                            ip_ad_cols = []
                            for image_count in range(modules.config.default_controlnet_image_count):
                                image_count += 1
                                with gr.Column():
                                    ip_image = grh.Image(label='Image', source='upload', type='numpy', show_label=False, height=300, value=modules.config.default_ip_images[image_count])
                                    ip_images.append(ip_image)
                                    ip_ctrls.append(ip_image)
                                    with gr.Column(visible=modules.config.default_image_prompt_advanced_checkbox) as ad_col:
                                        with gr.Row():
                                            ip_stop = gr.Slider(label='Stop At', minimum=0.0, maximum=1.0, step=0.001, value=modules.config.default_ip_stop_ats[image_count])
                                            ip_stops.append(ip_stop)
                                            ip_ctrls.append(ip_stop)

                                            ip_weight = gr.Slider(label='Weight', minimum=0.0, maximum=2.0, step=0.001, value=modules.config.default_ip_weights[image_count])
                                            ip_weights.append(ip_weight)
                                            ip_ctrls.append(ip_weight)

                                        ip_type = gr.Radio(label='Type', choices=flags.ip_list, value=modules.config.default_ip_types[image_count], container=False)
                                        ip_types.append(ip_type)
                                        ip_ctrls.append(ip_type)

                                        ip_type.change(lambda x: flags.default_parameters[x], inputs=[ip_type], outputs=[ip_stop, ip_weight], queue=False, show_progress=False)
                                    ip_ad_cols.append(ad_col)
                        ip_advanced = gr.Checkbox(label='Advanced', value=modules.config.default_image_prompt_advanced_checkbox, container=False)
                        gr.HTML('* \"Image Prompt\" is powered by Fooocus Image Mixture Engine (v1.0.1). <a href="https://github.com/lllyasviel/Fooocus/discussions/557" target="_blank">\U0001F4D4 Documentation</a>')

                        def ip_advance_checked(x):
                            return [gr.update(visible=x)] * len(ip_ad_cols) + \
                                [flags.default_ip] * len(ip_types) + \
                                [flags.default_parameters[flags.default_ip][0]] * len(ip_stops) + \
                                [flags.default_parameters[flags.default_ip][1]] * len(ip_weights)

                        ip_advanced.change(ip_advance_checked, inputs=ip_advanced,
                                           outputs=ip_ad_cols + ip_types + ip_stops + ip_weights,
                                           queue=False, show_progress=False)

                    with gr.Tab(label='Inpaint or Outpaint', id='inpaint_tab') as inpaint_tab:
                        with gr.Row():
                            with gr.Column():
                                inpaint_input_image = grh.Image(label='Image', source='upload', type='numpy', tool='sketch', height=500, brush_color="#FFFFFF", elem_id='inpaint_canvas', show_label=False)
                                inpaint_advanced_masking_checkbox = gr.Checkbox(label='Enable Advanced Masking Features', value=modules.config.default_inpaint_advanced_masking_checkbox)
                                inpaint_mode = gr.Dropdown(choices=modules.flags.inpaint_options, value=modules.config.default_inpaint_method, label='Method')
                                inpaint_additional_prompt = gr.Textbox(placeholder="Describe what you want to inpaint.", elem_id='inpaint_additional_prompt', label='Inpaint Additional Prompt', visible=False)
                                outpaint_selections = gr.CheckboxGroup(choices=['Left', 'Right', 'Top', 'Bottom'], value=[], label='Outpaint Direction')
                                example_inpaint_prompts = gr.Dataset(samples=modules.config.example_inpaint_prompts,
                                                                     label='Additional Prompt Quick List',
                                                                     components=[inpaint_additional_prompt],
                                                                     visible=False)
                                gr.HTML('* Powered by Fooocus Inpaint Engine <a href="https://github.com/lllyasviel/Fooocus/discussions/414" target="_blank">\U0001F4D4 Documentation</a>')
                                example_inpaint_prompts.click(lambda x: x[0], inputs=example_inpaint_prompts, outputs=inpaint_additional_prompt, show_progress=False, queue=False)

                            with gr.Column(visible=modules.config.default_inpaint_advanced_masking_checkbox) as inpaint_mask_generation_col:
                                inpaint_mask_image = grh.Image(label='Mask Upload', source='upload', type='numpy', tool='sketch', height=500, brush_color="#FFFFFF", mask_opacity=1, elem_id='inpaint_mask_canvas')
                                invert_mask_checkbox = gr.Checkbox(label='Invert Mask When Generating', value=modules.config.default_invert_mask_checkbox)
                                inpaint_mask_model = gr.Dropdown(label='Mask generation model',
                                                                 choices=flags.inpaint_mask_models,
                                                                 value=modules.config.default_inpaint_mask_model)
                                inpaint_mask_cloth_category = gr.Dropdown(label='Cloth category',
                                                             choices=flags.inpaint_mask_cloth_category,
                                                             value=modules.config.default_inpaint_mask_cloth_category,
                                                             visible=False)
                                inpaint_mask_dino_prompt_text = gr.Textbox(label='Detection prompt', value='', visible=False, info='Use singular whenever possible', placeholder='Describe what you want to detect.')
                                example_inpaint_mask_dino_prompt_text = gr.Dataset(
                                    samples=modules.config.example_enhance_detection_prompts,
                                    label='Detection Prompt Quick List',
                                    components=[inpaint_mask_dino_prompt_text],
                                    visible=modules.config.default_inpaint_mask_model == 'sam')
                                example_inpaint_mask_dino_prompt_text.click(lambda x: x[0],
                                                                            inputs=example_inpaint_mask_dino_prompt_text,
                                                                            outputs=inpaint_mask_dino_prompt_text,
                                                                            show_progress=False, queue=False)

                                with gr.Accordion("Advanced options", visible=False, open=False) as inpaint_mask_advanced_options:
                                    inpaint_mask_sam_model = gr.Dropdown(label='SAM model', choices=flags.inpaint_mask_sam_model, value=modules.config.default_inpaint_mask_sam_model)
                                    inpaint_mask_box_threshold = gr.Slider(label="Box Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.05)
                                    inpaint_mask_text_threshold = gr.Slider(label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.05)
                                    inpaint_mask_sam_max_detections = gr.Slider(label="Maximum number of detections", info="Set to 0 to detect all", minimum=0, maximum=10, value=modules.config.default_sam_max_detections, step=1, interactive=True)
                                generate_mask_button = gr.Button(value='Generate mask from image')

                                def generate_mask(image, mask_model, cloth_category, dino_prompt_text, sam_model, box_threshold, text_threshold, sam_max_detections, dino_erode_or_dilate, dino_debug):
                                    from extras.inpaint_mask import generate_mask_from_image

                                    extras = {}
                                    sam_options = None
                                    if mask_model == 'u2net_cloth_seg':
                                        extras['cloth_category'] = cloth_category
                                    elif mask_model == 'sam':
                                        sam_options = SAMOptions(
                                            dino_prompt=dino_prompt_text,
                                            dino_box_threshold=box_threshold,
                                            dino_text_threshold=text_threshold,
                                            dino_erode_or_dilate=dino_erode_or_dilate,
                                            dino_debug=dino_debug,
                                            max_detections=sam_max_detections,
                                            model_type=sam_model
                                        )

                                    mask, _, _, _ = generate_mask_from_image(image, mask_model, extras, sam_options)

                                    return mask


                                inpaint_mask_model.change(lambda x: [gr.update(visible=x == 'u2net_cloth_seg')] +
                                                                    [gr.update(visible=x == 'sam')] * 2 +
                                                                    [gr.Dataset.update(visible=x == 'sam',
                                                                                       samples=modules.config.example_enhance_detection_prompts)],
                                                          inputs=inpaint_mask_model,
                                                          outputs=[inpaint_mask_cloth_category,
                                                                   inpaint_mask_dino_prompt_text,
                                                                   inpaint_mask_advanced_options,
                                                                   example_inpaint_mask_dino_prompt_text],
                                                          queue=False, show_progress=False)

                    with gr.Tab(label='Describe', id='describe_tab') as describe_tab:
                        with gr.Row():
                            with gr.Column():
                                describe_input_image = grh.Image(label='Image', source='upload', type='numpy', show_label=False)
                            with gr.Column():
                                describe_methods = gr.CheckboxGroup(
                                    label='Content Type',
                                    choices=flags.describe_types,
                                    value=modules.config.default_describe_content_type)
                                describe_apply_styles = gr.Checkbox(label='Apply Styles', value=modules.config.default_describe_apply_prompts_checkbox)
                                describe_btn = gr.Button(value='Describe this Image into Prompt')
                                describe_image_size = gr.Textbox(label='Image Size and Recommended Size', elem_id='describe_image_size', visible=False)
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/1363" target="_blank">\U0001F4D4 Documentation</a>')

                                def trigger_show_image_properties(image):
                                    value = modules.util.get_image_size_info(image, modules.flags.sdxl_aspect_ratios)
                                    return gr.update(value=value, visible=True)

                                describe_input_image.upload(trigger_show_image_properties, inputs=describe_input_image,
                                                            outputs=describe_image_size, show_progress=False, queue=False)

                    with gr.Tab(label='Enhance', id='enhance_tab') as enhance_tab:
                        with gr.Row():
                            with gr.Column():
                                enhance_input_image = grh.Image(label='Use with Enhance, skips image generation', source='upload', type='numpy')
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/3281" target="_blank">\U0001F4D4 Documentation</a>')

                    with gr.Tab(label='Metadata', id='metadata_tab') as metadata_tab:
                        with gr.Column():
                            metadata_input_image = grh.Image(label='For images created by Fooocus', source='upload', type='pil')
                            metadata_json = gr.JSON(label='Metadata')
                            metadata_import_button = gr.Button(value='Apply Metadata')

                        def trigger_metadata_preview(file):
                            parameters, metadata_scheme = modules.meta_parser.read_info_from_image(file)

                            results = {}
                            if parameters is not None:
                                results['parameters'] = parameters

                            if isinstance(metadata_scheme, flags.MetadataScheme):
                                results['metadata_scheme'] = metadata_scheme.value

                            return results

                        metadata_input_image.upload(trigger_metadata_preview, inputs=metadata_input_image,
                                                    outputs=metadata_json, queue=False, show_progress=True)

            with gr.Row(visible=extra_ui.enabled_default()) as extra_plugins_panel:
                extra_ui.build_extra_panel(output_gallery=gallery)

            with gr.Row(visible=modules.config.default_enhance_checkbox) as enhance_input_panel:
                with gr.Tabs():
                    with gr.Tab(label='Upscale or Variation'):
                        with gr.Row():
                            with gr.Column():
                                enhance_uov_method = gr.Radio(label='Upscale or Variation:', choices=flags.uov_list,
                                                              value=modules.config.default_enhance_uov_method)
                                enhance_uov_processing_order = gr.Radio(label='Order of Processing',
                                                                        info='Use before to enhance small details and after to enhance large areas.',
                                                                        choices=flags.enhancement_uov_processing_order,
                                                                        value=modules.config.default_enhance_uov_processing_order)
                                # === custom-10.2: Upscaler model + Face restoration ===
                                # Wired to the SHARED apply_upscale() in async_worker, so these
                                # apply to both the Enhance upscale path AND the standalone
                                # Input Image upscale (which routes through the same function).
                                # Visibility tied to enhance_uov_method: shown only on Upscale modes.
                                _uov_models = modules.config.list_upscale_models()
                                _uov_choices = ['Fooocus Default (ESRGAN)'] + [n for n, _ in _uov_models]
                                _initial_uov_visible = isinstance(modules.config.default_enhance_uov_method, str) \
                                                       and modules.config.default_enhance_uov_method.startswith('Upscale')
                                with gr.Group(visible=_initial_uov_visible) as upscaler_extras_group:
                                    upscaler_model_name = gr.Dropdown(
                                        label='\U0001F50D Upscaler Model',
                                        choices=_uov_choices,
                                        value='Fooocus Default (ESRGAN)',
                                        info='Detected at startup. ESRGAN/RealESRGAN/SwinIR/HAT/DAT auto-recognized.'
                                    )
                                    # custom-10.4: surfaced from Developer Debug Mode
                                    overwrite_upscale_strength = gr.Slider(
                                        label='\U0001F39A Upscale Denoising Strength',
                                        minimum=-1, maximum=1.0, step=0.001,
                                        value=modules.config.default_overwrite_upscale,
                                        info='SDXL refinement pass strength after ESRGAN upscale. -1 = auto (0.382). '
                                             'Lower = preserve original detail. Higher = more creative redraw. '
                                             'Ignored on "Upscale (Fast 2x)" which skips the SDXL pass.'
                                    )
                                    face_restore_model = gr.Dropdown(
                                        label='\U0001F464 Face Restoration',
                                        choices=['Off', 'CodeFormer', 'GFPGAN v1.4'],
                                        value='Off',
                                        info='Auto-downloads on first use (~333-376 MB).'
                                    )
                                    face_restore_visibility = gr.Slider(
                                        label='Face Restoration Visibility',
                                        minimum=0.0, maximum=1.0, step=0.05, value=0.8,
                                        info='0 = no effect; 1 = full restoration. Blended with the source.'
                                    )
                                    face_restore_order = gr.Radio(
                                        label='Apply Face Restoration',
                                        choices=['Before upscale', 'After upscale'],
                                        value='After upscale',
                                        info='After upscale = A1111 standard (face restore on the high-res image).'
                                    )
                                enhance_uov_method.change(
                                    lambda m: gr.update(visible=(isinstance(m, str) and m.startswith('Upscale'))),
                                    inputs=enhance_uov_method, outputs=upscaler_extras_group,
                                    queue=False, show_progress=False
                                )
                                enhance_uov_prompt_type = gr.Radio(label='Prompt',
                                                                   info='Choose which prompt to use for Upscale or Variation.',
                                                                   choices=flags.enhancement_uov_prompt_types,
                                                                   value=modules.config.default_enhance_uov_prompt_type,
                                                                   visible=modules.config.default_enhance_uov_processing_order == flags.enhancement_uov_after)

                                enhance_uov_processing_order.change(lambda x: gr.update(visible=x == flags.enhancement_uov_after),
                                                                    inputs=enhance_uov_processing_order,
                                                                    outputs=enhance_uov_prompt_type,
                                                                    queue=False, show_progress=False)
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/3281" target="_blank">\U0001F4D4 Documentation</a>')
                    enhance_ctrls = []
                    enhance_inpaint_mode_ctrls = []
                    enhance_inpaint_engine_ctrls = []
                    enhance_inpaint_update_ctrls = []
                    for index in range(modules.config.default_enhance_tabs):
                        with gr.Tab(label=f'#{index + 1}') as enhance_tab_item:
                            enhance_enabled = gr.Checkbox(label='Enable', value=False, elem_classes='min_check',
                                                          container=False)

                            enhance_mask_dino_prompt_text = gr.Textbox(label='Detection prompt',
                                                                       info='Use singular whenever possible',
                                                                       placeholder='Describe what you want to detect.',
                                                                       interactive=True,
                                                                       visible=modules.config.default_enhance_inpaint_mask_model == 'sam')
                            example_enhance_mask_dino_prompt_text = gr.Dataset(
                                samples=modules.config.example_enhance_detection_prompts,
                                label='Detection Prompt Quick List',
                                components=[enhance_mask_dino_prompt_text],
                                visible=modules.config.default_enhance_inpaint_mask_model == 'sam')
                            example_enhance_mask_dino_prompt_text.click(lambda x: x[0],
                                                                        inputs=example_enhance_mask_dino_prompt_text,
                                                                        outputs=enhance_mask_dino_prompt_text,
                                                                        show_progress=False, queue=False)

                            enhance_prompt = gr.Textbox(label="Enhancement positive prompt",
                                                        placeholder="Uses original prompt instead if empty.",
                                                        elem_id='enhance_prompt')
                            enhance_negative_prompt = gr.Textbox(label="Enhancement negative prompt",
                                                                 placeholder="Uses original negative prompt instead if empty.",
                                                                 elem_id='enhance_negative_prompt')

                            with gr.Accordion("Detection", open=False):
                                enhance_mask_model = gr.Dropdown(label='Mask generation model',
                                                                 choices=flags.inpaint_mask_models,
                                                                 value=modules.config.default_enhance_inpaint_mask_model)
                                enhance_mask_cloth_category = gr.Dropdown(label='Cloth category',
                                                                          choices=flags.inpaint_mask_cloth_category,
                                                                          value=modules.config.default_inpaint_mask_cloth_category,
                                                                          visible=modules.config.default_enhance_inpaint_mask_model == 'u2net_cloth_seg',
                                                                          interactive=True)

                                with gr.Accordion("SAM Options",
                                                  visible=modules.config.default_enhance_inpaint_mask_model == 'sam',
                                                  open=False) as sam_options:
                                    enhance_mask_sam_model = gr.Dropdown(label='SAM model',
                                                                         choices=flags.inpaint_mask_sam_model,
                                                                         value=modules.config.default_inpaint_mask_sam_model,
                                                                         interactive=True)
                                    enhance_mask_box_threshold = gr.Slider(label="Box Threshold", minimum=0.0,
                                                                           maximum=1.0, value=0.3, step=0.05,
                                                                           interactive=True)
                                    enhance_mask_text_threshold = gr.Slider(label="Text Threshold", minimum=0.0,
                                                                            maximum=1.0, value=0.25, step=0.05,
                                                                            interactive=True)
                                    enhance_mask_sam_max_detections = gr.Slider(label="Maximum number of detections",
                                                                                info="Set to 0 to detect all",
                                                                                minimum=0, maximum=10,
                                                                                value=modules.config.default_sam_max_detections,
                                                                                step=1, interactive=True)

                            with gr.Accordion("Inpaint", visible=True, open=False):
                                enhance_inpaint_mode = gr.Dropdown(choices=modules.flags.inpaint_options,
                                                                   value=modules.config.default_inpaint_method,
                                                                   label='Method', interactive=True)
                                enhance_inpaint_disable_initial_latent = gr.Checkbox(
                                    label='Disable initial latent in inpaint', value=False)
                                enhance_inpaint_engine = gr.Dropdown(label='Inpaint Engine',
                                                                     value=modules.config.default_inpaint_engine_version,
                                                                     choices=flags.inpaint_engine_versions,
                                                                     info='Version of Fooocus inpaint model. If set, use performance Quality or Speed (no performance LoRAs) for best results.')
                                enhance_inpaint_strength = gr.Slider(label='Inpaint Denoising Strength',
                                                                     minimum=0.0, maximum=1.0, step=0.001,
                                                                     value=1.0,
                                                                     info='Same as the denoising strength in A1111 inpaint. '
                                                                          'Only used in inpaint, not used in outpaint. '
                                                                          '(Outpaint always use 1.0)')
                                enhance_inpaint_respective_field = gr.Slider(label='Inpaint Respective Field',
                                                                             minimum=0.0, maximum=1.0, step=0.001,
                                                                             value=0.618,
                                                                             info='The area to inpaint. '
                                                                                  'Value 0 is same as "Only Masked" in A1111. '
                                                                                  'Value 1 is same as "Whole Image" in A1111. '
                                                                                  'Only used in inpaint, not used in outpaint. '
                                                                                  '(Outpaint always use 1.0)')
                                enhance_inpaint_erode_or_dilate = gr.Slider(label='Mask Erode or Dilate',
                                                                            minimum=-64, maximum=64, step=1, value=0,
                                                                            info='Positive value will make white area in the mask larger, '
                                                                                 'negative value will make white area smaller. '
                                                                                 '(default is 0, always processed before any mask invert)')
                                enhance_mask_invert = gr.Checkbox(label='Invert Mask', value=False)

                            gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/3281" target="_blank">\U0001F4D4 Documentation</a>')

                        enhance_ctrls += [
                            enhance_enabled,
                            enhance_mask_dino_prompt_text,
                            enhance_prompt,
                            enhance_negative_prompt,
                            enhance_mask_model,
                            enhance_mask_cloth_category,
                            enhance_mask_sam_model,
                            enhance_mask_text_threshold,
                            enhance_mask_box_threshold,
                            enhance_mask_sam_max_detections,
                            enhance_inpaint_disable_initial_latent,
                            enhance_inpaint_engine,
                            enhance_inpaint_strength,
                            enhance_inpaint_respective_field,
                            enhance_inpaint_erode_or_dilate,
                            enhance_mask_invert
                        ]

                        enhance_inpaint_mode_ctrls += [enhance_inpaint_mode]
                        enhance_inpaint_engine_ctrls += [enhance_inpaint_engine]

                        enhance_inpaint_update_ctrls += [[
                            enhance_inpaint_mode, enhance_inpaint_disable_initial_latent, enhance_inpaint_engine,
                            enhance_inpaint_strength, enhance_inpaint_respective_field
                        ]]

                        enhance_inpaint_mode.change(inpaint_mode_change, inputs=[enhance_inpaint_mode, inpaint_engine_state], outputs=[
                            inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
                            enhance_inpaint_disable_initial_latent, enhance_inpaint_engine,
                            enhance_inpaint_strength, enhance_inpaint_respective_field
                        ], show_progress=False, queue=False)

                        enhance_mask_model.change(
                            lambda x: [gr.update(visible=x == 'u2net_cloth_seg')] +
                                      [gr.update(visible=x == 'sam')] * 2 +
                                      [gr.Dataset.update(visible=x == 'sam',
                                                         samples=modules.config.example_enhance_detection_prompts)],
                            inputs=enhance_mask_model,
                            outputs=[enhance_mask_cloth_category, enhance_mask_dino_prompt_text, sam_options,
                                     example_enhance_mask_dino_prompt_text],
                            queue=False, show_progress=False)

            switch_js = "(x) => {if(x){viewer_to_bottom(100);viewer_to_bottom(500);}else{viewer_to_top();} return x;}"
            down_js = "() => {viewer_to_bottom();}"

            input_image_checkbox.change(lambda x: gr.update(visible=x), inputs=input_image_checkbox,
                                        outputs=image_input_panel, queue=False, show_progress=False, _js=switch_js)
            ip_advanced.change(lambda: None, queue=False, show_progress=False, _js=down_js)

            current_tab = gr.Textbox(value='uov', visible=False)
            uov_tab.select(lambda: 'uov', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            inpaint_tab.select(lambda: 'inpaint', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            ip_tab.select(lambda: 'ip', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            describe_tab.select(lambda: 'desc', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            enhance_tab.select(lambda: 'enhance', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            metadata_tab.select(lambda: 'metadata', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            enhance_checkbox.change(lambda x: gr.update(visible=x), inputs=enhance_checkbox,
                                        outputs=enhance_input_panel, queue=False, show_progress=False, _js=switch_js)

        with gr.Column(scale=1, visible=modules.config.default_advanced_checkbox) as advanced_column:
            with gr.Tab(label='Settings'):
                if not args_manager.args.disable_preset_selection:
                    preset_selection = gr.Dropdown(label='Preset',
                                                   choices=modules.config.available_presets,
                                                   value=args_manager.args.preset if args_manager.args.preset else "initial",
                                                   interactive=True)

                performance_selection = gr.Radio(label='Performance',
                                                 choices=flags.Performance.values(),
                                                 value=modules.config.default_performance,
                                                 elem_classes=['performance_selection'])

                with gr.Accordion(label='Aspect Ratios', open=False, elem_id='aspect_ratios_accordion') as aspect_ratios_accordion:
                    # custom-7: Aspect Ratios as a Dropdown with 'Custom' as the first
                    # option. add_ratio() now returns plain text (no <span>), so we can
                    # pass the labels directly as Dropdown string choices — Gradio 3.41
                    # does not support (label, value) tuples cleanly, so plain strings
                    # are required.
                    _CUSTOM_AR_SENTINEL = 'Custom'
                    _ar_choices = [_CUSTOM_AR_SENTINEL] + list(modules.config.available_aspect_ratios_labels)
                    aspect_ratios_selection = gr.Dropdown(
                        label='Aspect Ratios', show_label=False,
                        choices=_ar_choices,
                        value=modules.config.default_aspect_ratio,
                        info='Pick a preset, or "Custom" to use the inputs below.',
                        elem_classes='aspect_ratios',
                        interactive=True)

                    aspect_ratios_selection.change(lambda x: None, inputs=aspect_ratios_selection, queue=False, show_progress=False, _js='(x)=>{refresh_aspect_ratios_label(x);}')
                    shared.gradio_root.load(lambda x: None, inputs=aspect_ratios_selection, queue=False, show_progress=False, _js='(x)=>{refresh_aspect_ratios_label(x);}')

                    use_aspect_for_vary = gr.Checkbox(
                        label='Use selected Aspect Ratio for Vary (crop input to fit)',
                        value=False,
                        info='When enabled, Vary (Subtle/Strong) outputs use the Aspect Ratio above instead '
                             'of the input image\u2019s native size. The input is centre-cropped and resized to fit. '
                             'Does not affect Upscale (which keeps its fixed factor).')

                    # === custom-7: Custom Resolution =====================================
                    # custom_res_enabled is hidden — its value is now driven by the
                    # 'Custom' selection in the Aspect Ratios dropdown above. Kept in the
                    # ctrls list so the worker still receives the explicit flag.
                    custom_res_enabled = gr.Checkbox(value=False, visible=False)
                    with gr.Column(visible=False) as custom_res_panel:
                        with gr.Row():
                            custom_ratio_w = gr.Number(label='Ratio W', value=16, precision=0,
                                                        minimum=1, maximum=9999, scale=1)
                            custom_ratio_h = gr.Number(label='Ratio H', value=9, precision=0,
                                                        minimum=1, maximum=9999, scale=1)
                            custom_res_swap = gr.Button(value='\U0001F504 Swap', scale=0)
                        custom_res_mode = gr.Radio(
                            label='Mode',
                            choices=['Max edge', '~1 MP target', 'Min edge'],
                            value='Max edge',
                            info='Max/Min edge sizes the longer/shorter side; ~1 MP keeps total area near size\u00b2.')
                        custom_res_size = gr.Slider(
                            label='Size (px, snapped to /64)',
                            minimum=512, maximum=2048, step=64, value=1024)
                        custom_res_display = gr.HTML(value='', elem_id='custom_res_display')
                        gr.HTML('<div style="font-size:11px;color:#888;margin-top:-4px;">Quick ratios:</div>')
                        # 3-column grid, 2 rows; small buttons with tight min-width.
                        _ratio_chips = [('1:1', 1, 1), ('3:2', 3, 2), ('4:3', 4, 3),
                                         ('16:9', 16, 9), ('21:9', 21, 9), ('\u221a2 (A4)', 1000, 1414)]
                        _chip_buttons = []
                        for _i in range(0, len(_ratio_chips), 3):
                            with gr.Row():
                                for _label, _rw, _rh in _ratio_chips[_i:_i + 3]:
                                    _btn = gr.Button(value=_label, size='sm', min_width=50)
                                    _chip_buttons.append((_btn, _rw, _rh))
                        custom_res_save_entry = gr.Button(
                            value='\U0001F4BE Save as preset entry (config.txt)',
                            variant='secondary')
                        custom_res_save_status = gr.HTML(value='')

                    def _format_custom_res_display(rw, rh, mode, size):
                        try:
                            from modules.util import compute_custom_wh
                            import math as _math
                            ww, hh = compute_custom_wh(rw, rh, mode, size)
                            mp = (ww * hh) / 1_000_000.0
                            rw_i = max(1, int(round(float(rw or 1))))
                            rh_i = max(1, int(round(float(rh or 1))))
                            g = _math.gcd(rw_i, rh_i)
                            warn = ''
                            if mp < 0.25:
                                warn = ' &middot; <span style="color:#c80;">low MP, quality may drop</span>'
                            elif mp > 2.0:
                                warn = ' &middot; <span style="color:#c80;">high MP, may OOM</span>'
                            return (f'<div style="margin:4px 0;font-size:13px;">\u2192 '
                                    f'<b>{ww} \u00d7 {hh}</b> &middot; {mp:.2f} MP &middot; '
                                    f'{rw_i // g}:{rh_i // g}{warn}</div>')
                        except Exception as _e:
                            return f'<div style="color:#c66;">Invalid input: {_e}</div>'

                    # Drive custom_res_enabled + panel visibility from the Aspect Ratios
                    # dropdown: selecting 'Custom' shows the panel and flags the worker.
                    def _on_aspect_dropdown_change(value):
                        is_custom = (str(value).strip() == _CUSTOM_AR_SENTINEL)
                        return gr.update(value=is_custom), gr.update(visible=is_custom)

                    aspect_ratios_selection.change(
                        _on_aspect_dropdown_change,
                        inputs=aspect_ratios_selection,
                        outputs=[custom_res_enabled, custom_res_panel],
                        queue=False, show_progress=False)
                    shared.gradio_root.load(
                        _on_aspect_dropdown_change,
                        inputs=aspect_ratios_selection,
                        outputs=[custom_res_enabled, custom_res_panel],
                        queue=False, show_progress=False)

                    _custom_res_compute_inputs = [custom_ratio_w, custom_ratio_h,
                                                   custom_res_mode, custom_res_size]
                    for _comp in _custom_res_compute_inputs:
                        _comp.change(_format_custom_res_display,
                                     inputs=_custom_res_compute_inputs,
                                     outputs=custom_res_display,
                                     queue=False, show_progress=False)
                    shared.gradio_root.load(_format_custom_res_display,
                                             inputs=_custom_res_compute_inputs,
                                             outputs=custom_res_display,
                                             queue=False, show_progress=False)

                    custom_res_swap.click(
                        lambda w, h: (h, w),
                        inputs=[custom_ratio_w, custom_ratio_h],
                        outputs=[custom_ratio_w, custom_ratio_h],
                        queue=False, show_progress=False)

                    for _btn, _rw, _rh in _chip_buttons:
                        _btn.click(
                            (lambda rw=_rw, rh=_rh: (rw, rh)),
                            inputs=[],
                            outputs=[custom_ratio_w, custom_ratio_h],
                            queue=False, show_progress=False)

                    def _save_custom_as_entry(rw, rh, mode, size):
                        try:
                            from modules.util import compute_custom_wh
                            ww, hh = compute_custom_wh(rw, rh, mode, size)
                            new_entry = f'{ww}*{hh}'
                            cfg = modules.config.config_dict
                            existing = list(cfg.get('available_aspect_ratios',
                                                     modules.config.available_aspect_ratios))
                            if new_entry in existing:
                                return (f'<span style="color:#aaa;">\u2139 {new_entry} already in list. '
                                        f'Restart to use it from the Aspect Ratios block.</span>')
                            existing.append(new_entry)
                            cfg['available_aspect_ratios'] = existing
                            try:
                                with open(modules.config.config_path, 'w', encoding='utf-8') as f:
                                    json.dump(cfg, f, indent=4, ensure_ascii=False)
                            except Exception as e:
                                return f'<span style="color:#c66;">Could not write config.txt: {e}</span>'
                            return (f'<span style="color:#4ecdc4;">\u2713 Added {new_entry} to '
                                    f'available_aspect_ratios. Restart Fooocus to see it in the radio block.</span>')
                        except Exception as e:
                            return f'<span style="color:#c66;">Error: {e}</span>'

                    custom_res_save_entry.click(
                        _save_custom_as_entry,
                        inputs=_custom_res_compute_inputs,
                        outputs=custom_res_save_status,
                        queue=False, show_progress=False)
                    # === end custom-7 ====================================================

                image_number = gr.Slider(label='Image Number', minimum=1, maximum=modules.config.default_max_image_number, step=1, value=modules.config.default_image_number)

                output_format = gr.Radio(label='Output Format',
                                         choices=flags.OutputFormat.list(),
                                         value=modules.config.default_output_format)

                with gr.Accordion(label='\U0001F4BE Preset Manager', open=False):
                    gr.HTML('<div style="font-size:12px;color:#888;margin-bottom:4px;">Save the current settings (prompts, styles, LoRAs, embeddings, samplers\u2026) as a reusable preset.</div>')
                    with gr.Row():
                        save_preset_name = gr.Textbox(label='New Preset Name', placeholder='my_custom_preset',
                                                      value='', scale=3)
                        save_preset_overwrite_dropdown = gr.Dropdown(
                            label='Or Overwrite Existing',
                            choices=modules.config.get_user_presets(),
                            value=None, scale=3)
                    with gr.Row():
                        save_preset_new_btn = gr.Button(value='\U0001F4BE Save as New Preset',
                                                        variant='secondary', scale=1)
                        save_preset_overwrite_btn = gr.Button(value='\U0000270F Overwrite Selected',
                                                               variant='stop', scale=1)
                    save_preset_status = gr.HTML(value='', visible=True)

                seed_random = gr.Checkbox(label='Random', value=True)
                image_seed = gr.Textbox(label='Seed', value=0, max_lines=1, visible=False) # workaround for https://github.com/gradio-app/gradio/issues/5354

                def random_checked(r):
                    return gr.update(visible=not r)

                def refresh_seed(r, seed_string):
                    if r:
                        return random.randint(constants.MIN_SEED, constants.MAX_SEED)
                    else:
                        try:
                            seed_value = int(seed_string)
                            if constants.MIN_SEED <= seed_value <= constants.MAX_SEED:
                                return seed_value
                        except ValueError:
                            pass
                        return random.randint(constants.MIN_SEED, constants.MAX_SEED)

                seed_random.change(random_checked, inputs=[seed_random], outputs=[image_seed],
                                   queue=False, show_progress=False)

                def update_history_link():
                    if args_manager.args.disable_image_log:
                        return gr.update(value='')

                    return gr.update(value=f'<a href="file={get_current_html_path(output_format)}" target="_blank">\U0001F4DA History Log</a>')

                # custom-8: Asset Browser shortcut. Enabled = clickable link;
                # disabled = greyed-out hint pointing to the toggle in Advanced.
                def update_browser_link():
                    if args_manager.args.disable_image_log:
                        return gr.update(value='')
                    if not modules.config.asset_browser_enabled():
                        return gr.update(value='<span style="margin-left:12px;color:#888;" '
                                                'title="Enable in Advanced > Models > Asset Browser, then Restart UI">'
                                                '\U0001F5BC️ Asset Browser (disabled)</span>')
                    # Lazy-create outputs/index.html on first link render so the user
                    # never sees a 404 just because they haven't generated yet.
                    try:
                        from modules.gallery_writer import ensure_gallery_assets
                        ensure_gallery_assets()
                    except Exception:
                        pass
                    abs_index = os.path.abspath(os.path.join(modules.config.path_outputs, 'index.html'))
                    return gr.update(value=f'<a href="file={abs_index}" target="_blank" '
                                            f'style="margin-left:12px;">\U0001F5BC️ Asset Browser</a>')

                with gr.Row():
                    history_link = gr.HTML()
                    browser_link = gr.HTML()
                shared.gradio_root.load(update_history_link, outputs=history_link, queue=False, show_progress=False)
                shared.gradio_root.load(update_browser_link, outputs=browser_link, queue=False, show_progress=False)

            with gr.Tab(label='Styles', elem_classes=['style_selections_tab']):
                style_sorter.try_load_sorted_styles(
                    style_names=legal_style_names,
                    default_selected=modules.config.default_styles)

                style_search_bar = gr.Textbox(show_label=False, container=False,
                                              placeholder="\U0001F50E Type here to search styles ...",
                                              value="",
                                              label='Search Styles')
                style_selections = gr.CheckboxGroup(show_label=False, container=False,
                                                    choices=copy.deepcopy(style_sorter.all_styles),
                                                    value=copy.deepcopy(modules.config.default_styles),
                                                    label='Selected Styles',
                                                    elem_classes=['style_selections'])
                gradio_receiver_style_selections = gr.Textbox(elem_id='gradio_receiver_style_selections', visible=False)

                shared.gradio_root.load(lambda: gr.update(choices=copy.deepcopy(style_sorter.all_styles)),
                                        outputs=style_selections)

                style_search_bar.change(style_sorter.search_styles,
                                        inputs=[style_selections, style_search_bar],
                                        outputs=style_selections,
                                        queue=False,
                                        show_progress=False).then(
                    lambda: None, _js='()=>{refresh_style_localization();}')

                gradio_receiver_style_selections.input(style_sorter.sort_styles,
                                                       inputs=style_selections,
                                                       outputs=style_selections,
                                                       queue=False,
                                                       show_progress=False).then(
                    lambda: None, _js='()=>{refresh_style_localization();}')

            with gr.Tab(label='Models'):
                with gr.Group():
                    with gr.Row():
                        base_model = gr.Dropdown(label='Base Model (SDXL only)', choices=modules.config.model_filenames, value=modules.config.default_base_model_name, show_label=True)
                        refiner_model = gr.Dropdown(label='Refiner (SDXL or SD 1.5)', choices=['None'] + modules.config.model_filenames, value=modules.config.default_refiner_model_name, show_label=True)

                    refiner_switch = gr.Slider(label='Refiner Switch At', minimum=0.1, maximum=1.0, step=0.0001,
                                               info='Use 0.4 for SD1.5 realistic models; '
                                                    'or 0.667 for SD1.5 anime models; '
                                                    'or 0.8 for XL-refiners; '
                                                    'or any value for switching two SDXL models.',
                                               value=modules.config.default_refiner_switch,
                                               visible=modules.config.default_refiner_model_name != 'None')

                    refiner_model.change(lambda x: gr.update(visible=x != 'None'),
                                         inputs=refiner_model, outputs=refiner_switch, show_progress=False, queue=False)

                with gr.Accordion(label='\U0001F3A8 CivitAI Model Settings', open=False):
                    _civitai_key = modules.config.civitai_api_key
                    _civitai_key_display = (f'{_civitai_key[:4]}...{_civitai_key[-4:]}' if len(_civitai_key) > 8 else _civitai_key)
                    _civitai_key_status = ' (saved)' if _civitai_key else ''
                    with gr.Row():
                        civitai_api_key_input = gr.Textbox(
                            label=f'CivitAI API Key{_civitai_key_status}',
                            value=_civitai_key_display if _civitai_key else '',
                            placeholder='Enter your CivitAI API key...',
                            scale=4)
                        civitai_save_key_btn = gr.Button(value='\U0001F4BE Save Key', variant='secondary', scale=1)
                    with gr.Row():
                        civitai_fetch_btn = gr.Button(
                            value='\U0001F50D Fetch CivitAI Settings',
                            variant='secondary', scale=2)
                        civitai_apply_btn = gr.Button(
                            value='\U00002705 Apply These Settings',
                            variant='primary', scale=2, visible=False)
                        civitai_refresh_btn = gr.Button(
                            value='\U0001F504 Refresh from CivitAI',
                            variant='secondary', scale=2, visible=False)

                    civitai_panel = gr.HTML(
                        value='<div style="padding:8px;border:1px solid #444;border-radius:8px;color:#888;">'
                              'Select a model and click "Fetch CivitAI Settings" to get community recommendations.</div>',
                        visible=True)

                    # Hidden state to store raw settings for the Apply button
                    civitai_settings_state = gr.State(value=None)
                    # Hidden state to store triggers + model_info for the Copy/Save buttons
                    civitai_model_info_state = gr.State(value=None)

                    with gr.Row():
                        civitai_copy_triggers_btn = gr.Button(
                            value='\U0001F4CB Copy checkpoint triggers to prompt',
                            variant='secondary', scale=3, visible=False)
                    with gr.Row():
                        civitai_preset_name = gr.Textbox(
                            show_label=False,
                            placeholder='Preset name for CivitAI consensus (e.g. civitai_<modelname>)',
                            value='', scale=3, container=False, visible=False)
                        civitai_save_preset_btn = gr.Button(
                            value='\U0001F4BE Save CivitAI consensus as preset',
                            variant='primary', scale=2, visible=False)
                    civitai_save_preset_status = gr.HTML(value='', visible=False)

                with gr.Accordion(label='\U0001F9EC LoRA', open=True):
                    lora_ctrls = []
                    lora_model_dropdowns = []
                    lora_trigger_displays = []
                    lora_copy_btns = []

                    for i, (enabled, filename, weight) in enumerate(modules.config.default_loras):
                        with gr.Row():
                            lora_enabled = gr.Checkbox(label='Enable', value=enabled,
                                                       elem_classes=['lora_enable', 'min_check'], scale=1)
                            lora_model = gr.Dropdown(label=f'LoRA {i + 1}',
                                                     choices=['None'] + modules.config.lora_filenames, value=filename,
                                                     elem_classes='lora_model', scale=5)
                            lora_weight = gr.Slider(label='Weight', minimum=modules.config.default_loras_min_weight,
                                                    maximum=modules.config.default_loras_max_weight, step=0.01, value=weight,
                                                    elem_classes='lora_weight', scale=5)
                            lora_ctrls += [lora_enabled, lora_model, lora_weight]
                            lora_model_dropdowns.append(lora_model)

                        with gr.Row():
                            lora_trigger_display = gr.Textbox(
                                show_label=False, value='', interactive=False,
                                placeholder=f'LoRA {i + 1} trigger words (auto-fetched from CivitAI)',
                                elem_classes='lora_triggers', scale=10, container=False)
                            lora_copy_btn = gr.Button(
                                value='\U0001F4CB Copy to prompt', size='sm',
                                variant='secondary', scale=2)
                            lora_trigger_displays.append(lora_trigger_display)
                            lora_copy_btns.append(lora_copy_btn)

                    with gr.Row():
                        lora_copy_all_btn = gr.Button(
                            value='\U0001F4CB Copy ALL active LoRA triggers to prompt',
                            variant='secondary', size='sm')

                # === Textual Inversion / Embeddings ===
                with gr.Accordion(label='\U0001F9E9 Textual Inversion / Embeddings', open=False):
                    embedding_ctrls = []
                    embedding_dropdowns = []
                    embedding_weights = []
                    embedding_trigger_displays = []
                    embedding_insert_prompt_btns = []
                    embedding_insert_negative_btns = []

                    gr.HTML('<div style="font-size:12px;color:#888;padding:2px 4px;">'
                            'The checkbox only filters the <b>Insert ALL active</b> bulk button \u2014 '
                            'per-slot <b>Prompt</b> / <b>Negative</b> buttons always work. '
                            'Embeddings activate purely from their token in the text, not from any enable flag.'
                            '</div>')

                    _embedding_default_count = 5
                    for i in range(_embedding_default_count):
                        with gr.Row():
                            emb_enabled = gr.Checkbox(
                                label='Include', value=(i == 0),
                                elem_classes=['emb_enable', 'min_check'], scale=1)
                            emb_model = gr.Dropdown(
                                label=f'Embedding {i + 1}',
                                choices=['None'] + modules.config.embedding_filenames,
                                value='None',
                                elem_classes='emb_model', scale=5)
                            emb_weight = gr.Slider(
                                label='Weight', minimum=0.1, maximum=3.0,
                                step=0.01, value=1.0,
                                elem_classes='emb_weight', scale=5)
                            embedding_ctrls += [emb_enabled, emb_model, emb_weight]
                            embedding_dropdowns.append(emb_model)
                            embedding_weights.append(emb_weight)

                        with gr.Row():
                            emb_trigger_display = gr.Textbox(
                                show_label=False, value='', interactive=False,
                                placeholder=f'Embedding {i + 1} activation token (auto-detected)',
                                elem_classes='emb_triggers', scale=6, container=False)
                            emb_insert_prompt_btn = gr.Button(
                                value='\U0001F4CB Prompt',
                                variant='secondary', scale=2, min_width=90)
                            emb_insert_negative_btn = gr.Button(
                                value='\U0001F4CB Negative',
                                variant='secondary', scale=2, min_width=90)
                            embedding_trigger_displays.append(emb_trigger_display)
                            embedding_insert_prompt_btns.append(emb_insert_prompt_btn)
                            embedding_insert_negative_btns.append(emb_insert_negative_btn)

                    with gr.Row():
                        emb_insert_all_prompt_btn = gr.Button(
                            value='\U0001F4CB Insert ALL active embeddings to prompt',
                            variant='secondary', size='sm', scale=1)
                        emb_insert_all_negative_btn = gr.Button(
                            value='\U0001F4CB Insert ALL active embeddings to negative',
                            variant='secondary', size='sm', scale=1)

                # === Wildcards ===
                with gr.Accordion(label='\U0001F3B2 Wildcards', open=False):
                    with gr.Row():
                        wildcard_dropdown = gr.Dropdown(
                            label='Wildcard file',
                            choices=['None'] + modules.config.wildcard_filenames,
                            value='None', scale=5)
                        wildcard_insert_btn = gr.Button(
                            value='\U0001F4CB Insert __token__ to prompt',
                            variant='secondary', scale=2, min_width=90)
                    wildcard_editor = gr.Textbox(
                        label='Contents (one entry per line, edit freely)',
                        value='', lines=12, max_lines=30,
                        interactive=True,
                        placeholder='Select a wildcard file above to edit its contents, '
                                    'or type a new name below and click Create to start a new file.')
                    with gr.Row():
                        wildcard_save_btn = gr.Button(
                            value='\U0001F4BE Save', variant='secondary', scale=1, min_width=90)
                        wildcard_new_name = gr.Textbox(
                            show_label=False,
                            placeholder='name_for_new_wildcard (no extension)',
                            value='', scale=4, container=False)
                        wildcard_create_btn = gr.Button(
                            value='\U00002795 Create new', variant='primary', scale=1, min_width=90)
                    wildcard_status = gr.HTML(value='', visible=True)
                    gr.HTML('<div style="font-size:12px;color:#888;padding:2px 4px;">'
                            'Wildcards expand to a random line at generation time. '
                            'Token format: <code>__filename__</code> (without the .txt extension). '
                            'Save writes back to the selected file; Create makes a new <code>.txt</code> '
                            'in the wildcards folder using the current contents.'
                            '</div>')

                # === Layout / Omost ============================================
                # On by default (omost.enabled). When disabled, the accordion is
                # not created at all : no import, no thread, no network call.
                _omost_cfg = getattr(modules.config, 'omost_config', {})
                _omost_enabled = bool(_omost_cfg.get('enabled', False))
                if _omost_enabled:
                    import modules.omost_client as _omost_client
                    with gr.Accordion(label='\U0001F9ED Layout / Omost', open=False,
                                      elem_id='omost_accordion'):
                        gr.HTML('<div style="font-size:12px;color:#888;margin-bottom:6px;">'
                                'Turn a short idea into a structured layout (Omost Canvas DSL) '
                                'via a local LLM, then flatten it into an injectable SDXL prompt. '
                                'Requires Ollama with the Omost model loaded. '
                                'Endpoint, model and timeout are set in config.txt.'
                                '</div>')
                        omost_idea = gr.Textbox(
                            label='\U0001F4A1 Idea', lines=1,
                            placeholder='e.g. amira sato-chan in a neon tokyo alley at night')
                        omost_generate_btn = gr.Button(
                            value='\U0001F9ED Generate layout', variant='primary')
                        omost_layout_box = gr.Textbox(
                            label='Raw layout (JSON, stored for a future V2)',
                            value='', lines=10, max_lines=24, interactive=False)
                        omost_prompt_box = gr.Textbox(
                            label='Proposed flattened prompt',
                            value='', lines=4, max_lines=12, interactive=False)
                        omost_inject_btn = gr.Button(
                            value='\U00002795 Inject into prompt', variant='secondary')

                with gr.Row():
                    refresh_files = gr.Button(label='Refresh', value='\U0001f504 Refresh All Files', variant='secondary', elem_classes='refresh_button', scale=3)
                gr.HTML('<div style="font-size:11px;color:#888;padding:2px 6px;">'
                        'Use Refresh to pick up new files (models, LoRAs, wildcards). '
                        'The Restart UI button now lives at the end of the Advanced tab.</div>')
            with gr.Tab(label='Advanced'):
                guidance_scale = gr.Slider(label='Guidance Scale', minimum=1.0, maximum=30.0, step=0.01,
                                           value=modules.config.default_cfg_scale,
                                           info='Higher value means style is cleaner, vivider, and more artistic.')
                sharpness = gr.Slider(label='Image Sharpness', minimum=0.0, maximum=30.0, step=0.001,
                                      value=modules.config.default_sample_sharpness,
                                      info='Higher value means image and texture are sharper.')
                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/117" target="_blank">\U0001F4D4 Documentation</a>')

                # === custom-8: Asset Browser settings ============================
                _ab_cfg = modules.config.asset_browser_config
                _ab_state_label = '(off)' if not _ab_cfg.get('enabled') else '(on)'
                with gr.Accordion(label=f'\U0001F5BC️ Asset Browser {_ab_state_label}',
                                   open=False, elem_id='asset_browser_accordion'):
                    gr.HTML('<div style="font-size:12px;color:#888;margin-bottom:6px;">'
                            'Standalone HTML gallery for outputs + LoRAs/Checkpoints/Embeddings previews. '
                            'Opens in a new browser tab. <b>ON by default</b>: when disabled (asset_browser.enabled=false) this feature '
                            'has &lt;1µs overhead and never touches Fooocus generation.'
                            '</div>')
                    ab_enabled = gr.Checkbox(
                        label='Enable Asset Browser',
                        value=bool(_ab_cfg.get('enabled', True)),
                        info='Master switch. Restart UI required for full effect.')
                    with gr.Group():
                        ab_thumbs = gr.Checkbox(
                            label='Generate thumbnails on save (~10ms / image)',
                            value=bool(_ab_cfg.get('generate_thumbnails', True)))
                        ab_dzi = gr.Radio(
                            label='Deep-zoom tiles for big images',
                            choices=['auto', 'always', 'never'],
                            value=str(_ab_cfg.get('generate_dzi_tiles', 'auto')),
                            info="'auto' tiles only images > 4 MP (~150ms when triggered).")
                        ab_index_boot = gr.Checkbox(
                            label='Index models on startup (~2-5s in background thread)',
                            value=bool(_ab_cfg.get('index_models_on_boot', True)))
                        ab_blur = gr.Checkbox(
                            label='\U0001F32B Blur thumbnails by default (NSFW privacy — hover/click to reveal)',
                            value=bool(_ab_cfg.get('blur_thumbnails', False)),
                            info='Sets the SPA default. Users can toggle on/off temporarily from the Browser header (sticky in browser localStorage).')
                    with gr.Row():
                        ab_save_btn = gr.Button(value='\U0001F4BE Save settings',
                                                 variant='primary', scale=1)
                        ab_reindex_btn = gr.Button(value='\U0001F504 Reindex everything',
                                                    variant='secondary', scale=1,
                                                    elem_id='ab-reindex-btn')
                        ab_reindex_smart_btn = gr.Button(
                            value='\U0001F9E0 Reindex missing only',
                            variant='secondary', scale=1,
                            elem_id='ab-reindex-smart-btn')
                    ab_status = gr.HTML(value='', elem_id='ab-status')
                    if _ab_cfg.get('enabled'):
                        try:
                            from modules.gallery_writer import ensure_gallery_assets as _ab_ensure
                            _ab_ensure()
                        except Exception:
                            pass
                        _ab_abs_index = os.path.abspath(os.path.join(modules.config.path_outputs, 'index.html'))
                        _ab_open_html = (f'<a href="file={_ab_abs_index}" target="_blank">'
                                          '\U0001F5BC️ Open Asset Browser</a>')
                    else:
                        _ab_open_html = ('<span style="color:#888;">'
                                          '\U0001F5BC️ Asset Browser (disabled — enable above and Save)</span>')
                    ab_open_link = gr.HTML(value=_ab_open_html)
                    gr.HTML('<div style="font-size:11px;color:#888;margin-top:4px;">'
                            'Outputs gallery + LoRAs / Checkpoints / Embeddings tabs · '
                            'PhotoSwipe v5 lightbox · Dynamic Caption sidebar · '
                            'click-to-zoom · Copy buttons. '
                            'Generated artifacts under <code>outputs/_index</code> + '
                            '<code>outputs/_previews</code> + <code>outputs/_assets</code> '
                            '(all gitignored).'
                            '</div>')

                # === custom-10: Upscale & Face Restoration paths ================
                with gr.Accordion(label='\U0001F50D Upscale & Face Restoration Paths',
                                   open=False, elem_id='upscale_paths_accordion'):
                    gr.HTML('<div style="font-size:12px;color:#888;margin-bottom:6px;">'
                            'Optional A1111-compatible folders. Each folder is scanned for upscale / face restoration '
                            'models in addition to the Fooocus default folders. Leave empty to disable a path. '
                            '<b>Restart UI</b> after saving for the dropdowns to refresh.'
                            '</div>')
                    up_path_esrgan = gr.Textbox(label='ESRGAN folder (path_esrgan)',
                                                 value=modules.config.path_esrgan,
                                                 placeholder='e.g. D:\\stable-diffusion-webui\\models\\ESRGAN')
                    up_path_realesrgan = gr.Textbox(label='RealESRGAN folder (path_realesrgan)',
                                                     value=modules.config.path_realesrgan,
                                                     placeholder='e.g. D:\\stable-diffusion-webui\\models\\RealESRGAN')
                    up_path_swinir = gr.Textbox(label='SwinIR folder (path_swinir)',
                                                 value=modules.config.path_swinir,
                                                 placeholder='e.g. D:\\stable-diffusion-webui\\models\\SwinIR')
                    up_path_dat = gr.Textbox(label='DAT folder (path_dat)',
                                              value=modules.config.path_dat,
                                              placeholder='e.g. D:\\stable-diffusion-webui\\models\\DAT')
                    up_path_gfpgan = gr.Textbox(label='GFPGAN folder (path_gfpgan)',
                                                 value=modules.config.path_gfpgan,
                                                 placeholder='e.g. D:\\stable-diffusion-webui\\models\\GFPGAN')
                    up_path_codeformer = gr.Textbox(label='CodeFormer folder (path_codeformer)',
                                                     value=modules.config.path_codeformer,
                                                     placeholder='e.g. D:\\stable-diffusion-webui\\models\\CodeFormer')
                    up_path_save_btn = gr.Button(value='\U0001F4BE Save paths',
                                                  variant='primary')
                    up_path_status = gr.HTML(value='', elem_id='upscale-paths-status')

                    def _save_upscale_paths(p_esrgan, p_realesrgan, p_swinir, p_dat, p_gfpgan, p_codeformer):
                        try:
                            import json as _json
                            cfg_path = modules.config.config_path
                            try:
                                with open(cfg_path, 'r', encoding='utf-8') as f:
                                    cfg = _json.load(f)
                            except Exception:
                                cfg = {}
                            cfg['path_esrgan'] = (p_esrgan or '').strip()
                            cfg['path_realesrgan'] = (p_realesrgan or '').strip()
                            cfg['path_swinir'] = (p_swinir or '').strip()
                            cfg['path_dat'] = (p_dat or '').strip()
                            cfg['path_gfpgan'] = (p_gfpgan or '').strip()
                            cfg['path_codeformer'] = (p_codeformer or '').strip()
                            with open(cfg_path, 'w', encoding='utf-8') as f:
                                _json.dump(cfg, f, indent=4, ensure_ascii=False)
                            return gr.update(value='<span style="color:#4ecdc4;">Saved. Restart UI to refresh model dropdowns.</span>')
                        except Exception as _e:
                            return gr.update(value=f'<span style="color:#ff6b6b;">Error: {_e}</span>')

                    up_path_save_btn.click(
                        _save_upscale_paths,
                        inputs=[up_path_esrgan, up_path_realesrgan, up_path_swinir,
                                up_path_dat, up_path_gfpgan, up_path_codeformer],
                        outputs=[up_path_status],
                        queue=False, show_progress=False)

                    def _ab_save_and_status(enabled, thumbs, dzi, index_boot, blur):
                        ok, msg = modules.config.write_asset_browser_settings({
                            'enabled': enabled,
                            'generate_thumbnails': thumbs,
                            'generate_dzi_tiles': dzi,
                            'index_models_on_boot': index_boot,
                            'blur_thumbnails': blur,
                        })
                        # If we just turned the feature on (or saved any change while
                        # enabled), drop the SPA assets + refresh spa_settings.json
                        # so the Browser picks up the new defaults on next reload.
                        if ok and enabled:
                            try:
                                from modules.gallery_writer import ensure_gallery_assets
                                ensure_gallery_assets()
                            except Exception as _e:
                                msg += f' (asset bootstrap warning: {_e})'
                        color = '#4ecdc4' if ok else '#ff6b6b'
                        return gr.update(value=f'<span style="color:{color};">{msg}</span>')

                    ab_save_btn.click(
                        _ab_save_and_status,
                        inputs=[ab_enabled, ab_thumbs, ab_dzi, ab_index_boot, ab_blur],
                        outputs=[ab_status],
                        queue=False, show_progress=False)

                    def _ab_reindex_now():
                        # Async: spawn the worker and return immediately. The UI
                        # polls reindex_status() every 2 s (set up below) so the
                        # browser HTTP request never blocks for the multi-minute
                        # job (used to time out on Win10054).
                        try:
                            from modules.gallery_writer import reindex_outputs_async
                            started, msg = reindex_outputs_async()
                        except Exception as e:
                            return gr.update(value=f'<span style="color:#ff6b6b;">Reindex failed to start: {e}</span>')
                        color = '#4ecdc4' if started else '#ffa500'
                        return gr.update(value=f'<span style="color:{color};">{msg}</span>')

                    # _ab_format_status_html removed — replaced by client-side
                    # JS polling against /run/ab_reindex_status (see HTML block
                    # below). Gradio 3.41's `every=N` on .load() requires
                    # queue=True to repeat, which would serialise polling
                    # against generation. Client-side polling avoids both
                    # issues and never blocks the queue.

                    def _ab_reindex_smart():
                        try:
                            from modules.gallery_writer import reindex_outputs_async
                            started, msg = reindex_outputs_async(missing_only=True)
                        except Exception as e:
                            return gr.update(value=f'<span style="color:#ff6b6b;">Smart reindex failed to start: {e}</span>')
                        color = '#4ecdc4' if started else '#ffa500'
                        return gr.update(value=f'<span style="color:{color};">{msg}</span>')

                    ab_reindex_btn.click(
                        _ab_reindex_now,
                        inputs=[],
                        outputs=[ab_status],
                        queue=False, show_progress=False)

                    ab_reindex_smart_btn.click(
                        _ab_reindex_smart,
                        inputs=[],
                        outputs=[ab_status],
                        queue=False, show_progress=False)

                    # Hidden API endpoint that returns the reindex status JSON.
                    # Polled by the inline JS below — bypasses Gradio's queue
                    # entirely so it never blocks generations.
                    def _ab_reindex_status_json():
                        try:
                            from modules.gallery_writer import reindex_status
                            return reindex_status()
                        except Exception as e:
                            return {'phase': 'error', 'message': f'status read failed: {e}',
                                     'last_summary': f'status read failed: {e}', 'running': False}
                    ab_status_api_btn = gr.Button(visible=False)
                    ab_status_api_out = gr.JSON(visible=False)
                    ab_status_api_btn.click(
                        _ab_reindex_status_json,
                        inputs=[],
                        outputs=[ab_status_api_out],
                        api_name='ab_reindex_status',
                        queue=False, show_progress=False,
                    )

                    # Inject the polling JS via gradio_root.load(_js=...).
                    # gr.HTML(value='<script>...') is stripped by Gradio's
                    # XSS sanitizer in 3.41, but `_js=` is the official
                    # Gradio escape hatch for arbitrary page-load JS and
                    # runs as-is in the browser.
                    _ab_reindex_js = r"""() => {
  if (window._abReindexPollSetup) return [];
  window._abReindexPollSetup = true;
  let pollTimer = null;
  function fmtPhase(s) {
    if (!s) return '';
    if (s.phase === 'starting') return '<span style="color:#888;">Reindex starting…</span>';
    if (s.phase === 'outputs')  return `<span style="color:#bb86fc;">\u{1F504} Outputs: [${s.days_seen||0}/${s.days_total||0}] · current: <b>${s.current_date||'?'}</b> · ${s.images_total||0} image(s) processed so far</span>`;
    if (s.phase === 'models')   return `<span style="color:#bb86fc;">\u{1F504} Outputs done: ${s.days_done||0}/${s.days_total||0} day(s), ${s.images_total||0} image(s). Now scanning models…</span>`;
    if (s.phase === 'done')     return `<span style="color:#4ecdc4;">✓ ${s.last_summary||s.message||'Done.'}</span>`;
    if (s.phase === 'error')    return `<span style="color:#ff6b6b;">${s.last_summary||s.message||'Failed.'}</span>`;
    return '';
  }
  async function pollOnce() {
    try {
      let resp = null;
      for (const url of ['/run/ab_reindex_status', '/api/ab_reindex_status']) {
        try { resp = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:'{"data":[]}'}); } catch (_) { continue; }
        if (resp && resp.ok) break;
      }
      if (!resp || !resp.ok) return;
      const json = await resp.json();
      const s = (json && Array.isArray(json.data)) ? json.data[0] : json;
      // gradio app DOM is shadowed inside <gradio-app>; resolve via window.gradioApp() if present.
      const root = (typeof window.gradioApp === 'function') ? window.gradioApp() : document;
      const el = root.querySelector('#ab-status') || document.getElementById('ab-status');
      if (!el || !s) return;
      const html = fmtPhase(s);
      if (html) el.innerHTML = html;
      if (!s.running && (s.phase === 'done' || s.phase === 'error' || s.phase === 'idle')) {
        if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
      }
    } catch (e) { console.warn('[asset-browser] poll error:', e); }
  }
  window._abReindexStartPoll = function() {
    if (pollTimer) clearInterval(pollTimer);
    pollOnce();
    pollTimer = setInterval(pollOnce, 2000);
  };
  function attach() {
    const root = (typeof window.gradioApp === 'function') ? window.gradioApp() : document;
    const btn = root.querySelector('#ab-reindex-btn') || document.getElementById('ab-reindex-btn');
    if (!btn) { setTimeout(attach, 500); return; }
    if (btn._abPollAttached) return;
    btn._abPollAttached = true;
    btn.addEventListener('click', () => setTimeout(window._abReindexStartPoll, 200));
    console.log('[asset-browser] reindex polling attached');
  }
  attach();
  return [];
}"""
                    shared.gradio_root.load(
                        None, inputs=[], outputs=[], _js=_ab_reindex_js,
                        queue=False, show_progress=False,
                    )

                    # === Hidden API bridge for the SPA's "Fetch from CivitAI" button.
                    # Exposed via api_name so the standalone HTML can POST to it
                    # over Gradio's REST endpoint (/run/ab_fetch_preview in v3.x,
                    # /api/predict via Gradio queue). Pure backend bridge, no UI.
                    ab_fetch_kind_in     = gr.Textbox(visible=False)
                    ab_fetch_filename_in = gr.Textbox(visible=False)
                    ab_fetch_result_out  = gr.JSON(visible=False)
                    ab_fetch_btn         = gr.Button(visible=False)

                    def _ab_fetch_civitai_preview(kind, rel_filename):
                        try:
                            from modules.model_indexer import fetch_civitai_preview_for
                            return fetch_civitai_preview_for(kind, rel_filename)
                        except Exception as e:
                            return {'success': False,
                                     'message': f'Internal error: {e}',
                                     'kind': kind, 'rel_path': rel_filename}

                    ab_fetch_btn.click(
                        _ab_fetch_civitai_preview,
                        inputs=[ab_fetch_kind_in, ab_fetch_filename_in],
                        outputs=[ab_fetch_result_out],
                        api_name='ab_fetch_preview',
                        queue=False, show_progress=False,
                    )
                # === end custom-8 =================================================

                # Panneau Extra (plugins externes, ex: crispz). Active comme l'Asset Browser.
                extra_plugins_checkbox = gr.Checkbox(label='\U0001F9E9 Extra Plugins', value=extra_ui.enabled_default(), container=False,
                                                     info='Shows the Extra panel: external upscalers via plugins (e.g. crispz).')

                dev_mode = gr.Checkbox(label='Developer Debug Mode', value=modules.config.default_developer_debug_mode_checkbox, container=False)

                with gr.Column(visible=modules.config.default_developer_debug_mode_checkbox) as dev_tools:
                    with gr.Tab(label='Debug Tools'):
                        adm_scaler_positive = gr.Slider(label='Positive ADM Guidance Scaler', minimum=0.1, maximum=3.0,
                                                        step=0.001, value=1.5, info='The scaler multiplied to positive ADM (use 1.0 to disable). ')
                        adm_scaler_negative = gr.Slider(label='Negative ADM Guidance Scaler', minimum=0.1, maximum=3.0,
                                                        step=0.001, value=0.8, info='The scaler multiplied to negative ADM (use 1.0 to disable). ')
                        adm_scaler_end = gr.Slider(label='ADM Guidance End At Step', minimum=0.0, maximum=1.0,
                                                   step=0.001, value=0.3,
                                                   info='When to end the guidance from positive/negative ADM. ')

                        refiner_swap_method = gr.Dropdown(label='Refiner swap method', value=flags.refiner_swap_method,
                                                          choices=['joint', 'separate', 'vae'])

                        adaptive_cfg = gr.Slider(label='CFG Mimicking from TSNR', minimum=1.0, maximum=30.0, step=0.01,
                                                 value=modules.config.default_cfg_tsnr,
                                                 info='Enabling Fooocus\'s implementation of CFG mimicking for TSNR '
                                                      '(effective when real CFG > mimicked CFG).')
                        clip_skip = gr.Slider(label='CLIP Skip', minimum=1, maximum=flags.clip_skip_max, step=1,
                                                 value=modules.config.default_clip_skip,
                                                 info='Bypass CLIP layers to avoid overfitting (use 1 to not skip any layers, 2 is recommended).')
                        sampler_name = gr.Dropdown(label='Sampler', choices=flags.sampler_list,
                                                   value=modules.config.default_sampler)
                        scheduler_name = gr.Dropdown(label='Scheduler', choices=flags.scheduler_list,
                                                     value=modules.config.default_scheduler)
                        vae_name = gr.Dropdown(label='VAE', choices=[modules.flags.default_vae] + modules.config.vae_filenames,
                                                     value=modules.config.default_vae, show_label=True)

                        generate_image_grid = gr.Checkbox(label='Generate Image Grid for Each Batch',
                                                          info='(Experimental) This may cause performance problems on some computers and certain internet conditions.',
                                                          value=False)

                        overwrite_step = gr.Slider(label='Forced Overwrite of Sampling Step',
                                                   minimum=-1, maximum=200, step=1,
                                                   value=modules.config.default_overwrite_step,
                                                   info='Set as -1 to disable. For developer debugging.')
                        overwrite_switch = gr.Slider(label='Forced Overwrite of Refiner Switch Step',
                                                     minimum=-1, maximum=200, step=1,
                                                     value=modules.config.default_overwrite_switch,
                                                     info='Set as -1 to disable. For developer debugging.')
                        overwrite_width = gr.Slider(label='Forced Overwrite of Generating Width',
                                                    minimum=-1, maximum=2048, step=1, value=-1,
                                                    info='Set as -1 to disable. For developer debugging. '
                                                         'Results will be worse for non-standard numbers that SDXL is not trained on.')
                        overwrite_height = gr.Slider(label='Forced Overwrite of Generating Height',
                                                     minimum=-1, maximum=2048, step=1, value=-1,
                                                     info='Set as -1 to disable. For developer debugging. '
                                                          'Results will be worse for non-standard numbers that SDXL is not trained on.')
                        overwrite_vary_strength = gr.Slider(label='Forced Overwrite of Denoising Strength of "Vary"',
                                                            minimum=-1, maximum=1.0, step=0.001, value=-1,
                                                            info='Set as negative number to disable. For developer debugging.')
                        # custom-10.4: overwrite_upscale_strength slider moved out of
                        # Developer Debug to the custom-10 block in Enhance > UOV for
                        # easier discoverability (see further down in webui.py).

                        disable_preview = gr.Checkbox(label='Disable Preview', value=modules.config.default_black_out_nsfw,
                                                      interactive=not modules.config.default_black_out_nsfw,
                                                      info='Disable preview during generation.')
                        disable_intermediate_results = gr.Checkbox(label='Disable Intermediate Results',
                                                      value=flags.Performance.has_restricted_features(modules.config.default_performance),
                                                      info='Disable intermediate results during generation, only show final gallery.')

                        disable_seed_increment = gr.Checkbox(label='Disable seed increment',
                                                             info='Disable automatic seed increment when image number is > 1.',
                                                             value=False)
                        read_wildcards_in_order = gr.Checkbox(label="Read wildcards in order", value=False)

                        black_out_nsfw = gr.Checkbox(label='Black Out NSFW', value=modules.config.default_black_out_nsfw,
                                                     interactive=not modules.config.default_black_out_nsfw,
                                                     info='Use black image if NSFW is detected.')

                        black_out_nsfw.change(lambda x: gr.update(value=x, interactive=not x),
                                              inputs=black_out_nsfw, outputs=disable_preview, queue=False,
                                              show_progress=False)

                        if not args_manager.args.disable_image_log:
                            save_final_enhanced_image_only = gr.Checkbox(label='Save only final enhanced image',
                                                                         value=modules.config.default_save_only_final_enhanced_image)

                        if not args_manager.args.disable_metadata:
                            save_metadata_to_images = gr.Checkbox(label='Save Metadata to Images', value=modules.config.default_save_metadata_to_images,
                                                                  info='Adds parameters to generated images allowing manual regeneration.')
                            metadata_scheme = gr.Radio(label='Metadata Scheme', choices=flags.metadata_scheme, value=modules.config.default_metadata_scheme,
                                                       info='Image Prompt parameters are not included. Use png and a1111 for compatibility with Civitai.',
                                                       visible=modules.config.default_save_metadata_to_images)

                            save_metadata_to_images.change(lambda x: gr.update(visible=x), inputs=[save_metadata_to_images], outputs=[metadata_scheme],
                                                           queue=False, show_progress=False)

                    with gr.Tab(label='Control'):
                        debugging_cn_preprocessor = gr.Checkbox(label='Debug Preprocessors', value=False,
                                                                info='See the results from preprocessors.')
                        skipping_cn_preprocessor = gr.Checkbox(label='Skip Preprocessors', value=False,
                                                               info='Do not preprocess images. (Inputs are already canny/depth/cropped-face/etc.)')

                        mixing_image_prompt_and_vary_upscale = gr.Checkbox(label='Mixing Image Prompt and Vary/Upscale',
                                                                           value=False)
                        mixing_image_prompt_and_inpaint = gr.Checkbox(label='Mixing Image Prompt and Inpaint',
                                                                      value=False)

                        controlnet_softness = gr.Slider(label='Softness of ControlNet', minimum=0.0, maximum=1.0,
                                                        step=0.001, value=0.25,
                                                        info='Similar to the Control Mode in A1111 (use 0.0 to disable). ')

                        with gr.Tab(label='Canny'):
                            canny_low_threshold = gr.Slider(label='Canny Low Threshold', minimum=1, maximum=255,
                                                            step=1, value=64)
                            canny_high_threshold = gr.Slider(label='Canny High Threshold', minimum=1, maximum=255,
                                                             step=1, value=128)

                    with gr.Tab(label='Inpaint'):
                        debugging_inpaint_preprocessor = gr.Checkbox(label='Debug Inpaint Preprocessing', value=False)
                        debugging_enhance_masks_checkbox = gr.Checkbox(label='Debug Enhance Masks', value=False,
                                                                       info='Show enhance masks in preview and final results')
                        debugging_dino = gr.Checkbox(label='Debug GroundingDINO', value=False,
                                                     info='Use GroundingDINO boxes instead of more detailed SAM masks')
                        inpaint_disable_initial_latent = gr.Checkbox(label='Disable initial latent in inpaint', value=False)
                        inpaint_engine = gr.Dropdown(label='Inpaint Engine',
                                                     value=modules.config.default_inpaint_engine_version,
                                                     choices=flags.inpaint_engine_versions,
                                                     info='Version of Fooocus inpaint model. If set, use performance Quality or Speed (no performance LoRAs) for best results.')
                        inpaint_strength = gr.Slider(label='Inpaint Denoising Strength',
                                                     minimum=0.0, maximum=1.0, step=0.001, value=1.0,
                                                     info='Same as the denoising strength in A1111 inpaint. '
                                                          'Only used in inpaint, not used in outpaint. '
                                                          '(Outpaint always use 1.0)')
                        inpaint_respective_field = gr.Slider(label='Inpaint Respective Field',
                                                             minimum=0.0, maximum=1.0, step=0.001, value=0.618,
                                                             info='The area to inpaint. '
                                                                  'Value 0 is same as "Only Masked" in A1111. '
                                                                  'Value 1 is same as "Whole Image" in A1111. '
                                                                  'Only used in inpaint, not used in outpaint. '
                                                                  '(Outpaint always use 1.0)')
                        inpaint_erode_or_dilate = gr.Slider(label='Mask Erode or Dilate',
                                                            minimum=-64, maximum=64, step=1, value=0,
                                                            info='Positive value will make white area in the mask larger, '
                                                                 'negative value will make white area smaller. '
                                                                 '(default is 0, always processed before any mask invert)')
                        dino_erode_or_dilate = gr.Slider(label='GroundingDINO Box Erode or Dilate',
                                                         minimum=-64, maximum=64, step=1, value=0,
                                                         info='Positive value will make white area in the mask larger, '
                                                              'negative value will make white area smaller. '
                                                              '(default is 0, processed before SAM)')

                        inpaint_mask_color = gr.ColorPicker(label='Inpaint brush color', value='#FFFFFF', elem_id='inpaint_brush_color')

                        inpaint_ctrls = [debugging_inpaint_preprocessor, inpaint_disable_initial_latent, inpaint_engine,
                                         inpaint_strength, inpaint_respective_field,
                                         inpaint_advanced_masking_checkbox, invert_mask_checkbox, inpaint_erode_or_dilate]

                        inpaint_advanced_masking_checkbox.change(lambda x: [gr.update(visible=x)] * 2,
                                                                 inputs=inpaint_advanced_masking_checkbox,
                                                                 outputs=[inpaint_mask_image, inpaint_mask_generation_col],
                                                                 queue=False, show_progress=False)

                        inpaint_mask_color.change(lambda x: gr.update(brush_color=x), inputs=inpaint_mask_color,
                                                  outputs=inpaint_input_image,
                                                  queue=False, show_progress=False)

                    with gr.Tab(label='FreeU'):
                        freeu_enabled = gr.Checkbox(label='Enabled', value=False)
                        freeu_b1 = gr.Slider(label='B1', minimum=0, maximum=2, step=0.01, value=1.01)
                        freeu_b2 = gr.Slider(label='B2', minimum=0, maximum=2, step=0.01, value=1.02)
                        freeu_s1 = gr.Slider(label='S1', minimum=0, maximum=4, step=0.01, value=0.99)
                        freeu_s2 = gr.Slider(label='S2', minimum=0, maximum=4, step=0.01, value=0.95)
                        freeu_ctrls = [freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2]

                def dev_mode_checked(r):
                    return gr.update(visible=r)

                dev_mode.change(dev_mode_checked, inputs=[dev_mode], outputs=[dev_tools],
                                queue=False, show_progress=False)

                def refresh_files_clicked():
                    modules.config.update_files()
                    results = [gr.update(choices=modules.config.model_filenames)]
                    results += [gr.update(choices=['None'] + modules.config.model_filenames)]
                    results += [gr.update(choices=[flags.default_vae] + modules.config.vae_filenames)]
                    if not args_manager.args.disable_preset_selection:
                        results += [gr.update(choices=modules.config.available_presets)]
                    for i in range(modules.config.default_max_lora_number):
                        results += [gr.update(interactive=True),
                                    gr.update(choices=['None'] + modules.config.lora_filenames), gr.update()]
                    return results

                refresh_files_output = [base_model, refiner_model, vae_name]
                if not args_manager.args.disable_preset_selection:
                    refresh_files_output += [preset_selection]
                refresh_files.click(refresh_files_clicked, [], refresh_files_output + lora_ctrls,
                                    queue=False, show_progress=False)

                def _restart_ui():
                    """Exit the Python process with code 42, which the launcher .bat
                    interprets as 'please restart me'. See run*.bat for the loop.

                    If the user's launcher doesn't implement the restart loop, the
                    process will just exit and the console will stay open (pause).
                    """
                    def _do_exit():
                        time.sleep(0.4)  # let the Gradio response leave the socket first
                        os._exit(42)
                    import threading
                    threading.Thread(target=_do_exit, daemon=True).start()
                    return gr.update(value='<div style="padding:8px;border:1px solid #ffa500;border-radius:6px;'
                                           'color:#ffa500;">\u26A0 Restarting\u2026 wait ~30 s then refresh this page. '
                                           'If the page does not come back, your launcher does not implement the '
                                           'restart loop \u2014 re-run the .bat manually.</div>')

                # === Preset Manager Event Handlers ===
                if not args_manager.args.disable_metadata:

                    # Inputs for collecting current settings
                    _n_lora_ctrls = len(lora_ctrls)
                    _n_embedding_ctrls = len(embedding_ctrls)

                    preset_save_inputs = [
                        save_preset_name,
                        save_preset_overwrite_dropdown,
                        base_model, refiner_model, refiner_switch,
                        guidance_scale, sharpness, adaptive_cfg, clip_skip,
                        sampler_name, scheduler_name,
                        overwrite_step, overwrite_switch,
                        performance_selection, image_number,
                        prompt, negative_prompt,
                        style_selections, aspect_ratios_selection,
                        vae_name,
                        # custom-7: custom resolution
                        custom_res_enabled, custom_ratio_w, custom_ratio_h,
                        custom_res_mode, custom_res_size,
                    ] + lora_ctrls + embedding_ctrls

                    def _collect_current_values(preset_name, overwrite_target,
                                                 base_model_v, refiner_model_v, refiner_switch_v,
                                                 guidance_scale_v, sharpness_v, adaptive_cfg_v, clip_skip_v,
                                                 sampler_name_v, scheduler_name_v,
                                                 overwrite_step_v, overwrite_switch_v,
                                                 performance_v, image_number_v,
                                                 prompt_v, negative_prompt_v,
                                                 styles_v, aspect_ratio_v,
                                                 vae_name_v,
                                                 custom_res_enabled_v, custom_ratio_w_v, custom_ratio_h_v,
                                                 custom_res_mode_v, custom_res_size_v,
                                                 *lora_and_embedding_args):
                        """Collect all current UI values into a dict for save_preset_to_file."""
                        current = {
                            'base_model': base_model_v,
                            'refiner_model': refiner_model_v,
                            'refiner_switch': refiner_switch_v,
                            'guidance_scale': guidance_scale_v,
                            'sharpness': sharpness_v,
                            'adaptive_cfg': adaptive_cfg_v,
                            'clip_skip': clip_skip_v,
                            'sampler_name': sampler_name_v,
                            'scheduler_name': scheduler_name_v,
                            'overwrite_step': overwrite_step_v,
                            'overwrite_switch': overwrite_switch_v,
                            'performance_selection': performance_v,
                            'image_number': image_number_v,
                            'prompt': prompt_v,
                            'negative_prompt': negative_prompt_v,
                            'style_selections': styles_v,
                            'aspect_ratios_selection': aspect_ratio_v,
                            'vae_name': vae_name_v,
                            # custom-7
                            'custom_res_enabled': custom_res_enabled_v,
                            'custom_ratio_w': custom_ratio_w_v,
                            'custom_ratio_h': custom_ratio_h_v,
                            'custom_res_mode': custom_res_mode_v,
                            'custom_res_size': custom_res_size_v,
                        }

                        all_args = list(lora_and_embedding_args)
                        lora_args = all_args[:_n_lora_ctrls]
                        embedding_args = all_args[_n_lora_ctrls:_n_lora_ctrls + _n_embedding_ctrls]

                        # Reconstruct LoRA list from flat args: [enabled, model, weight, ...]
                        loras = []
                        for i in range(0, len(lora_args), 3):
                            if i + 2 < len(lora_args):
                                loras.append([bool(lora_args[i]), str(lora_args[i + 1]), float(lora_args[i + 2])])
                        current['loras'] = loras

                        # Reconstruct Embeddings list from flat args: [enabled, model, weight, ...]
                        embeddings = []
                        for i in range(0, len(embedding_args), 3):
                            if i + 2 < len(embedding_args):
                                embeddings.append([bool(embedding_args[i]), str(embedding_args[i + 1]), float(embedding_args[i + 2])])
                        current['embeddings'] = embeddings

                        return preset_name, overwrite_target, current

                    has_preset_dropdown = not args_manager.args.disable_preset_selection

                    def _build_result(html_update, dropdown_update, presets_update=None):
                        """Build result tuple matching the number of expected outputs."""
                        result = [html_update, dropdown_update]
                        if has_preset_dropdown:
                            result.append(presets_update if presets_update is not None else gr.update())
                        return result

                    def save_new_preset(preset_name, overwrite_target, *args):
                        name, _, current = _collect_current_values(preset_name, overwrite_target, *args)
                        if not name or not name.strip():
                            return _build_result(
                                gr.update(value='<span style="color: #ff6b6b;">Saisis un nom de preset.</span>'),
                                gr.update())
                        success, msg = modules.config.save_preset_to_file(name, current, overwrite=False)
                        color = '#4ecdc4' if success else '#ff6b6b'
                        html = f'<span style="color: {color};">{msg}</span>'
                        if success:
                            user_presets = modules.config.get_user_presets()
                            return _build_result(
                                gr.update(value=html),
                                gr.update(choices=user_presets, value=name.strip()),
                                gr.update(choices=modules.config.available_presets))
                        return _build_result(gr.update(value=html), gr.update())

                    def overwrite_existing_preset(preset_name, overwrite_target, *args):
                        _, target, current = _collect_current_values(preset_name, overwrite_target, *args)
                        if not target:
                            return _build_result(
                                gr.update(value='<span style="color: #ff6b6b;">Selectionne un preset a ecraser.</span>'),
                                gr.update())
                        success, msg = modules.config.save_preset_to_file(target, current, overwrite=True)
                        color = '#4ecdc4' if success else '#ff6b6b'
                        html = f'<span style="color: {color};">{msg}</span>'
                        if success:
                            user_presets = modules.config.get_user_presets()
                            return _build_result(
                                gr.update(value=html),
                                gr.update(choices=user_presets, value=target),
                                gr.update(choices=modules.config.available_presets))
                        return _build_result(gr.update(value=html), gr.update())

                    save_preset_save_outputs = [save_preset_status, save_preset_overwrite_dropdown]
                    if has_preset_dropdown:
                        save_preset_save_outputs += [preset_selection]

                    save_preset_new_btn.click(
                        save_new_preset,
                        inputs=preset_save_inputs,
                        outputs=save_preset_save_outputs,
                        queue=False, show_progress=False
                    )

                    save_preset_overwrite_btn.click(
                        overwrite_existing_preset,
                        inputs=preset_save_inputs,
                        outputs=save_preset_save_outputs,
                        queue=False, show_progress=False
                    )

                # === CivitAI Integration Event Handlers ===
                def civitai_save_key(api_key):
                    # Don't save the masked version
                    if '...' in str(api_key):
                        return gr.update()
                    success, msg = modules.config.save_civitai_api_key(api_key)
                    if success and api_key:
                        masked = f'{api_key[:4]}...{api_key[-4:]}' if len(api_key) > 8 else api_key
                        return gr.update(value=masked, label='CivitAI API Key <span style="color:#4ecdc4;">(saved)</span>')
                    return gr.update(label=f'CivitAI API Key <span style="color:#ff6b6b;">({msg})</span>')

                civitai_save_key_btn.click(
                    civitai_save_key,
                    inputs=[civitai_api_key_input],
                    outputs=[civitai_api_key_input],
                    queue=False, show_progress=False
                )

                def civitai_fetch_settings(model_name, api_key_field, force_refresh=False):
                    empty = (
                        gr.update(value='<div style="padding:8px;border:1px solid #ff6b6b;border-radius:8px;color:#ff6b6b;">'
                                        'No model selected.</div>'),
                        gr.update(visible=False),  # apply_btn
                        gr.update(visible=False),  # refresh_btn
                        None,                       # civitai_settings_state
                        None,                       # civitai_model_info_state
                        gr.update(visible=False),  # copy_triggers_btn
                        gr.update(visible=False),  # preset_name
                        gr.update(visible=False),  # save_preset_btn
                        gr.update(visible=False),  # save_preset_status
                    )
                    if not model_name or model_name == 'None':
                        return empty

                    actual_key = api_key_field
                    if not actual_key or '...' in str(actual_key):
                        actual_key = modules.config.civitai_api_key

                    result = modules.civitai_api.fetch_recommended_settings(
                        model_filename=model_name,
                        paths_checkpoints=modules.config.paths_checkpoints,
                        api_key=actual_key if actual_key else None,
                        force_refresh=force_refresh
                    )

                    html = modules.civitai_api.format_settings_html(result)
                    has_settings = 'settings' in result
                    raw_settings = result.get('settings') if has_settings else None
                    model_info = result.get('model_info') if has_settings else None
                    triggers = (model_info or {}).get('trainedWords') or []
                    has_triggers = bool(triggers)
                    # Default preset-name suggestion
                    safe_model = ''
                    if model_info and model_info.get('modelName'):
                        safe_model = ''.join(c for c in model_info['modelName'] if c.isalnum() or c in ('_', '-'))[:40]
                    suggested_preset_name = f'civitai_{safe_model}' if safe_model else ''

                    return (
                        gr.update(value=html),
                        gr.update(visible=has_settings),
                        gr.update(visible=has_settings),
                        raw_settings,
                        model_info,
                        gr.update(visible=has_triggers),
                        gr.update(visible=has_settings, value=suggested_preset_name),
                        gr.update(visible=has_settings),
                        gr.update(visible=False, value=''),
                    )

                def civitai_refresh_settings(model_name, api_key_field):
                    return civitai_fetch_settings(model_name, api_key_field, force_refresh=True)

                _civitai_fetch_outputs = [
                    civitai_panel, civitai_apply_btn, civitai_refresh_btn,
                    civitai_settings_state, civitai_model_info_state,
                    civitai_copy_triggers_btn,
                    civitai_preset_name, civitai_save_preset_btn, civitai_save_preset_status,
                ]
                civitai_fetch_btn.click(
                    civitai_fetch_settings,
                    inputs=[base_model, civitai_api_key_input],
                    outputs=_civitai_fetch_outputs,
                    queue=True, show_progress=True
                )
                civitai_refresh_btn.click(
                    civitai_refresh_settings,
                    inputs=[base_model, civitai_api_key_input],
                    outputs=_civitai_fetch_outputs,
                    queue=True, show_progress=True
                )

                def civitai_copy_triggers_to_prompt(model_info_data, current_prompt):
                    if not model_info_data or not isinstance(model_info_data, dict):
                        return gr.update()
                    words = model_info_data.get('trainedWords') or []
                    if not words:
                        return gr.update()
                    current_prompt = current_prompt or ''
                    existing = {t.strip().lower() for t in current_prompt.split(',') if t.strip()}
                    to_add = []
                    for w in words:
                        k = w.strip().lower()
                        if k and k not in existing:
                            to_add.append(w.strip())
                            existing.add(k)
                    if not to_add:
                        return gr.update()
                    sep = '' if not current_prompt else (', ' if not current_prompt.rstrip().endswith(',') else ' ')
                    return gr.update(value=current_prompt + sep + ', '.join(to_add))

                civitai_copy_triggers_btn.click(
                    civitai_copy_triggers_to_prompt,
                    inputs=[civitai_model_info_state, prompt],
                    outputs=[prompt],
                    queue=False, show_progress=False
                )

                def civitai_save_consensus_as_preset(preset_name_v, civitai_settings, civitai_model_info,
                                                      base_model_v, refiner_model_v, refiner_switch_v,
                                                      guidance_scale_v, sharpness_v, adaptive_cfg_v, clip_skip_v,
                                                      performance_v, image_number_v,
                                                      prompt_v, negative_prompt_v,
                                                      styles_v, aspect_ratio_v, vae_name_v,
                                                      *lora_and_embedding_args):
                    if not civitai_settings or not isinstance(civitai_settings, dict):
                        return gr.update(value=('<div style="padding:6px;border:1px solid #ff6b6b;'
                                                'border-radius:6px;color:#ff6b6b;">No CivitAI settings to save.</div>'),
                                         visible=True)
                    name = (preset_name_v or '').strip()
                    if not name:
                        info = civitai_model_info or {}
                        stem = ''.join(c for c in (info.get('modelName') or '') if c.isalnum() or c in ('_', '-'))[:40]
                        name = f'civitai_{stem}' if stem else 'civitai_preset'

                    # Start from the CivitAI consensus, falling back to current UI values
                    sampler_v = civitai_settings.get('sampler_fooocus') or modules.config.default_sampler
                    scheduler_v = civitai_settings.get('scheduler_fooocus') or modules.config.default_scheduler
                    cfg_v = civitai_settings.get('cfg_scale', guidance_scale_v)
                    steps_v = civitai_settings.get('steps')
                    if steps_v is None:
                        steps_v = -1  # let Fooocus use the performance default
                    clip_v = civitai_settings.get('clip_skip', clip_skip_v)

                    current = {
                        'base_model': base_model_v,
                        'refiner_model': refiner_model_v,
                        'refiner_switch': refiner_switch_v,
                        'guidance_scale': cfg_v,
                        'sharpness': sharpness_v,
                        'adaptive_cfg': adaptive_cfg_v,
                        'clip_skip': clip_v,
                        'sampler_name': sampler_v,
                        'scheduler_name': scheduler_v,
                        'overwrite_step': steps_v,
                        'overwrite_switch': -1,
                        'performance_selection': performance_v,
                        'image_number': image_number_v,
                        'prompt': prompt_v,
                        'negative_prompt': negative_prompt_v,
                        'style_selections': styles_v,
                        'aspect_ratios_selection': aspect_ratio_v,
                        'vae_name': vae_name_v,
                    }

                    all_args = list(lora_and_embedding_args)
                    lora_args = all_args[:_n_lora_ctrls]
                    embedding_args = all_args[_n_lora_ctrls:_n_lora_ctrls + _n_embedding_ctrls]
                    loras = []
                    for i in range(0, len(lora_args), 3):
                        if i + 2 < len(lora_args):
                            loras.append([bool(lora_args[i]), str(lora_args[i + 1]), float(lora_args[i + 2])])
                    current['loras'] = loras
                    embeddings = []
                    for i in range(0, len(embedding_args), 3):
                        if i + 2 < len(embedding_args):
                            embeddings.append([bool(embedding_args[i]), str(embedding_args[i + 1]), float(embedding_args[i + 2])])
                    current['embeddings'] = embeddings

                    success, msg = modules.config.save_preset_to_file(name, current, overwrite=False)
                    color = '#4ecdc4' if success else '#ff6b6b'
                    prefix = 'Saved' if success else 'Failed'
                    return gr.update(
                        value=(f'<div style="padding:6px;border:1px solid {color};border-radius:6px;color:{color};">'
                               f'{prefix}: {msg}</div>'),
                        visible=True,
                    )

                civitai_save_preset_inputs = [
                    civitai_preset_name, civitai_settings_state, civitai_model_info_state,
                    base_model, refiner_model, refiner_switch,
                    guidance_scale, sharpness, adaptive_cfg, clip_skip,
                    performance_selection, image_number,
                    prompt, negative_prompt,
                    style_selections, aspect_ratios_selection, vae_name,
                ] + lora_ctrls + embedding_ctrls

                civitai_save_preset_btn.click(
                    civitai_save_consensus_as_preset,
                    inputs=civitai_save_preset_inputs,
                    outputs=[civitai_save_preset_status],
                    queue=False, show_progress=False
                )

                def civitai_apply_settings(settings_data, current_sampler, current_scheduler,
                                            current_cfg, current_steps, current_clip_skip):
                    if not settings_data or not isinstance(settings_data, dict):
                        return [gr.update()] * 5 + [gr.update(value='<div style="padding:8px;border:1px solid #ff6b6b;border-radius:8px;color:#ff6b6b;">No settings to apply.</div>')]

                    updates = {}

                    # Sampler
                    if settings_data.get('sampler_fooocus'):
                        updates['sampler'] = settings_data['sampler_fooocus']
                    # Scheduler
                    if settings_data.get('scheduler_fooocus'):
                        updates['scheduler'] = settings_data['scheduler_fooocus']
                    # CFG
                    if 'cfg_scale' in settings_data:
                        updates['cfg'] = settings_data['cfg_scale']
                    # Steps
                    if 'steps' in settings_data:
                        updates['steps'] = settings_data['steps']
                    # Clip Skip
                    if 'clip_skip' in settings_data:
                        updates['clip_skip'] = settings_data['clip_skip']

                    applied = []
                    sampler_up = gr.update(value=updates['sampler']) if 'sampler' in updates else gr.update()
                    if 'sampler' in updates:
                        applied.append(f'Sampler: {updates["sampler"]}')

                    scheduler_up = gr.update(value=updates['scheduler']) if 'scheduler' in updates else gr.update()
                    if 'scheduler' in updates:
                        applied.append(f'Scheduler: {updates["scheduler"]}')

                    cfg_up = gr.update(value=updates['cfg']) if 'cfg' in updates else gr.update()
                    if 'cfg' in updates:
                        applied.append(f'CFG: {updates["cfg"]}')

                    steps_up = gr.update(value=updates['steps']) if 'steps' in updates else gr.update()
                    if 'steps' in updates:
                        applied.append(f'Steps: {updates["steps"]}')

                    clip_up = gr.update(value=updates['clip_skip']) if 'clip_skip' in updates else gr.update()
                    if 'clip_skip' in updates:
                        applied.append(f'Clip Skip: {updates["clip_skip"]}')

                    applied_str = ', '.join(applied) if applied else 'Nothing changed'
                    panel_html = f'<div style="padding:8px;border:1px solid #4ecdc4;border-radius:8px;color:#4ecdc4;">' \
                                 f'Applied: {applied_str}</div>'

                    return [sampler_up, scheduler_up, cfg_up, steps_up, clip_up, gr.update(value=panel_html)]

                civitai_apply_btn.click(
                    civitai_apply_settings,
                    inputs=[civitai_settings_state, sampler_name, scheduler_name,
                            guidance_scale, overwrite_step, clip_skip],
                    outputs=[sampler_name, scheduler_name, guidance_scale, overwrite_step,
                             clip_skip, civitai_panel],
                    queue=False, show_progress=False
                )

                # === LoRA trigger words (CivitAI) event handlers ===
                def _resolve_civitai_key(api_key_field):
                    if api_key_field and '...' not in str(api_key_field):
                        return api_key_field
                    return modules.config.civitai_api_key or None

                def fetch_lora_triggers_for_slot(lora_name, api_key_field):
                    if not lora_name or lora_name == 'None':
                        return gr.update(value='')
                    result = modules.civitai_api.fetch_lora_triggers_combined(
                        lora_filename=lora_name,
                        paths_loras=modules.config.paths_loras,
                        api_key=_resolve_civitai_key(api_key_field),
                    )
                    return gr.update(value=modules.civitai_api.format_lora_triggers_display(result))

                for _dd, _disp in zip(lora_model_dropdowns, lora_trigger_displays):
                    _dd.change(
                        fetch_lora_triggers_for_slot,
                        inputs=[_dd, civitai_api_key_input],
                        outputs=[_disp],
                        queue=True, show_progress=False
                    )

                def _append_words_to_prompt(new_words, current_prompt):
                    current_prompt = current_prompt or ''
                    existing = {w.strip().lower() for w in current_prompt.split(',') if w.strip()}
                    to_add = []
                    for w in new_words:
                        w = w.strip()
                        if w and w.lower() not in existing:
                            to_add.append(w)
                            existing.add(w.lower())
                    if not to_add:
                        return gr.update()
                    sep = '' if not current_prompt else (', ' if not current_prompt.rstrip().endswith(',') else ' ')
                    return gr.update(value=current_prompt + sep + ', '.join(to_add))

                def copy_slot_triggers_to_prompt(trigger_text, current_prompt):
                    if not trigger_text or trigger_text.startswith('('):
                        return gr.update()
                    words = [w for w in trigger_text.split(',') if w.strip()]
                    return _append_words_to_prompt(words, current_prompt)

                for _btn, _disp in zip(lora_copy_btns, lora_trigger_displays):
                    _btn.click(
                        copy_slot_triggers_to_prompt,
                        inputs=[_disp, prompt],
                        outputs=[prompt],
                        queue=False, show_progress=False
                    )

                def copy_all_active_lora_triggers(current_prompt, *args):
                    n = len(args) // 2
                    trigger_texts = args[:n]
                    enables = args[n:]
                    words = []
                    for txt, en in zip(trigger_texts, enables):
                        if not en or not txt or txt.startswith('('):
                            continue
                        words.extend(w for w in txt.split(',') if w.strip())
                    return _append_words_to_prompt(words, current_prompt)

                _lora_enable_checkboxes = [lora_ctrls[i] for i in range(0, len(lora_ctrls), 3)]
                lora_copy_all_btn.click(
                    copy_all_active_lora_triggers,
                    inputs=[prompt] + lora_trigger_displays + _lora_enable_checkboxes,
                    outputs=[prompt],
                    queue=False, show_progress=False
                )

                # === Layout / Omost event handlers ===
                # Wired only when the feature is enabled (components exist only then).
                if _omost_enabled:
                    def omost_generate(idea):
                        cfg = getattr(modules.config, 'omost_config', {})
                        endpoint = cfg.get('endpoint', 'http://localhost:11434/v1/chat/completions')
                        model = cfg.get('model', 'omost-llama3')
                        try:
                            timeout = int(cfg.get('timeout', 120))
                        except (TypeError, ValueError):
                            timeout = 120
                        try:
                            code = _omost_client.call_omost_llm(idea, endpoint, model, timeout)
                        except Exception as exc:
                            return gr.update(value='[Omost error] ' + str(exc)), gr.update(value='')
                        layout = _omost_client.parse_canvas(code)
                        if 'error' in layout:
                            msg = ('[Parsing error] ' + layout['error']
                                   + '\n\n--- Code returned by the LLM ---\n' + (code or ''))
                            return gr.update(value=msg), gr.update(value='')
                        layout_json = json.dumps(layout, indent=2, ensure_ascii=False)
                        flat = _omost_client.flatten_to_prompt(layout)
                        return gr.update(value=layout_json), gr.update(value=flat)

                    omost_generate_btn.click(
                        omost_generate,
                        inputs=[omost_idea],
                        outputs=[omost_layout_box, omost_prompt_box],
                        queue=True, show_progress=True
                    )

                    # Reuse the LoRA dedup helper : never overwrite, append deduplicated.
                    def omost_inject(flat_prompt, current_prompt):
                        if not flat_prompt or not flat_prompt.strip():
                            return gr.update()
                        words = [w for w in flat_prompt.split(',') if w.strip()]
                        return _append_words_to_prompt(words, current_prompt)

                    omost_inject_btn.click(
                        omost_inject,
                        inputs=[omost_prompt_box, prompt],
                        outputs=[prompt],
                        queue=False, show_progress=False
                    )

                # === Embeddings event handlers ===
                def fetch_embedding_triggers_for_slot(emb_name, api_key_field):
                    if not emb_name or emb_name == 'None':
                        return gr.update(value='')
                    result = modules.civitai_api.fetch_model_triggers_combined(
                        filename=emb_name,
                        paths=[modules.config.path_embeddings],
                        kind='embedding',
                        api_key=_resolve_civitai_key(api_key_field),
                    )
                    return gr.update(value=modules.civitai_api.format_lora_triggers_display(result))

                for _dd, _disp in zip(embedding_dropdowns, embedding_trigger_displays):
                    _dd.change(
                        fetch_embedding_triggers_for_slot,
                        inputs=[_dd, civitai_api_key_input],
                        outputs=[_disp],
                        queue=True, show_progress=False
                    )

                def _format_embedding_token(emb_name, weight):
                    """Build the Fooocus activation token for an embedding slot."""
                    if not emb_name or emb_name == 'None':
                        return ''
                    stem = os.path.splitext(os.path.basename(str(emb_name)))[0]
                    try:
                        w = float(weight)
                    except (TypeError, ValueError):
                        w = 1.0
                    if abs(w - 1.0) < 1e-3:
                        return f'(embedding:{stem}:1.0)'
                    return f'(embedding:{stem}:{round(w, 2)})'

                _emb_dup_suffix_re = __import__('re').compile(r'\s*\(\d+\)\s*$')

                def _normalize_emb_token(s):
                    """Lowercase + strip Windows-duplicate suffixes like ' (1)', ' (2)'.

                    Same file copied as 'foo.safetensors' and 'foo (1).safetensors' in the
                    embeddings folder produces the suffixed stem, while CivitAI returns the
                    canonical 'foo'. Without this normalisation both end up appended to the
                    prompt as separate trigger words.
                    """
                    if not s:
                        return ''
                    return _emb_dup_suffix_re.sub('', str(s).strip()).lower()

                def _extract_extra_trigger_words(trigger_text, emb_stem):
                    """Return trigger-display words that aren't the filename stem itself
                    and aren't one of the placeholder '(...)' hints.

                    Stems and candidates are compared after stripping a trailing
                    Windows-duplicate marker (' (N)'), so a CivitAI canonical
                    'neg_realism512' deduplicates against a local 'neg_realism512 (1)'.
                    """
                    if not trigger_text or trigger_text.startswith('('):
                        return []
                    stem_norm = _normalize_emb_token(emb_stem)
                    seen = {stem_norm} if stem_norm else set()
                    words = []
                    for w in trigger_text.split(','):
                        w = w.strip()
                        if not w:
                            continue
                        key = _normalize_emb_token(w)
                        if key in seen:
                            continue
                        seen.add(key)
                        words.append(w)
                    return words

                def _append_with_dedup(current_text, items):
                    if not items:
                        return gr.update()
                    current_text = current_text or ''
                    existing_tokens = {t.strip().lower() for t in current_text.split(',') if t.strip()}
                    to_add = []
                    for item in items:
                        key = item.strip().lower()
                        if not key or key in existing_tokens:
                            continue
                        to_add.append(item)
                        existing_tokens.add(key)
                    if not to_add:
                        return gr.update()
                    sep = '' if not current_text else (', ' if not current_text.rstrip().endswith(',') else ' ')
                    return gr.update(value=current_text + sep + ', '.join(to_add))

                def _emb_canonical_keyword(stem):
                    """Stem with the Windows-duplicate ' (N)' suffix stripped — what
                    we want to insert as a bare keyword alongside the embedding tag.
                    """
                    if not stem:
                        return ''
                    return _emb_dup_suffix_re.sub('', str(stem)).strip()

                def _append_embedding_slot_to_textbox(emb_name, weight, trigger_text, current_text):
                    token = _format_embedding_token(emb_name, weight)
                    if not token:
                        return gr.update()
                    stem = os.path.splitext(os.path.basename(str(emb_name)))[0]
                    canonical = _emb_canonical_keyword(stem)
                    items = [token]
                    if canonical:
                        items.append(canonical)
                    items.extend(_extract_extra_trigger_words(trigger_text, stem))
                    return _append_with_dedup(current_text, items)

                for _btn, _dd, _wt, _disp in zip(embedding_insert_prompt_btns, embedding_dropdowns, embedding_weights, embedding_trigger_displays):
                    _btn.click(
                        _append_embedding_slot_to_textbox,
                        inputs=[_dd, _wt, _disp, prompt],
                        outputs=[prompt],
                        queue=False, show_progress=False
                    )
                for _btn, _dd, _wt, _disp in zip(embedding_insert_negative_btns, embedding_dropdowns, embedding_weights, embedding_trigger_displays):
                    _btn.click(
                        _append_embedding_slot_to_textbox,
                        inputs=[_dd, _wt, _disp, negative_prompt],
                        outputs=[negative_prompt],
                        queue=False, show_progress=False
                    )

                def _insert_all_embeddings(current_text, *args):
                    # args layout: [name1..N, weight1..N, enable1..N, trigger_display1..N]
                    n = len(args) // 4
                    names = args[:n]
                    weights = args[n:2 * n]
                    enables = args[2 * n:3 * n]
                    triggers = args[3 * n:4 * n]
                    items = []
                    for nm, wt, en, tt in zip(names, weights, enables, triggers):
                        if not en:
                            continue
                        tok = _format_embedding_token(nm, wt)
                        if not tok:
                            continue
                        items.append(tok)
                        stem = os.path.splitext(os.path.basename(str(nm)))[0]
                        canonical = _emb_canonical_keyword(stem)
                        if canonical:
                            items.append(canonical)
                        items.extend(_extract_extra_trigger_words(tt, stem))
                    return _append_with_dedup(current_text, items)

                _emb_enable_checkboxes = [embedding_ctrls[i] for i in range(0, len(embedding_ctrls), 3)]
                emb_insert_all_prompt_btn.click(
                    _insert_all_embeddings,
                    inputs=[prompt] + embedding_dropdowns + embedding_weights + _emb_enable_checkboxes + embedding_trigger_displays,
                    outputs=[prompt],
                    queue=False, show_progress=False
                )
                emb_insert_all_negative_btn.click(
                    _insert_all_embeddings,
                    inputs=[negative_prompt] + embedding_dropdowns + embedding_weights + _emb_enable_checkboxes + embedding_trigger_displays,
                    outputs=[negative_prompt],
                    queue=False, show_progress=False
                )

                # === Wildcards event handlers ===
                def _sanitize_wildcard_name(name):
                    """Strip any path/extension, keep only safe filename characters."""
                    if not name:
                        return ''
                    stem = os.path.splitext(os.path.basename(str(name).strip()))[0]
                    return ''.join(c for c in stem if c.isalnum() or c in ('_', '-'))

                def _wildcard_status_html(ok, msg):
                    color = '#4ecdc4' if ok else '#ff6b6b'
                    return (f'<div style="padding:6px 8px;border:1px solid {color};'
                            f'border-radius:6px;color:{color};font-size:12px;">{msg}</div>')

                def _load_wildcard_into_editor(name):
                    if not name or name == 'None':
                        return gr.update(value=''), gr.update(value='')
                    try:
                        filepath = os.path.join(modules.config.path_wildcards, str(name))
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                    except Exception as e:
                        return (gr.update(value=''),
                                gr.update(value=_wildcard_status_html(False, f'Cannot read {name}: {e}')))
                    return gr.update(value=content), gr.update(value='')

                wildcard_dropdown.change(
                    _load_wildcard_into_editor,
                    inputs=[wildcard_dropdown],
                    outputs=[wildcard_editor, wildcard_status],
                    queue=False, show_progress=False
                )

                def _save_wildcard(name, content):
                    if not name or name == 'None':
                        return (gr.update(),
                                gr.update(value=_wildcard_status_html(False, 'Pick a wildcard file to save into, or use Create New.')))
                    safe = _sanitize_wildcard_name(name)
                    if not safe:
                        return (gr.update(),
                                gr.update(value=_wildcard_status_html(False, f'Invalid wildcard name: {name}')))
                    filepath = os.path.join(modules.config.path_wildcards, f'{safe}.txt')
                    try:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(content or '')
                    except Exception as e:
                        return (gr.update(),
                                gr.update(value=_wildcard_status_html(False, f'Save failed: {e}')))
                    modules.config.update_files()
                    return (gr.update(choices=['None'] + modules.config.wildcard_filenames, value=f'{safe}.txt'),
                            gr.update(value=_wildcard_status_html(True, f'Saved {safe}.txt ({len((content or "").splitlines())} lines).')))

                wildcard_save_btn.click(
                    _save_wildcard,
                    inputs=[wildcard_dropdown, wildcard_editor],
                    outputs=[wildcard_dropdown, wildcard_status],
                    queue=False, show_progress=False
                )

                def _create_wildcard(new_name, content):
                    safe = _sanitize_wildcard_name(new_name)
                    if not safe:
                        return (gr.update(), gr.update(),
                                gr.update(value=_wildcard_status_html(False, 'Provide a name (letters, digits, _ -).')))
                    filepath = os.path.join(modules.config.path_wildcards, f'{safe}.txt')
                    if os.path.exists(filepath):
                        return (gr.update(), gr.update(),
                                gr.update(value=_wildcard_status_html(False, f'{safe}.txt already exists. Select it from the dropdown and Save to overwrite.')))
                    try:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(content or '')
                    except Exception as e:
                        return (gr.update(), gr.update(),
                                gr.update(value=_wildcard_status_html(False, f'Create failed: {e}')))
                    modules.config.update_files()
                    return (
                        gr.update(choices=['None'] + modules.config.wildcard_filenames, value=f'{safe}.txt'),
                        gr.update(value=''),
                        gr.update(value=_wildcard_status_html(True, f'Created {safe}.txt.')),
                    )

                wildcard_create_btn.click(
                    _create_wildcard,
                    inputs=[wildcard_new_name, wildcard_editor],
                    outputs=[wildcard_dropdown, wildcard_new_name, wildcard_status],
                    queue=False, show_progress=False
                )

                def _insert_wildcard_token(name, current_prompt):
                    if not name or name == 'None':
                        return gr.update()
                    stem = os.path.splitext(os.path.basename(str(name)))[0]
                    if not stem:
                        return gr.update()
                    token = f'__{stem}__'
                    current_prompt = current_prompt or ''
                    if token in current_prompt:
                        return gr.update()
                    sep = '' if not current_prompt else (', ' if not current_prompt.rstrip().endswith(',') else ' ')
                    return gr.update(value=current_prompt + sep + token)

                wildcard_insert_btn.click(
                    _insert_wildcard_token,
                    inputs=[wildcard_dropdown, prompt],
                    outputs=[prompt],
                    queue=False, show_progress=False
                )

                # === Restart UI (moved here : end of the Advanced tab) ===
                with gr.Row():
                    restart_ui_btn = gr.Button(
                        value='\U000026A0 Restart UI', variant='stop',
                        scale=1, min_width=110,
                        elem_classes='refresh_button')
                _restart_notice = gr.HTML(value='', visible=True)
                gr.HTML('<div style="font-size:11px;color:#888;padding:2px 6px;">'
                        'Restart reloads the whole Python process '
                        '(re-reads config.txt, reimports modules, re-loads the model).</div>')
                restart_ui_btn.click(
                    _restart_ui,
                    inputs=[],
                    outputs=[_restart_notice],
                    queue=False, show_progress=False
                )

        state_is_generating = gr.State(False)

        load_data_outputs = [advanced_checkbox, image_number, prompt, negative_prompt, style_selections,
                             performance_selection, overwrite_step, overwrite_switch, aspect_ratios_selection,
                             overwrite_width, overwrite_height, guidance_scale, sharpness, adm_scaler_positive,
                             adm_scaler_negative, adm_scaler_end, refiner_swap_method, adaptive_cfg, clip_skip,
                             base_model, refiner_model, refiner_switch, sampler_name, scheduler_name, vae_name,
                             seed_random, image_seed, inpaint_engine, inpaint_engine_state,
                             inpaint_mode] + enhance_inpaint_mode_ctrls + [generate_button,
                             load_parameter_button] + freeu_ctrls + lora_ctrls

        if not args_manager.args.disable_preset_selection:
            def _validate_preset_models(preset_prepared):
                """Validate that preset models/LoRAs exist locally; fallback to defaults."""
                # Validate base model
                base = preset_prepared.get('base_model')
                if base and base not in modules.config.model_filenames:
                    print(f'[preset] Base model "{base}" not found locally, falling back to default')
                    preset_prepared['base_model'] = modules.config.default_model

                # Validate LoRAs
                loras = preset_prepared.get('default_loras', [])
                if loras:
                    validated = []
                    for lora_entry in loras:
                        if isinstance(lora_entry, list) and len(lora_entry) >= 2:
                            enabled, name = lora_entry[0], lora_entry[1]
                            weight = lora_entry[2] if len(lora_entry) > 2 else 1.0
                            if name and name != 'None' and name not in modules.config.lora_filenames:
                                print(f'[preset] LoRA "{name}" not found locally, disabling')
                                validated.append([False, name, weight])
                            else:
                                validated.append(lora_entry)
                        else:
                            validated.append(lora_entry)
                    preset_prepared['default_loras'] = validated

                # Validate refiner
                refiner = preset_prepared.get('refiner_model')
                if refiner and refiner != 'None' and refiner not in modules.config.model_filenames:
                    print(f'[preset] Refiner "{refiner}" not found locally, disabling')
                    preset_prepared['refiner_model'] = 'None'

                return preset_prepared

            def preset_selection_change(preset, is_generating, inpaint_mode):
                preset_content = modules.config.try_get_preset_content(preset) if preset != 'initial' else {}
                preset_prepared = modules.meta_parser.parse_meta_from_preset(preset_content)

                default_model = preset_prepared.get('base_model')
                previous_default_models = preset_prepared.get('previous_default_models', [])
                checkpoint_downloads = preset_prepared.get('checkpoint_downloads', {})
                embeddings_downloads = preset_prepared.get('embeddings_downloads', {})
                lora_downloads = preset_prepared.get('lora_downloads', {})
                vae_downloads = preset_prepared.get('vae_downloads', {})

                preset_prepared['base_model'], preset_prepared['checkpoint_downloads'] = launch.download_models(
                    default_model, previous_default_models, checkpoint_downloads, embeddings_downloads, lora_downloads,
                    vae_downloads)

                # Validate models exist locally after download attempt
                preset_prepared = _validate_preset_models(preset_prepared)

                if 'prompt' in preset_prepared and preset_prepared.get('prompt') == '':
                    del preset_prepared['prompt']

                return modules.meta_parser.load_parameter_button_click(json.dumps(preset_prepared), is_generating, inpaint_mode)


            def inpaint_engine_state_change(inpaint_engine_version, *args):
                if inpaint_engine_version == 'empty':
                    inpaint_engine_version = modules.config.default_inpaint_engine_version

                result = []
                for inpaint_mode in args:
                    if inpaint_mode != modules.flags.inpaint_option_detail:
                        result.append(gr.update(value=inpaint_engine_version))
                    else:
                        result.append(gr.update())

                return result

            preset_selection.change(preset_selection_change, inputs=[preset_selection, state_is_generating, inpaint_mode], outputs=load_data_outputs, queue=False, show_progress=True) \
                .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                .then(lambda: None, _js='()=>{refresh_style_localization();}') \
                .then(inpaint_engine_state_change, inputs=[inpaint_engine_state] + enhance_inpaint_mode_ctrls, outputs=enhance_inpaint_engine_ctrls, queue=False, show_progress=False)

        performance_selection.change(lambda x: [gr.update(interactive=not flags.Performance.has_restricted_features(x))] * 11 +
                                               [gr.update(visible=not flags.Performance.has_restricted_features(x))] * 1 +
                                               [gr.update(value=flags.Performance.has_restricted_features(x))] * 1,
                                     inputs=performance_selection,
                                     outputs=[
                                         guidance_scale, sharpness, adm_scaler_end, adm_scaler_positive,
                                         adm_scaler_negative, refiner_switch, refiner_model, sampler_name,
                                         scheduler_name, adaptive_cfg, refiner_swap_method, negative_prompt, disable_intermediate_results
                                     ], queue=False, show_progress=False)

        output_format.input(lambda x: gr.update(output_format=x), inputs=output_format)

        advanced_checkbox.change(lambda x: gr.update(visible=x), advanced_checkbox, advanced_column,
                                 queue=False, show_progress=False) \
            .then(fn=lambda: None, _js='refresh_grid_delayed', queue=False, show_progress=False)

        def _toggle_extra_plugins(x):
            extra_ui.save_enabled(x)
            return gr.update(visible=x)
        extra_plugins_checkbox.change(_toggle_extra_plugins, extra_plugins_checkbox,
                                      extra_plugins_panel, queue=False, show_progress=False)

        inpaint_mode.change(inpaint_mode_change, inputs=[inpaint_mode, inpaint_engine_state], outputs=[
            inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
            inpaint_disable_initial_latent, inpaint_engine,
            inpaint_strength, inpaint_respective_field
        ], show_progress=False, queue=False)

        # load configured default_inpaint_method
        default_inpaint_ctrls = [inpaint_mode, inpaint_disable_initial_latent, inpaint_engine, inpaint_strength, inpaint_respective_field]
        for mode, disable_initial_latent, engine, strength, respective_field in [default_inpaint_ctrls] + enhance_inpaint_update_ctrls:
            shared.gradio_root.load(inpaint_mode_change, inputs=[mode, inpaint_engine_state], outputs=[
                inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts, disable_initial_latent,
                engine, strength, respective_field
            ], show_progress=False, queue=False)

        generate_mask_button.click(fn=generate_mask,
                                   inputs=[inpaint_input_image, inpaint_mask_model, inpaint_mask_cloth_category,
                                           inpaint_mask_dino_prompt_text, inpaint_mask_sam_model,
                                           inpaint_mask_box_threshold, inpaint_mask_text_threshold,
                                           inpaint_mask_sam_max_detections, dino_erode_or_dilate, debugging_dino],
                                   outputs=inpaint_mask_image, show_progress=True, queue=True)

        ctrls = [currentTask, generate_image_grid]
        ctrls += [
            prompt, negative_prompt, style_selections,
            performance_selection, aspect_ratios_selection, image_number, output_format, image_seed,
            read_wildcards_in_order, sharpness, guidance_scale
        ]

        ctrls += [base_model, refiner_model, refiner_switch] + lora_ctrls
        ctrls += [input_image_checkbox, current_tab]
        ctrls += [uov_method, uov_input_image]
        ctrls += [outpaint_selections, inpaint_input_image, inpaint_additional_prompt, inpaint_mask_image]
        ctrls += [disable_preview, disable_intermediate_results, disable_seed_increment, black_out_nsfw]
        ctrls += [adm_scaler_positive, adm_scaler_negative, adm_scaler_end, adaptive_cfg, clip_skip]
        ctrls += [sampler_name, scheduler_name, vae_name]
        ctrls += [overwrite_step, overwrite_switch, overwrite_width, overwrite_height, overwrite_vary_strength]
        ctrls += [overwrite_upscale_strength, mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint]
        ctrls += [use_aspect_for_vary]  # custom-6: force aspect-ratio for Vary/Upscale
        ctrls += [custom_res_enabled, custom_ratio_w, custom_ratio_h, custom_res_mode, custom_res_size]  # custom-7: custom resolution override
        ctrls += [debugging_cn_preprocessor, skipping_cn_preprocessor, canny_low_threshold, canny_high_threshold]
        ctrls += [refiner_swap_method, controlnet_softness]
        ctrls += freeu_ctrls
        ctrls += inpaint_ctrls

        if not args_manager.args.disable_image_log:
            ctrls += [save_final_enhanced_image_only]

        if not args_manager.args.disable_metadata:
            ctrls += [save_metadata_to_images, metadata_scheme]

        ctrls += ip_ctrls
        ctrls += [debugging_dino, dino_erode_or_dilate, debugging_enhance_masks_checkbox,
                  enhance_input_image, enhance_checkbox, enhance_uov_method, enhance_uov_processing_order,
                  enhance_uov_prompt_type]
        ctrls += enhance_ctrls
        # custom-10: upscaler model + face restoration controls (appended at end
        # to preserve all existing arg ordering for the AsyncTask parser).
        ctrls += [upscaler_model_name, face_restore_model, face_restore_visibility, face_restore_order]

        def _looks_like_a1111_meta(text):
            """Detect A1111-style metadata: prompt text followed by a params line
            containing key: value pairs like 'Steps: 25, CFG scale: 5, ...'"""
            if not text or len(text) < 20:
                return False
            lines = text.strip().split('\n')
            last = lines[-1].strip()
            import re as _re
            pairs = _re.findall(r'\w[\w \-/]+:\s*[^,]+', last)
            has_steps = 'Steps:' in last
            has_cfg = 'CFG scale:' in last or 'CFG Scale:' in last
            return has_steps and has_cfg and len(pairs) >= 3

        def parse_meta(raw_prompt_txt, is_generating):
            loaded_json = None
            if is_json(raw_prompt_txt):
                loaded_json = json.loads(raw_prompt_txt)

            # Detect A1111-style pasted metadata (prompt + params line)
            if loaded_json is None and _looks_like_a1111_meta(raw_prompt_txt):
                try:
                    parser = modules.meta_parser.get_metadata_parser(modules.flags.MetadataScheme.A1111)
                    loaded_json = parser.to_json(raw_prompt_txt)
                except Exception as e:
                    print(f'[parse_meta] A1111 parse failed: {e}')
                    loaded_json = None

            if loaded_json is None:
                if is_generating:
                    return gr.update(), gr.update(), gr.update()
                else:
                    return gr.update(), gr.update(visible=True), gr.update(visible=False)

            return json.dumps(loaded_json), gr.update(visible=False), gr.update(visible=True)

        prompt.input(parse_meta, inputs=[prompt, state_is_generating], outputs=[prompt, generate_button, load_parameter_button], queue=False, show_progress=False)

        load_parameter_button.click(modules.meta_parser.load_parameter_button_click, inputs=[prompt, state_is_generating, inpaint_mode], outputs=load_data_outputs, queue=False, show_progress=False)

        def trigger_metadata_import(file, state_is_generating):
            parameters, metadata_scheme = modules.meta_parser.read_info_from_image(file)
            if parameters is None:
                print('Could not find metadata in the image!')
                parsed_parameters = {}
            else:
                metadata_parser = modules.meta_parser.get_metadata_parser(metadata_scheme)
                parsed_parameters = metadata_parser.to_json(parameters)

            return modules.meta_parser.load_parameter_button_click(parsed_parameters, state_is_generating, inpaint_mode)

        metadata_import_button.click(trigger_metadata_import, inputs=[metadata_input_image, state_is_generating], outputs=load_data_outputs, queue=False, show_progress=True) \
            .then(style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False)

        # === custom-14: cablage Job Queue =================================
        if modules.config.job_queue_enabled():
            from modules import job_queue as jq

            def queue_refresh():
                n = len(jq.queue)
                btn = '+ Queue' if n == 0 else f'+ Queue ({n})'
                return gr.update(choices=jq.queue.labels(), value=None), jq.queue.status_text(), gr.update(value=btn)

            def queue_add(*args):
                import traceback
                try:
                    a = list(args)
                    a.pop(0)  # currentTask (meme convention que get_task)
                    label = jq.make_label(a)
                    pos = jq.queue.add(a, label)
                    if pos < 0:
                        print(f'[JobQueue] File pleine ({jq.queue.max_jobs} jobs max), ajout refuse.')
                    else:
                        print(f'[JobQueue] Ajout #{pos} : {label}')
                except Exception:
                    traceback.print_exc()
                return queue_refresh()

            def queue_remove(sel):
                jq.queue.remove(jq.parse_index(sel))
                return queue_refresh()

            def queue_up(sel):
                jq.queue.move(jq.parse_index(sel), -1)
                return queue_refresh()

            def queue_down(sel):
                jq.queue.move(jq.parse_index(sel), +1)
                return queue_refresh()

            def queue_clear():
                jq.queue.clear()
                return queue_refresh()

            jq.queue.max_jobs = int(modules.config.job_queue_setting('max_jobs') or 50)

            queue_checkbox.change(lambda x: gr.update(visible=x), inputs=queue_checkbox,
                                  outputs=job_queue_panel, queue=False, show_progress=False) \
                .then(fn=queue_refresh, outputs=[queue_display, queue_status, queue_add_button],
                      queue=False, show_progress=False)

            queue_add_button.click(fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed,
                                   queue=False, show_progress=False) \
                .then(fn=queue_add, inputs=ctrls, outputs=[queue_display, queue_status, queue_add_button],
                      show_progress=False)
            queue_remove_button.click(queue_remove, inputs=queue_display, outputs=[queue_display, queue_status, queue_add_button], queue=False, show_progress=False)
            queue_up_button.click(queue_up, inputs=queue_display, outputs=[queue_display, queue_status, queue_add_button], queue=False, show_progress=False)
            queue_down_button.click(queue_down, inputs=queue_display, outputs=[queue_display, queue_status, queue_add_button], queue=False, show_progress=False)
            queue_clear_button.click(queue_clear, outputs=[queue_display, queue_status, queue_add_button], queue=False, show_progress=False)

            queue_run_button.click(lambda: (gr.update(visible=True, interactive=True), gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=False), [], True),
                                   outputs=[stop_button, skip_button, generate_button, gallery, state_is_generating]) \
                .then(fn=run_queue_clicked, outputs=[progress_html, progress_window, progress_gallery, gallery]) \
                .then(lambda: (gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=False), gr.update(visible=False, interactive=False), False),
                      outputs=[generate_button, stop_button, skip_button, state_is_generating]) \
                .then(fn=queue_refresh, outputs=[queue_display, queue_status, queue_add_button], queue=False, show_progress=False) \
                .then(fn=update_history_link, outputs=history_link) \
                .then(fn=lambda: None, _js='playNotification').then(fn=lambda: None, _js='refresh_grid_delayed')

        generate_button.click(lambda: (gr.update(visible=True, interactive=True), gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=False), [], True),
                              outputs=[stop_button, skip_button, generate_button, gallery, state_is_generating]) \
            .then(fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed) \
            .then(fn=get_task, inputs=ctrls, outputs=currentTask) \
            .then(fn=generate_clicked, inputs=currentTask, outputs=[progress_html, progress_window, progress_gallery, gallery]) \
            .then(lambda: (gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=False), gr.update(visible=False, interactive=False), False),
                  outputs=[generate_button, stop_button, skip_button, state_is_generating]) \
            .then(fn=update_history_link, outputs=history_link) \
            .then(fn=lambda: None, _js='playNotification').then(fn=lambda: None, _js='refresh_grid_delayed')

        reset_button.click(lambda: [worker.AsyncTask(args=[]), False, gr.update(visible=True, interactive=True)] +
                                   [gr.update(visible=False)] * 6 +
                                   [gr.update(visible=True, value=[])],
                           outputs=[currentTask, state_is_generating, generate_button,
                                    reset_button, stop_button, skip_button,
                                    progress_html, progress_window, progress_gallery, gallery],
                           queue=False)

        for notification_file in ['notification.ogg', 'notification.mp3']:
            if os.path.exists(notification_file):
                gr.Audio(interactive=False, value=notification_file, elem_id='audio_notification', visible=False)
                break

        def trigger_describe(modes, img, apply_styles):
            describe_prompts = []
            styles = set()

            if flags.describe_type_photo in modes:
                from extras.interrogate import default_interrogator as default_interrogator_photo
                describe_prompts.append(default_interrogator_photo(img))
                styles.update(["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"])

            if flags.describe_type_anime in modes:
                from extras.wd14tagger import default_interrogator as default_interrogator_anime
                describe_prompts.append(default_interrogator_anime(img))
                styles.update(["Fooocus V2", "Fooocus Masterpiece"])

            if len(styles) == 0 or not apply_styles:
                styles = gr.update()
            else:
                styles = list(styles)

            if len(describe_prompts) == 0:
                describe_prompt = gr.update()
            else:
                describe_prompt = ', '.join(describe_prompts)

            return describe_prompt, styles

        describe_btn.click(trigger_describe, inputs=[describe_methods, describe_input_image, describe_apply_styles],
                           outputs=[prompt, style_selections], show_progress=True, queue=True) \
            .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
            .then(lambda: None, _js='()=>{refresh_style_localization();}')

        if args_manager.args.enable_auto_describe_image:
            def trigger_auto_describe(mode, img, prompt, apply_styles):
                # keep prompt if not empty
                if prompt == '':
                    return trigger_describe(mode, img, apply_styles)
                return gr.update(), gr.update()

            uov_input_image.upload(trigger_auto_describe, inputs=[describe_methods, uov_input_image, prompt, describe_apply_styles],
                                   outputs=[prompt, style_selections], show_progress=True, queue=True) \
                .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                .then(lambda: None, _js='()=>{refresh_style_localization();}')

            enhance_input_image.upload(lambda: gr.update(value=True), outputs=enhance_checkbox, queue=False, show_progress=False) \
                .then(trigger_auto_describe, inputs=[describe_methods, enhance_input_image, prompt, describe_apply_styles],
                      outputs=[prompt, style_selections], show_progress=True, queue=True) \
                .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                .then(lambda: None, _js='()=>{refresh_style_localization();}')

def dump_default_english_config():
    from modules.localization import dump_english_config
    dump_english_config(grh.all_components)


# dump_default_english_config()

shared.gradio_root.launch(
    inbrowser=args_manager.args.in_browser,
    server_name=args_manager.args.listen,
    server_port=args_manager.args.port,
    share=args_manager.args.share,
    auth=check_auth if (args_manager.args.share or args_manager.args.listen) and auth_enabled else None,
    allowed_paths=[modules.config.path_outputs],
    blocked_paths=[constants.AUTH_FILENAME]
)
