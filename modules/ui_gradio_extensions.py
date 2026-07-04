# based on https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/v1.6.0/modules/ui_gradio_extensions.py

import os
import gradio as gr
import args_manager

from modules.localization import localization_js


GradioTemplateResponseOriginal = gr.routes.templates.TemplateResponse

modules_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.dirname(modules_path)


def webpath(fn):
    if fn.startswith(script_path):
        web_path = os.path.relpath(fn, script_path).replace('\\', '/')
    else:
        web_path = os.path.abspath(fn)

    return f'file={web_path}?{os.path.getmtime(fn)}'


def javascript_html():
    script_js_path = webpath('javascript/script.js')
    context_menus_js_path = webpath('javascript/contextMenus.js')
    localization_js_path = webpath('javascript/localization.js')
    zoom_js_path = webpath('javascript/zoom.js')
    edit_attention_js_path = webpath('javascript/edit-attention.js')
    viewer_js_path = webpath('javascript/viewer.js')
    image_viewer_js_path = webpath('javascript/imageviewer.js')
    samples_path = webpath(os.path.abspath('./sdxl_styles/samples/fooocus_v2.jpg'))
    head = f'<script type="text/javascript">{localization_js(args_manager.args.language)}</script>\n'
    head += f'<script type="text/javascript" src="{script_js_path}"></script>\n'
    head += f'<script type="text/javascript" src="{context_menus_js_path}"></script>\n'
    head += f'<script type="text/javascript" src="{localization_js_path}"></script>\n'
    head += f'<script type="text/javascript" src="{zoom_js_path}"></script>\n'
    head += f'<script type="text/javascript" src="{edit_attention_js_path}"></script>\n'
    head += f'<script type="text/javascript" src="{viewer_js_path}"></script>\n'
    head += f'<script type="text/javascript" src="{image_viewer_js_path}"></script>\n'
    head += f'<meta name="samples-path" content="{samples_path}">\n'

    # Asset Browser icons script
    import modules.config
    ab_index = os.path.join(modules.config.path_outputs, 'index.html').replace('\\', '/')
    head += f'<meta name="ab-base-url" content="/file={ab_index}">\n'
    ab_icons_js_path = webpath('javascript/ab_icons.js')
    head += f'<script type="text/javascript" src="{ab_icons_js_path}"></script>\n'

    # custom-12: Tag Autocomplete (optionnel, tag_autocomplete.enabled dans config.txt)
    if modules.config.tag_autocomplete_enabled():
        import json as _json
        import modules.tag_autocomplete as _tag_ac
        _sources = _tag_ac.init()
        _ta_cfg = {
            'sources': {s: '/' + webpath(_tag_ac.source_csv_path(s))
                        for s in _sources if os.path.isfile(_tag_ac.source_csv_path(s))},
            'localAssets': ('/' + webpath(_tag_ac.local_assets_path()))
                           if os.path.isfile(_tag_ac.local_assets_path()) else None,
            'minChars': modules.config.tag_autocomplete_setting('min_chars'),
            'maxResults': modules.config.tag_autocomplete_setting('max_results'),
            'replaceUnderscores': bool(modules.config.tag_autocomplete_setting('replace_underscores')),
            'insertComma': bool(modules.config.tag_autocomplete_setting('insert_comma')),
        }
        _ta_json = _json.dumps(_ta_cfg).replace("'", '&#39;')
        head += f"<meta name=\"ta-config\" content='{_ta_json}'>\n"
        tag_ac_js_path = webpath('javascript/tag_autocomplete.js')
        head += f'<script type="text/javascript" src="{tag_ac_js_path}"></script>\n'

    if args_manager.args.theme:
        head += f'<script type="text/javascript">set_theme(\"{args_manager.args.theme}\");</script>\n'

    return head


def css_html():
    style_css_path = webpath('css/style.css')
    head = f'<link rel="stylesheet" property="stylesheet" href="{style_css_path}">'
    return head


def reload_javascript():
    js = javascript_html()
    css = css_html()

    def template_response(*args, **kwargs):
        res = GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b'</head>', f'{js}</head>'.encode("utf8"))
        res.body = res.body.replace(b'</body>', f'{css}</body>'.encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response
