# Onglet Extra — intégration dans Fooocus

Sous-système autonome qui ajoute un onglet **Extra** à Fooocus : un gestionnaire
de plugins externes installables depuis une URL GitHub, qui tournent dans leur
propre venv et dialoguent avec Fooocus en CLI (subprocess).

Conçu pour être portable. Aucun fichier Fooocus existant n'est modifié par le
sous-système lui-même. L'intégration tient en un patch de quelques lignes.

## Où poser le dossier

Place `extra_plugins/` à la racine du code Fooocus, à côté de `webui.py` :

```
Fooocus/
  webui.py
  extra_plugins/        <- ce dossier
    __init__.py
    manifest.py
    runner.py
    installer.py
    registry.py
    ui.py
    installed/          <- les plugins clonés atterrissent ici (cree au besoin)
    outputs/            <- images de sortie des plugins
```

Pour porter vers Fooocus2026 : copie `extra_plugins/` tel quel et applique le
même patch. La seule dépendance versionnée est le hook d'offload (voir plus bas),
qui est tolérant aux pannes.

## Le patch webui.py (3 endroits)

Le panneau Extra s'active par une case dans la colonne Advanced, sur le modèle de
l'Asset Browser (case d'activation dans Advanced).

1. En haut du fichier, avec les autres imports :

```python
from extra_plugins import ui as extra_ui
```

2. Panneau Extra masqué, au premier niveau de la colonne de gauche (à côté de
`image_input_panel` / `enhance_input_panel`) :

```python
            with gr.Row(visible=False) as extra_plugins_panel:
                extra_ui.build_extra_panel()
```

3. Case d'activation dans la colonne Advanced (ici en tête de l'onglet Settings) :

```python
            with gr.Tab(label='Settings'):
                extra_plugins_checkbox = gr.Checkbox(label='\U0001F9E9 Extra Plugins', value=False,
                                                     info='Affiche le panneau Extra (upscalers externes via plugins).')
```

4. Et le câblage du toggle, là où tous les composants existent (à côté du
`advanced_checkbox.change`, vers la fin du fichier) :

```python
        extra_plugins_checkbox.change(lambda x: gr.update(visible=x), extra_plugins_checkbox,
                                      extra_plugins_panel, queue=False, show_progress=False)
```

Usage : coche **Advanced**, onglet **Settings**, puis **Extra Plugins**. Le panneau
apparaît. C'est le même geste que pour activer l'Asset Browser.

> Variante : pour un onglet toujours visible parmi les entrées d'image, mets
> `with gr.Tab(label='Extra'): extra_ui.build_extra_panel()` dans le `gr.Tabs` des
> entrées (mais il reste alors masqué tant que la case Input Image n'est pas cochée).

## Le hook d'offload VRAM

Avant chaque appel plugin, le runner appelle `offload_host_models()` qui décharge
le modèle SDXL de Fooocus via `ldm_patched.modules.model_management.unload_all_models()`.
Si l'API n'est pas trouvée (autre version de Fooocus), c'est un no-op silencieux,
l'appel plugin se fait quand même.

Pourquoi c'est nécessaire : mesures sur RTX 5090 32 Go, le pipeline crispz 2K sans
offload sature la carte (pic réservé 32.35 Go). En déchargeant le modèle hôte avant
l'appel, crispz récupère le GPU. Fooocus rechargera son modèle à la génération
suivante. En complément, l'UI expose `cpu_offload` (mettre `sequential` ~9 Go) et
`refine_tile` (1024 pour le 4K) côté plugin.

## Mode serveur et libération VRAM (custom-20)

Quand le manifeste d'un plugin déclare un bloc `server` (commande `launch` avec
`--port`), l'onglet du plugin propose **Server mode**. `extra_plugins/server.py`
lance le serveur au premier appel, attend `GET /health`, puis envoie
`POST /upscale` avec le chemin d'entrée et les params du manifeste **par clé**
(`runner.build_server_payload`). Le modèle reste chargé entre deux appels.

Pour que le modèle chaud du plugin ne dispute pas la carte à SDXL, ajoute ce
hook au début de la génération Fooocus (dans `execute_task_streaming`, juste
après le test `len(task.args) == 0`) :

```python
    try:
        from extra_plugins import server as extra_server
        extra_server.release_vram_all()   # POST /unload aux serveurs chauds, no-op sinon
    except Exception:
        pass
```

Serveur introuvable, fastapi absent, port pris : le statut l'annonce avec la fin
du journal serveur et l'appel passe en CLI. Un serveur déjà actif sur le port
est réutilisé, jamais tué. Les serveurs lancés par Fooocus sont arrêtés à la
sortie (`atexit`), au Restart UI et avant la mise à jour du plugin.

## Plusieurs actions par plugin (custom-24)

Un manifeste peut déclarer un bloc `actions`. Sans lui, le plugin a une seule
action implicite *Upscale* (tous les params, mode serveur autorisé) : les
manifestes existants ne changent pas.

```json
"actions": [
  {"id": "upscale", "label": "Upscale", "server": true},
  {
    "id": "faceswap", "label": "Face swap",
    "args": ["--faceswap-only"],
    "params": [],
    "image_params": [{"key": "faceswap_src", "label": "Source face", "arg": "--faceswap-src"}],
    "note": "texte d'aide affiché sous le bouton"
  }
]
```

- `params` : clés de la section `params` montrées pour cette action (absent = toutes).
- `args` : flags ajoutés juste après l'image d'entrée.
- `image_params` : une image chargée dans l'onglet est écrite en temp puis passée
  en `arg <chemin>`.
- `server` : l'action peut-elle passer par le serveur (`POST /upscale`) ? Par défaut
  vrai seulement sans `args` ni `image_params`.

L'onglet du plugin montre un sous-onglet par action. Le dossier ESRGAN n'apparaît
que si l'action utilise le param qui a un `choices_cmd`. Le contrat de sortie reste
`print_output` pour toutes les actions.

## Mettre à jour un plugin (custom-20)

Gestionnaire → **Mises à jour** : **Vérifier** fait un `git fetch` et liste les
commits, **Mettre à jour** applique `git merge --ff-only`. Refus, sans rien
toucher, si un commit touche un fichier modifié dans le plugin, s'il ajoute un
chemin présent hors de git, ou si la branche a divergé. Seules les étapes
`... -r <fichier>` de la stratégie d'environnement sont rejouées, et seulement
si ce fichier a changé (jamais la création du venv ni torch). Une étape marquée
`"rerun_with_deps": true` est rejouée juste après, chaque fois que les deps l'ont
été : c'est la place d'un correctif posé après `pip -r`, que la résolution des
deps défait à chaque fois (ex. crispz : `pillow==12.3.0` en `--no-deps`, que la
borne `pillow<12` de gradio redescend). La stratégie
choisie à l'install est mémorisée dans `settings.json`. Si le manifeste a
changé, un Restart UI reconstruit l'onglet.

## Prérequis

- `git` accessible sur le PATH (pour le clone). Sur Windows, installer Git for Windows.
- Connexion réseau pour le premier clone et l'install du venv (torch cu128 + deps).

## Comment ça marche

1. **Gestionnaire** : colle une URL GitHub, choisis la stratégie d'environnement
   (`fresh_venv` par défaut, ou `reuse_python` avec un interpréteur qui a déjà le bon
   torch), clique Installer. Le sous-système clone, lit `fooocus_extra.json`, crée le
   venv et installe les deps.
2. Relance Fooocus : un sous-onglet apparaît pour le plugin, avec ses contrôles
   générés depuis le manifeste (l'ajout d'onglet à chaud n'est pas supporté par
   Gradio, comme les custom-nodes ComfyUI demandent un restart).
3. **Upscale** : charge une image, règle les paramètres, clique Upscale. Le runner
   écrit l'image en temp, décharge le modèle hôte, lance le plugin en subprocess,
   parse le chemin de sortie sur stdout (contrat `--print-output`) et réaffiche le
   résultat. La ligne `[VRAM]` de stderr remonte dans le statut.

## Contrat attendu d'un plugin

Un dépôt compatible expose à sa racine un `fooocus_extra.json` (manifest_version 1)
déclarant : identité, stratégie d'environnement, commande CLI, mapping des params
vers les flags, et le mode de sortie `print_output` (le plugin imprime le chemin
absolu de l'image sur stdout, rien d'autre). Voir le manifeste de référence dans le
dépôt crispz.
