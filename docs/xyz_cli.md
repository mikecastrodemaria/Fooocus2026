# xyz_cli.py — grilles X/Y/Z en ligne de commande

Lance une grille de comparaison sans navigateur, avec les memes moteurs que
l'UI (queue custom-14, assemblage custom-15). Ideal pour les batchs nocturnes.

## Usage de base

```bat
cd E:\Fooocus_win64_2-5-0\Fooocus
..\python_embeded\Scripts\python.exe -s xyz_cli.py ^
    --prompt "a castle at dawn, masterpiece" ^
    --x "CFG:3,5,7" --y "Steps:20,40" ^
    --seed 12345 --dry-run
```

`--dry-run` construit et valide les jobs sans rien generer : commencez
toujours par la. Enlevez-le pour lancer la generation.

## Arguments

| Argument | Role |
|---|---|
| `--prompt` (requis) | Prompt positif, identique pour toutes les cases. |
| `--negative` | Prompt negatif (defaut : celui de config.txt). |
| `--x "Axe:v1,v2,..."` (requis) | Axe X. Axes : CFG, Steps, Sampler, Scheduler, Sharpness, Checkpoint, LoRA 1 weight, Preset, Prompt S/R (noms insensibles a la casse). |
| `--y`, `--z` | Axes optionnels, meme format. Z = une planche par valeur. |
| `--preset nom` | Applique un preset au snapshot de base (nom partiel accepte). |
| `--seed N` | Seed fixe (defaut : aleatoire, identique pour toutes les cases). |
| `--performance` | Quality, Speed, etc. (defaut : config.txt). |
| `--aspect` | Ex. `1152×896` (defaut : config.txt). |
| `--output-format` | png, jpeg, webp. |
| `--dry-run` | Valide tout, ne genere rien. |

Tout flag inconnu est transmis a Fooocus (`--always-gpu`,
`--disable-offload-from-vram`, etc.). Pour retrouver les performances du
profil 5090, reprenez les flags de `run_quality_rtx5090.bat`.

## Notes

- Checkpoint et Preset acceptent les noms partiels (`juggernaut` suffit si
  unique) ; erreur claire si introuvable ou ambigu.
- Prompt S/R : premiere valeur = terme cherche dans le prompt, suivantes =
  remplacements. Guillemets CSV pour proteger une virgule :
  `--x "Prompt S/R:dawn,dusk,\"stormy night, lightning\""`.
- Purge VRAM automatique entre deux cases qui changent de checkpoint.
- Ctrl+C : arret propre, les cases deja generees restent dans outputs/.
- Les planches finissent dans `outputs/xyz_grids/`, les images individuelles
  dans outputs/ comme d'habitude (Asset Browser inclus).
- Garde-fou : chaque job est valide contre AsyncTask avant la moindre
  generation ; si l'ordre des ctrls a change apres une mise a jour, le
  script refuse de demarrer (mettre a jour build_base_args()).

## Exemple : script de nuit

Voir `run_xyz_night.bat` a la racine du bundle : trois grilles enchainees
(calibrage CFG/steps, duel de checkpoints, variantes de prompt) pendant que
vous dormez.
