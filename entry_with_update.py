import os
import sys


root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)


# custom-19 : l'auto-update d'origine (pygit2 + reset --hard) ecrasait sans un mot
# les fichiers suivis modifies localement. update_check ne met a jour que quand
# c'est sur, et le propose au lieu de l'imposer (voir update_check.py).
try:
    import update_check
    update_check.boot()
except Exception as e:
    print('[Update] Verification impossible, demarrage tel quel.')
    print(str(e))

from launch import *
