"""custom-14 — Job Queue (file d'attente de generations).

File en memoire, thread-safe, consommee par run_queue_clicked (webui.py).
Chaque job est un snapshot complet des ctrls au moment de l'ajout, donc
totalement independant de l'etat de l'UI au moment de l'execution.

Import-safe : stdlib uniquement, aucun import modules.config ici, pour que
stop_clicked puisse l'importer sans cout meme quand la feature est off.

Stop pendant Run queue = interrompt le job courant + met la file en pause
(les jobs restants attendent un nouveau Run queue). Rien n'est jamais perdu.
"""
import re
import threading
import time


class Job:
    __slots__ = ('args', 'label', 'added_at', 'meta')

    def __init__(self, args, label, meta=None):
        self.args = list(args)
        self.label = str(label)
        self.added_at = time.time()
        self.meta = meta  # custom-15 : {'group','x','y','z'} pour les jobs XYZ


class JobQueue:
    def __init__(self, max_jobs=50):
        self._jobs = []
        self._lock = threading.Lock()
        self.max_jobs = max_jobs
        self.paused = False
        self.current_task = None  # AsyncTask en cours quand Run queue tourne

    def __len__(self):
        with self._lock:
            return len(self._jobs)

    def add(self, args, label, meta=None):
        """Ajoute un job. Renvoie sa position (1-based), ou -1 si file pleine."""
        with self._lock:
            if len(self._jobs) >= self.max_jobs:
                return -1
            self._jobs.append(Job(args, label, meta))
            return len(self._jobs)

    def pop_next(self):
        with self._lock:
            return self._jobs.pop(0) if self._jobs else None

    def remove(self, index):
        with self._lock:
            if index is None or not (0 <= index < len(self._jobs)):
                return False
            self._jobs.pop(index)
            return True

    def move(self, index, delta):
        with self._lock:
            if index is None or not (0 <= index < len(self._jobs)):
                return index
            new = max(0, min(len(self._jobs) - 1, index + delta))
            self._jobs.insert(new, self._jobs.pop(index))
            return new

    def clear(self):
        with self._lock:
            self._jobs.clear()

    def labels(self):
        with self._lock:
            return [f'#{i + 1} | {j.label}' for i, j in enumerate(self._jobs)]

    def status_text(self):
        n = len(self)
        if n == 0:
            return 'File vide.'
        state = 'en pause (Stop) — relancez Run queue' if self.paused else 'prete'
        return f'{n} job(s) en attente, file {state}.'


queue = JobQueue()


def parse_index(selection):
    """'#3 | ...' -> 2 (0-based). None ou invalide -> None."""
    if not selection:
        return None
    m = re.match(r'#(\d+)', str(selection).strip())
    return int(m.group(1)) - 1 if m else None


def make_label(args):
    """Etiquette lisible depuis le snapshot ctrls (apres pop du currentTask).
    Positions calees sur AsyncTask.__init__ : 1=prompt, 4=performance,
    6=image_number, 8=seed, 12=base_model. Defensive : jamais d'exception."""
    try:
        prompt = ' '.join(str(args[1]).split()) or '(prompt vide)'
        if len(prompt) > 60:
            prompt = prompt[:57] + '...'
        base = str(args[12]).rsplit('.', 1)[0]
        return f'{prompt} | {base} | {args[4]} | seed {args[8]} | x{args[6]}'
    except Exception:
        return 'job'
