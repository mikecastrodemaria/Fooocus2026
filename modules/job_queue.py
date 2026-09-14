"""custom-14 — Job Queue (file d'attente de generations).

File en memoire, thread-safe, consommee par queue_runner (webui.py).
Chaque job est un snapshot complet des ctrls au moment de l'ajout, donc
totalement independant de l'etat de l'UI au moment de l'execution.

Import-safe : stdlib uniquement a l'import, aucun import modules.config ici, pour
que stop_clicked puisse l'importer sans cout meme quand la feature est off. numpy
et PIL ne sont importes que pour (de)serialiser les images d'entree (custom-21).

Stop pendant Run queue = interrompt le job courant + met la file en pause
(les jobs restants attendent un nouveau Run queue). Rien n'est jamais perdu.

custom-17 : la file est desormais drainee par un *runner vivant* (voir
queue_runner() dans webui.py). Le runner ne sort plus quand la file est vide :
il idle et ramasse les jobs ajoutes a chaud, ce qui permet de lancer une
generation pendant qu'une autre tourne. Un seul runner a la fois, garanti par
try_acquire_runner() / release_runner().

custom-21 : la file SURVIT a un redemarrage et a un crash. Elle est ecrite dans
cache/job_queue/ a chaque mutation et apres chaque job ; les images d'entree
(upscale, inpaint, image prompts) sont stockees a part, dedupliquees par
empreinte. Le job en cours reste EN TETE de file jusqu'a ce qu'il soit termine :
un Stop le laisse en file, il sera re-execute entier a la reprise (avant, il
etait retire au demarrage et perdu). Pause douce : finir le job en cours puis
suspendre. Au rechargement, un snapshot dont le nombre de ctrls ne correspond
plus a l'UI (Fooocus mis a jour entre-temps) est ecarte et sauvegarde a part,
jamais rejoue avec des arguments decales.
"""
import hashlib
import json
import os
import re
import threading
import time
import uuid

PERSIST_VERSION = 1
QUEUE_FILE = 'queue.json'
ASSETS_DIR = 'assets'


class UnserializableJob(Exception):
    """Un ctrl du snapshot n'a pas de representation sur disque."""


class Job:
    __slots__ = ('args', 'label', 'added_at', 'meta', 'id')

    def __init__(self, args, label, meta=None, added_at=None, job_id=None):
        self.args = list(args)
        self.label = str(label)
        self.added_at = added_at if added_at is not None else time.time()
        self.meta = meta  # custom-15 : {'group','x','y','z'} pour les jobs XYZ
        self.id = job_id or uuid.uuid4().hex[:12]


# --------------------------------------------------------------- serialisation

def _save_array(arr, assets_dir, used):
    """Ecrit un ndarray une seule fois (nom = empreinte du contenu). PNG sans perte
    pour les images uint8, .npy pour le reste (masques float, etc.)."""
    import numpy as np
    a = np.ascontiguousarray(arr)
    h = hashlib.sha1()
    h.update(repr((a.dtype.str, a.shape)).encode('utf-8'))
    h.update(a.tobytes())
    png_ok = a.dtype == np.uint8 and (a.ndim == 2 or (a.ndim == 3 and a.shape[2] in (1, 3, 4)))
    name = h.hexdigest()[:20] + ('.png' if png_ok else '.npy')
    path = os.path.join(assets_dir, name)
    if not os.path.isfile(path):
        os.makedirs(assets_dir, exist_ok=True)
        tmp = path + '.tmp'
        if png_ok:
            from PIL import Image
            img = a[:, :, 0] if (a.ndim == 3 and a.shape[2] == 1) else a
            Image.fromarray(img).save(tmp, format='PNG')
        else:
            with open(tmp, 'wb') as f:
                np.save(f, a, allow_pickle=False)
        os.replace(tmp, path)
    used.add(name)
    return {'__nd__': name, 'shape': list(a.shape), 'dtype': a.dtype.str}


def _load_array(ref, assets_dir):
    import numpy as np
    path = os.path.join(assets_dir, os.path.basename(str(ref['__nd__'])))
    if path.endswith('.png'):
        from PIL import Image
        with Image.open(path) as im:
            a = np.array(im)
    else:
        a = np.load(path, allow_pickle=False)
    shape = tuple(ref.get('shape') or a.shape)
    if a.shape != shape:
        a = a.reshape(shape)  # (H, W, 1) ecrit en niveaux de gris
    return a.astype(np.dtype(ref.get('dtype') or a.dtype.str), copy=False)


def encode_value(v, assets_dir, used):
    """Valeur de ctrl -> JSON. Les conteneurs sont marques pour revenir a l'identique
    (tuple != list, dict jamais confondu avec un marqueur)."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, tuple):
        return {'__t__': [encode_value(x, assets_dir, used) for x in v]}
    if isinstance(v, list):
        return [encode_value(x, assets_dir, used) for x in v]
    if isinstance(v, dict):
        return {'__d__': {str(k): encode_value(x, assets_dir, used) for k, x in v.items()}}
    mod = type(v).__module__ or ''
    if mod == 'numpy':
        import numpy as np
        if isinstance(v, np.ndarray):
            return _save_array(v, assets_dir, used)
        if isinstance(v, np.generic):
            return v.item()
    if mod.startswith('PIL.'):
        import numpy as np
        return {'__pil__': _save_array(np.asarray(v), assets_dir, used)}
    raise UnserializableJob(f'unserializable type: {type(v).__name__}')


def decode_value(v, assets_dir):
    if isinstance(v, list):
        return [decode_value(x, assets_dir) for x in v]
    if isinstance(v, dict):
        if '__t__' in v:
            return tuple(decode_value(x, assets_dir) for x in v['__t__'])
        if '__d__' in v:
            return {k: decode_value(x, assets_dir) for k, x in v['__d__'].items()}
        if '__nd__' in v:
            return _load_array(v, assets_dir)
        if '__pil__' in v:
            from PIL import Image
            return Image.fromarray(_load_array(v['__pil__'], assets_dir))
        raise ValueError('unknown object in the saved queue')
    return v


# ---------------------------------------------------------------------- file

class JobQueue:
    def __init__(self, max_jobs=50):
        self._jobs = []
        self._lock = threading.Lock()
        self.max_jobs = max_jobs
        self.paused = False
        self.pause_requested = False  # custom-21 : pause douce apres le job en cours
        self.current_task = None  # AsyncTask en cours quand Run queue tourne
        self.running_job = None   # custom-21 : le Job (encore en tete de file) en cours
        # custom-17 : un seul runner a la fois, sinon deux generateurs se
        # disputent la GPU et le flag global interrupt_processing.
        self._runner_lock = threading.Lock()
        self.runner_active = False
        self.idle_timeout = 60.0  # secondes de file vide avant que le runner sorte
        # custom-21 : persistance (desactivee tant que configure_persistence n'est pas appele)
        self.persist_dir = None
        self.expected_len = None
        self.groups_export = None  # () -> dict JSON des groupes XYZ en cours
        self.groups_import = None  # (dict) -> None
        self.restored = 0
        self._save_lock = threading.Lock()
        self._warned_unserializable = set()

    # -- runner ------------------------------------------------------------
    def try_acquire_runner(self):
        """True si l'appelant devient LE runner. False si un runner tourne deja
        (auquel cas il ramassera les jobs tout seul, rien a faire)."""
        with self._runner_lock:
            if self.runner_active:
                return False
            self.runner_active = True
            return True

    def release_runner(self):
        with self._runner_lock:
            self.runner_active = False

    def __len__(self):
        with self._lock:
            return len(self._jobs)

    # -- mutations (chacune persiste) ---------------------------------------
    def add(self, args, label, meta=None):
        """Ajoute un job. Renvoie sa position (1-based), ou -1 si file pleine."""
        with self._lock:
            if len(self._jobs) >= self.max_jobs:
                return -1
            self._jobs.append(Job(args, label, meta))
            pos = len(self._jobs)
        self.save()
        return pos

    def start_next(self):
        """custom-21 : le job de tete, marque en cours mais LAISSE dans la file.
        Il n'en sort que par finish(job, done=True)."""
        with self._lock:
            job = self._jobs[0] if self._jobs else None
            self.running_job = job
        if job is not None:
            self.restored = 0
        return job

    def finish(self, job, done=True):
        """custom-21 : fin d'execution. done=False (Stop en plein vol) garde le job
        en tete pour le re-executer entier a la reprise. Un job retire par
        l'utilisateur pendant son execution n'est pas re-ajoute."""
        with self._lock:
            if done and job is not None:
                for i, j in enumerate(self._jobs):
                    if j is job:
                        del self._jobs[i]
                        break
            self.running_job = None
        self.save()

    def pop_next(self):
        with self._lock:
            job = self._jobs.pop(0) if self._jobs else None
        if job is not None:
            self.save()
        return job

    def remove(self, index):
        with self._lock:
            if index is None or not (0 <= index < len(self._jobs)):
                return False
            self._jobs.pop(index)
        self.save()
        return True

    def move(self, index, delta):
        with self._lock:
            if index is None or not (0 <= index < len(self._jobs)):
                return index
            new = max(0, min(len(self._jobs) - 1, index + delta))
            self._jobs.insert(new, self._jobs.pop(index))
        self.save()
        return new

    def clear(self):
        with self._lock:
            self._jobs.clear()
        self.save()

    # -- pause douce (custom-21) ----------------------------------------------
    def request_pause(self):
        """Le job en cours se TERMINE, puis la file se suspend. Sans runner actif,
        il n'y a rien a suspendre."""
        if not self.runner_active:
            return False
        self.pause_requested = True
        return True

    def consume_pause_request(self):
        """Appele par le runner apres chaque job : True = s'arreter maintenant."""
        if not self.pause_requested:
            return False
        self.pause_requested = False
        self.paused = True
        return True

    # -- affichage ---------------------------------------------------------
    def labels(self):
        with self._lock:
            return [f'#{i + 1} | {"▶ " if j is self.running_job else ""}{j.label}'
                    for i, j in enumerate(self._jobs)]

    def status_text(self):
        # custom-17 : refletent aussi l'etat du runner (actif / en cours de job).
        n = len(self)
        cur = self.current_task
        if self.restored and not self.runner_active and cur is None:
            return (f'{self.restored} job(s) restored from the previous session, pending '
                    '— click Run queue to resume.')
        running = ''
        if cur is not None:
            running = ' Job running.'
        elif self.runner_active:
            running = ' Runner waiting for jobs.'
        if self.pause_requested:
            running += ' Pause requested: the queue will stop after the current job.'
        if n == 0:
            return ('Queue empty.' + running).strip()
        if self.paused:
            return f'{n} job(s) pending, queue paused — click Run queue to resume.'
        return f'{n} job(s) pending, queue processing.{running}'.strip()

    # -- persistance (custom-21) ----------------------------------------------
    def configure_persistence(self, directory, expected_len=None, groups_export=None,
                              groups_import=None):
        """directory None = pas de persistance. expected_len = nombre de ctrls d'un
        snapshot pour l'UI courante (len(ctrls) - 1) : garde-fou au rechargement."""
        self.persist_dir = directory or None
        self.expected_len = expected_len
        self.groups_export = groups_export
        self.groups_import = groups_import

    def _paths(self):
        return (os.path.join(self.persist_dir, QUEUE_FILE),
                os.path.join(self.persist_dir, ASSETS_DIR))

    def save(self):
        """Ecrit la file (atomique). Ne leve jamais : une file non sauvee ne doit pas
        casser l'UI. Un job non serialisable reste en memoire, averti une fois."""
        if not self.persist_dir:
            return False
        with self._lock:
            jobs = list(self._jobs)
        with self._save_lock:
            try:
                qfile, assets = self._paths()
                os.makedirs(self.persist_dir, exist_ok=True)
                used, out = set(), []
                for j in jobs:
                    try:
                        args = [encode_value(a, assets, used) for a in j.args]
                    except UnserializableJob as e:
                        if j.id not in self._warned_unserializable:
                            self._warned_unserializable.add(j.id)
                            print(f'[JobQueue] WARNING job not saved to disk ({j.label}): {e}')
                        continue
                    out.append({'id': j.id, 'label': j.label, 'added_at': j.added_at,
                                'meta': j.meta, 'args': args})
                groups = {}
                if self.groups_export is not None:
                    try:
                        groups = self.groups_export() or {}
                    except Exception as e:
                        print(f'[JobQueue] WARNING XYZ groups not saved: {e}')
                data = {'version': PERSIST_VERSION, 'saved_at': time.time(),
                        'expected_len': self.expected_len, 'jobs': out, 'xyz_groups': groups}
                tmp = qfile + '.tmp'
                with open(tmp, 'w', encoding='utf-8') as f:
                    json.dump(data, f)
                os.replace(tmp, qfile)
                if os.path.isdir(assets):
                    for name in os.listdir(assets):
                        if name not in used and not name.endswith('.tmp'):
                            try:
                                os.remove(os.path.join(assets, name))
                            except OSError:
                                pass
                return True
            except Exception as e:
                print(f'[JobQueue] WARNING queue not saved: {e}')
                return False

    def _backup(self, suffix, payload=None):
        """Copie de sauvegarde datee a cote du fichier (jamais d'effacement silencieux)."""
        qfile, _ = self._paths()
        dst = f'{qfile}.{suffix}-{time.strftime("%Y%m%d-%H%M%S")}'
        try:
            if payload is None:
                os.replace(qfile, dst)
            else:
                with open(dst, 'w', encoding='utf-8') as f:
                    json.dump(payload, f)
            return dst
        except OSError:
            return None

    def load(self):
        """Recharge la file du disque (au boot). Renvoie le nombre de jobs restaures.
        Ne leve jamais."""
        if not self.persist_dir:
            return 0
        qfile, assets = self._paths()
        if not os.path.isfile(qfile):
            return 0
        try:
            with open(qfile, encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict) or data.get('version') != PERSIST_VERSION:
                raise ValueError(f'version {data.get("version") if isinstance(data, dict) else "?"}')
        except Exception as e:
            dst = self._backup('bad')
            print(f'[JobQueue] WARNING saved queue unreadable ({e}), set aside: {dst}')
            return 0
        loaded, rejected = [], []
        for raw in data.get('jobs') or []:
            try:
                args = raw['args']
                if self.expected_len is not None and len(args) != self.expected_len:
                    raise ValueError(f'{len(args)} ctrls instead of {self.expected_len} '
                                     '(Fooocus has changed since)')
                decoded = [decode_value(a, assets) for a in args]
                loaded.append(Job(decoded, raw.get('label', 'job'), raw.get('meta'),
                                  raw.get('added_at'), raw.get('id')))
            except Exception as e:
                rejected.append({'reason': str(e), 'job': raw})
        extra = loaded[self.max_jobs:]
        loaded = loaded[:self.max_jobs]
        for j in extra:
            rejected.append({'reason': f'queue full ({self.max_jobs} jobs max)',
                             'job': {'label': j.label, 'id': j.id}})
        if rejected:
            dst = self._backup('rejected', {'rejected': rejected})
            print(f'[JobQueue] WARNING {len(rejected)} job(s) not restored, details: {dst}')
            for r in rejected[:5]:
                label = (r.get('job') or {}).get('label', '?')
                print(f'[JobQueue]   - {label}: {r["reason"]}')
        if self.groups_import is not None and data.get('xyz_groups'):
            try:
                self.groups_import(data['xyz_groups'])
            except Exception as e:
                print(f'[JobQueue] WARNING XYZ groups not restored: {e}')
        with self._lock:
            self._jobs.extend(loaded)
        self.restored = len(loaded)
        if loaded:
            print(f'[JobQueue] {len(loaded)} job(s) restored from the previous session '
                  '(queue pending: Run queue to resume).')
        if rejected:
            self.save()
        return len(loaded)


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
        prompt = ' '.join(str(args[1]).split()) or '(empty prompt)'
        if len(prompt) > 60:
            prompt = prompt[:57] + '...'
        base = str(args[12]).rsplit('.', 1)[0]
        return f'{prompt} | {base} | {args[4]} | seed {args[8]} | x{args[6]}'
    except Exception:
        return 'job'
