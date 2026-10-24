import hashlib
import json
import os
import urllib.request

MANIFEST = os.path.join(os.path.dirname(__file__), '..', 'weights', 'manifest.json')
CACHE = os.environ.get('GM_GREFEL_CACHE', os.path.expanduser('~/.cache/gm_grefel'))


def _manifest():
    with open(MANIFEST, encoding='utf-8') as f:
        return json.load(f)['models']


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def resolve(source):
    # source: local file, key in weights/manifest.json, or direct URL
    if os.path.isfile(source):
        return source
    entry = _manifest().get(source, {'url': source})
    url = entry['url']
    if not url.startswith(('http://', 'https://')):
        raise FileNotFoundError(f"{source} is not a file, manifest key or URL")
    path = os.path.join(CACHE, os.path.basename(url.split('?')[0]))
    if not os.path.isfile(path):
        os.makedirs(CACHE, exist_ok=True)
        urllib.request.urlretrieve(url, path + '.part')
        os.replace(path + '.part', path)
    if 'sha256' in entry and _sha256(path) != entry['sha256']:
        os.remove(path)
        raise ValueError(f"sha256 mismatch for {source}")
    return path


def load_checkpoint(source):
    # torch < 2 cannot unpickle floats with weights_only, so only load trusted checkpoints there
    import torch
    safe = int(torch.__version__.split('.')[0]) >= 2
    return torch.load(resolve(source), map_location='cpu', weights_only=safe)


def load_pretrained(model, source, strict=True):
    ckpt = load_checkpoint(source)
    state = next((ckpt[k] for k in ('model', 'module', 'state_dict') if k in ckpt), ckpt)
    model.load_state_dict(state, strict=strict)
    return ckpt.get('meta', {}) if isinstance(ckpt, dict) else {}


def save_checkpoint(path, model, state=None, **meta):
    import torch
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({'model': model.state_dict(), 'meta': meta, **(state or {})}, path + '.tmp')
    os.replace(path + '.tmp', path)


if __name__ == '__main__':
    for name, e in _manifest().items():
        print(name, e.get('dataset', ''), f"fold={e.get('fold', '')}", f"UAR={e.get('uar', '')}", f"WAR={e.get('war', '')}")
