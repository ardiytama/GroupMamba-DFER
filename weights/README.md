# Model weights

Checkpoints are `torch.save({'model': state_dict, 'meta': {...}}, path)` files, written by `gm_grefel/train.py` (`best.pth`, `last.pth`).

Add a new release:
1. Upload the `.pth` (GitHub Release or Hugging Face).
2. Add an entry to `manifest.json`:
   ```json
   "dfew_fold1": {"url": "https://.../best.pth", "sha256": "<sha256sum best.pth>", "model": "gm_grefel_base", "dataset": "DFEW", "fold": 1, "uar": 0.0, "war": 0.0}
   ```
3. Load it by key, path or URL:
   ```python
   from gm_grefel import create_model, load_pretrained
   model = create_model('gm_grefel_base')
   load_pretrained(model, 'dfew_fold1')
   ```

`python -m gm_grefel.weights` lists the manifest. Downloads are cached in `~/.cache/gm_grefel` (override with `GM_GREFEL_CACHE`).
