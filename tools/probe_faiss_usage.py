import os, glob
import numpy as np
from utils.parser_cfg import parse_args, safe_load_index

cfg = parse_args('config/config_pipeline.toml')
recog = cfg.get('recognition', {})
print('[probe] backend =', recog.get('backend'))
idx = safe_load_index(recog)
if idx is None:
    print('[probe] faiss_in_use = False (index not loaded)')
else:
    print('[probe] faiss_in_use = True')
    try:
        print('[probe] index_size =', idx.size())
    except Exception:
        print('[probe] index_size = unknown')
    # optional query using most recent saved feature
    feats_dir = cfg['pipeline'].get('save_feature_path','data/features')
    files = sorted(glob.glob(os.path.join(feats_dir, '*.npy')))
    if files:
        v = np.load(files[-1]).astype('float32').reshape(1,-1)
        try:
            names, D = idx.search(v, k=1)
            name = names[0][0] if names and names[0] else None
            score = float(D[0][0]) if D.size else None
            print('[probe] query_top1 =', name, 'score =', score)
        except Exception as e:
            print('[probe] query_top1 failed:', e)
    else:
        print('[probe] no feature files found to query')
