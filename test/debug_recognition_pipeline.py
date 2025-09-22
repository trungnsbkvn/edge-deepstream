import os, json, sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.faiss_index import FaceIndex

LABELS = 'data/index/labels.json'
INDEX = 'data/index/faiss.index'

def main():
    print('--- Recognition Debug ---')
    if os.path.exists(LABELS):
        with open(LABELS,'r',encoding='utf-8') as f:
            meta = json.load(f)
        persons = meta.get('persons', {})
        print(f'Labels file: {LABELS} persons={len(persons)} keys={list(persons.keys())[:5]}')
    else:
        print('Labels file missing:', LABELS)
    # Try load index via helper (to get wrapper object with size())
    if os.path.exists(INDEX) and os.path.exists(LABELS):
        try:
            fi = FaceIndex.load(INDEX, LABELS, use_gpu=False)
            print(f'FAISS index loaded: size={fi.size()} metric={fi.cfg.metric} type={fi.cfg.index_type}')
        except Exception as e:
            print('Index load error:', e)
    else:
        print('Index or labels path missing. Expected:')
        print('  ', INDEX, 'exists=' , os.path.exists(INDEX))
        print('  ', LABELS, 'exists=' , os.path.exists(LABELS))

if __name__ == '__main__':
    main()
