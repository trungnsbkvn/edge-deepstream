import os, cv2, numpy as np, hashlib

aligned_dir = 'data/faces/aligned'
recognized_dir = 'data/faces/recognized'


def image_hash(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def compare_any_pair():
    if not os.path.isdir(aligned_dir) or not os.path.isdir(recognized_dir):
        print('Missing aligned or recognized directory')
        return
    aligned_files = [f for f in os.listdir(aligned_dir) if f.endswith('-aligned.png')]
    recog_files = [f for f in os.listdir(recognized_dir) if f.lower().endswith('.png')]
    if not aligned_files or not recog_files:
        print('Not enough files to compare yet. Run pipeline first.')
        return
    # Load first aligned and first recognized
    a_path = os.path.join(aligned_dir, aligned_files[0])
    r_path = os.path.join(recognized_dir, recog_files[0])
    a_img = cv2.imread(a_path)
    r_img = cv2.imread(r_path)
    if a_img is None or r_img is None:
        print('Could not load images for comparison')
        return
    same_shape = a_img.shape == r_img.shape
    pixel_equal = same_shape and np.array_equal(a_img, r_img)
    # Compute simple difference metric (mean absolute diff after resizing recognized to 112 if needed)
    a_proc = a_img
    r_proc = r_img
    if r_proc.shape[:2] != a_proc.shape[:2]:
        r_proc = cv2.resize(r_proc, (a_proc.shape[1], a_proc.shape[0]))
    mad = float(np.mean(np.abs(a_proc.astype(np.int16) - r_proc.astype(np.int16))))
    print(f'Aligned: {a_path}')
    print(f'Recognized: {r_path}')
    print(f'Same shape: {same_shape}')
    print(f'Pixels identical: {pixel_equal}')
    print(f'Mean abs difference: {mad:.2f}')
    if pixel_equal:
        print('⚠️ Images are still identical. Investigate cropping path.')
    else:
        print('✅ Images differ (expected).')

if __name__ == '__main__':
    compare_any_pair()
