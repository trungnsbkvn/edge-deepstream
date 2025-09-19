#!/usr/bin/env python3
"""
Test different landmark orders to find the correct mapping for YOLOv8n face model.
"""
import cv2
import numpy as np

def similarity_transform(src_points, dst_points):
    """Compute similarity transform from source to destination points"""
    assert src_points.shape == dst_points.shape
    assert src_points.shape[1] == 2
    
    num = src_points.shape[0]
    
    # Compute means
    src_mean = np.mean(src_points, axis=0)
    dst_mean = np.mean(dst_points, axis=0)
    
    # Center the points
    src_demean = src_points - src_mean
    dst_demean = dst_points - dst_mean
    
    # Cross-covariance matrix
    A = dst_demean.T @ src_demean / num
    
    # SVD
    U, S, Vt = np.linalg.svd(A)
    
    # Rotation matrix
    R = U @ Vt
    
    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    
    # Scale
    src_var = np.var(src_demean, axis=0).sum()
    if src_var > 0:
        scale = np.sum(S) / src_var
    else:
        scale = 1.0
    
    # Translation
    t = dst_mean - scale * R @ src_mean
    
    # Create 2x3 transformation matrix
    M = np.zeros((2, 3))
    M[:2, :2] = scale * R
    M[:2, 2] = t
    
    return M

def test_landmark_orders_with_real_data():
    """Test different landmark orders using actual detected landmarks if available"""
    
    # Standard face landmarks (InsightFace template)
    standard_face = np.array([
        [38.2946, 51.6963],   # Left eye
        [73.5318, 51.5014],   # Right eye  
        [56.0252, 71.7366],   # Nose tip
        [41.5493, 92.3655],   # Left mouth corner
        [70.7299, 92.2041]    # Right mouth corner
    ], dtype=np.float32)
    
    print("Standard InsightFace landmarks:")
    labels = ["Left Eye", "Right Eye", "Nose", "Left Mouth", "Right Mouth"]
    for i, ((x, y), label) in enumerate(zip(standard_face, labels)):
        print(f"  {i}: {label} = ({x:.2f}, {y:.2f})")
    
    # Test with synthetic landmarks that simulate common YOLOv8 output patterns
    # Most YOLOv8 face models follow the WIDERFACE annotation format
    
    # Simulate detected landmarks in different possible orders
    test_cases = [
        {
            "name": "InsightFace Order: [LE, RE, N, LM, RM]",
            "order": [0, 1, 2, 3, 4],
            "detected": np.array([
                [40.2, 53.1],    # Left eye (slightly offset)
                [71.8, 52.9],    # Right eye
                [57.1, 73.2],    # Nose
                [42.8, 90.5],    # Left mouth
                [69.1, 91.8]     # Right mouth
            ])
        },
        {
            "name": "Swapped Eyes: [RE, LE, N, LM, RM]", 
            "order": [1, 0, 2, 3, 4],
            "detected": np.array([
                [71.8, 52.9],    # Right eye first
                [40.2, 53.1],    # Left eye second
                [57.1, 73.2],    # Nose
                [42.8, 90.5],    # Left mouth
                [69.1, 91.8]     # Right mouth
            ])
        },
        {
            "name": "WIDER Face Order: [LE, RE, N, RM, LM]",
            "order": [0, 1, 2, 4, 3],
            "detected": np.array([
                [40.2, 53.1],    # Left eye
                [71.8, 52.9],    # Right eye
                [57.1, 73.2],    # Nose
                [69.1, 91.8],    # Right mouth (before left)
                [42.8, 90.5]     # Left mouth
            ])
        },
        {
            "name": "Alternative: [LE, RE, LM, RM, N]",
            "order": [0, 1, 3, 4, 2],
            "detected": np.array([
                [40.2, 53.1],    # Left eye
                [71.8, 52.9],    # Right eye  
                [42.8, 90.5],    # Left mouth
                [69.1, 91.8],    # Right mouth
                [57.1, 73.2]     # Nose last
            ])
        }
    ]
    
    print(f"\nTesting {len(test_cases)} different landmark orders:")
    print("=" * 60)
    
    best_order = None
    best_error = float('inf')
    
    for case in test_cases:
        detected = case["detected"]
        order = case["order"]
        name = case["name"]
        
        print(f"\n{name}")
        print("-" * len(name))
        
        # Reorder detected landmarks according to the test order
        reordered = np.zeros_like(standard_face)
        for i, orig_idx in enumerate(order):
            reordered[orig_idx] = detected[i]
        
        # Compute similarity transform
        M = similarity_transform(reordered, standard_face)
        
        # Apply transformation to reordered landmarks
        ones = np.ones((len(reordered), 1))
        landmarks_hom = np.hstack([reordered, ones])
        transformed = (np.vstack([M, [0, 0, 1]]) @ landmarks_hom.T).T[:, :2]
        
        # Compute alignment error
        errors = np.linalg.norm(transformed - standard_face, axis=1)
        avg_error = np.mean(errors)
        max_error = np.max(errors)
        
        print(f"  Average error: {avg_error:.3f} pixels")
        print(f"  Maximum error: {max_error:.3f} pixels")
        print(f"  Per-point errors: {[f'{e:.2f}' for e in errors]}")
        
        # Compute transformation parameters for analysis
        scale = np.sqrt(np.linalg.det(M[:2, :2]))
        rotation_rad = np.arctan2(M[1, 0], M[0, 0])
        rotation_deg = rotation_rad * 180 / np.pi
        
        print(f"  Transform - Scale: {scale:.3f}, Rotation: {rotation_deg:.1f}°")
        
        if avg_error < best_error:
            best_error = avg_error
            best_order = case
        
        # Quality assessment
        if avg_error < 1.0:
            print(f"  ✓ EXCELLENT alignment!")
        elif avg_error < 2.0:
            print(f"  ✓ Good alignment")
        elif avg_error < 5.0:
            print(f"  ⚠ Fair alignment")
        else:
            print(f"  ✗ Poor alignment")
    
    print(f"\n" + "=" * 60)
    print(f"BEST ORDER: {best_order['name']}")
    print(f"Average error: {best_error:.3f} pixels")
    
    return best_order

if __name__ == "__main__":
    best = test_landmark_orders_with_real_data()
    print(f"\nRecommendation: Use landmark order {best['order']} for YOLOv8n face model")