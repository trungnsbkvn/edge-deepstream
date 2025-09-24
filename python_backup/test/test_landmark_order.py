#!/usr/bin/env python3
"""
Test face alignment with synthetic landmarks to identify the issue.
"""

import cv2
import numpy as np
import os

def test_alignment_with_synthetic_landmarks():
    """Test alignment by creating synthetic landmarks and seeing the result"""
    
    # Standard face landmarks for 112x112 aligned face (from align_functions.cpp)
    standard_face = np.array([
        [38.2946, 51.6963],   # Left eye (in aligned space)
        [73.5318, 51.5014],   # Right eye  
        [56.0252, 71.7366],   # Nose tip
        [41.5493, 92.3655],   # Left mouth corner
        [70.7299, 92.2041]    # Right mouth corner
    ], dtype=np.float32)
    
    print("Standard face landmarks (target for alignment):")
    for i, (x, y) in enumerate(standard_face):
        print(f"  Point {i}: ({x:.2f}, {y:.2f})")
    
    # Now let's analyze what happens when we have slightly rotated/scaled face
    # This simulates what should come from the detection model
    
    # Create a test face with some rotation and scaling
    angle = 15 * np.pi / 180  # 15 degrees rotation
    scale = 0.9
    
    # Rotation matrix
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]]) * scale
    
    # Translate to different center
    center_offset = np.array([10, 5])
    
    # Apply transformation to create "detected" landmarks
    detected_landmarks = standard_face @ R.T + center_offset + standard_face.mean(axis=0) * (1-scale)
    
    print(f"\nDetected landmarks (simulated, rotated {angle*180/np.pi:.1f}°, scaled {scale:.2f}):")
    for i, (x, y) in enumerate(detected_landmarks):
        print(f"  Point {i}: ({x:.2f}, {y:.2f})")
    
    # Now test the landmark order that YOLOv8 face typically uses
    # According to most YOLOv8 face models, the order is:
    # [left_eye, right_eye, nose, left_mouth, right_mouth]
    # This should match our standard_face order
    
    yolo_order = [0, 1, 2, 3, 4]  # Direct mapping if order is correct
    
    # However, some models use different orders. Let's test common variations:
    possible_orders = [
        [0, 1, 2, 3, 4],  # Standard: LE, RE, N, LM, RM
        [1, 0, 2, 4, 3],  # Swapped eyes and mouth corners
        [0, 1, 4, 2, 3],  # Different nose/mouth order
        [1, 0, 2, 3, 4],  # Just swapped eyes
    ]
    
    print("\nTesting different landmark orderings:")
    for order_idx, order in enumerate(possible_orders):
        reordered_landmarks = detected_landmarks[order]
        print(f"\nOrder {order_idx}: {order}")
        
        # Compute similarity transform (simplified version)
        # This is what SimilarTransform() does internally
        src_mean = np.mean(reordered_landmarks, axis=0)
        dst_mean = np.mean(standard_face, axis=0)
        
        src_centered = reordered_landmarks - src_mean
        dst_centered = standard_face - dst_mean
        
        # Cross-covariance matrix
        H = dst_centered.T @ src_centered / len(standard_face)
        
        # SVD
        U, S, Vt = np.linalg.svd(H)
        
        # Rotation matrix
        R_est = U @ Vt
        
        # Handle reflection
        if np.linalg.det(R_est) < 0:
            Vt[-1, :] *= -1
            R_est = U @ Vt
        
        # Scale
        src_var = np.var(src_centered, axis=0).sum()
        scale_est = np.sum(S) / src_var
        
        # Translation
        t_est = dst_mean - scale_est * R_est @ src_mean
        
        # Create transformation matrix
        M = np.eye(3)
        M[:2, :2] = scale_est * R_est
        M[:2, 2] = t_est
        
        print(f"  Estimated scale: {scale_est:.3f} (expected: {1/scale:.3f})")
        print(f"  Estimated rotation: {np.arctan2(R_est[1,0], R_est[0,0])*180/np.pi:.1f}° (expected: {-angle*180/np.pi:.1f}°)")
        
        # Apply transformation to see result
        ones = np.ones((len(reordered_landmarks), 1))
        landmarks_hom = np.hstack([reordered_landmarks, ones])
        transformed = (M @ landmarks_hom.T).T[:, :2]
        
        # Compute alignment error
        error = np.linalg.norm(transformed - standard_face, axis=1)
        avg_error = np.mean(error)
        max_error = np.max(error)
        
        print(f"  Average alignment error: {avg_error:.2f} pixels")
        print(f"  Maximum alignment error: {max_error:.2f} pixels")
        
        if avg_error < 2.0:  # Good alignment threshold
            print(f"  ✓ This order produces good alignment!")
        else:
            print(f"  ✗ Poor alignment with this order")

if __name__ == "__main__":
    test_alignment_with_synthetic_landmarks()