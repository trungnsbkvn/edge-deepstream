#!/usr/bin/env python3
"""
Debug script to examine face alignment issues.
Compares original vs aligned faces and visualizes landmark positions.
"""
import cv2
import numpy as np
import os

def visualize_landmarks(img, landmarks, color=(0, 255, 0), radius=3):
    """Draw landmarks on image"""
    img_vis = img.copy()
    if len(landmarks.shape) == 2:
        for i, (x, y) in enumerate(landmarks):
            cv2.circle(img_vis, (int(x), int(y)), radius, color, -1)
            cv2.putText(img_vis, str(i), (int(x)+5, int(y)+5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    return img_vis

def check_alignment_quality():
    """Analyze alignment quality by comparing multiple samples"""
    aligned_dir = "/home/m2n/edge-deepstream/data/faces/aligned"
    
    # Standard face landmarks for reference (from align_functions.cpp)
    standard_face = np.array([
        [38.2946, 51.6963],   # Left eye
        [73.5318, 51.5014],   # Right eye  
        [56.0252, 71.7366],   # Nose tip
        [41.5493, 92.3655],   # Left mouth corner
        [70.7299, 92.2041]    # Right mouth corner
    ])
    
    # Get aligned files
    files = os.listdir(aligned_dir)
    aligned_files = [f for f in files if f.endswith('-aligned.png')]
    
    print(f"Analyzing {len(aligned_files)} aligned face images...")
    
    # Process several samples
    for i, sample_file in enumerate(aligned_files[:5]):
        origin_file = sample_file.replace('-aligned.png', '-origin.png')
        
        aligned_path = os.path.join(aligned_dir, sample_file)
        origin_path = os.path.join(aligned_dir, origin_file)
        
        if not os.path.exists(origin_path):
            continue
            
        # Load images
        aligned_img = cv2.imread(aligned_path)
        origin_img = cv2.imread(origin_path)
        
        if aligned_img is None or origin_img is None:
            continue
            
        print(f"\n--- Sample {i+1}: {sample_file} ---")
        print(f"Origin shape: {origin_img.shape}")
        print(f"Aligned shape: {aligned_img.shape}")
        
        # Visualize standard landmarks on aligned image
        aligned_vis = visualize_landmarks(aligned_img, standard_face, (0, 255, 0))
        
        # Create comparison
        comparison = np.hstack([origin_img, aligned_vis])
        
        # Add text labels
        cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison, "Aligned + Standard", (origin_img.shape[1] + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        output_path = f'/home/m2n/edge-deepstream/debug_alignment_{i+1}.png'
        cv2.imwrite(output_path, comparison)
        print(f"Saved debug image: {output_path}")
        
        # Simple quality check - compute face center and eye distance
        if aligned_img.shape[0] == 112 and aligned_img.shape[1] == 112:
            # Calculate expected eye distance for 112x112 aligned face
            eye_distance = np.linalg.norm(standard_face[1] - standard_face[0])  # Right eye - Left eye
            expected_eye_dist = eye_distance
            print(f"Expected eye distance in aligned face: {expected_eye_dist:.2f} pixels")
            
            # Check if face appears centered and proper size by looking at image statistics
            gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
            mean_intensity = np.mean(gray)
            print(f"Mean intensity: {mean_intensity:.2f} (higher is brighter)")
            
            # Very basic check: proper aligned faces should have good contrast in eye/mouth regions
            eye_region = gray[45:65, 35:80]  # Approximate eye region
            mouth_region = gray[85:105, 35:80]  # Approximate mouth region
            
            eye_std = np.std(eye_region)
            mouth_std = np.std(mouth_region)
            print(f"Eye region std: {eye_std:.2f}, Mouth region std: {mouth_std:.2f}")
            print(f"Quality indicators - Good alignment should have reasonable contrast in facial features")

if __name__ == "__main__":
    check_alignment_quality()