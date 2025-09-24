#!/usr/bin/env python3

import cv2
import numpy as np
import os

def compare_alignment():
    """Compare aligned vs original images to verify tilt is fixed"""
    aligned_dir = "/home/m2n/edge-deepstream/data/faces/aligned"
    
    # Get a sample aligned/original pair
    aligned_files = [f for f in os.listdir(aligned_dir) if f.endswith('-aligned.png')]
    if not aligned_files:
        print("No aligned images found!")
        return
    
    # Take the first aligned image
    aligned_file = aligned_files[0]
    original_file = aligned_file.replace('-aligned.png', '-origin.png')
    
    aligned_path = os.path.join(aligned_dir, aligned_file)
    original_path = os.path.join(aligned_dir, original_file)
    
    if not os.path.exists(original_path):
        print(f"Original file not found: {original_path}")
        return
        
    # Load images
    aligned_img = cv2.imread(aligned_path)
    original_img = cv2.imread(original_path)
    
    if aligned_img is None or original_img is None:
        print("Failed to load images")
        return
    
    print(f"Comparing: {aligned_file} vs {original_file}")
    print(f"Aligned shape: {aligned_img.shape}")
    print(f"Original shape: {original_img.shape}")
    
    # Create a side-by-side comparison
    comparison = np.hstack([original_img, aligned_img])
    
    # Save comparison image
    comparison_path = "/home/m2n/edge-deepstream/alignment_comparison_fixed.png"
    cv2.imwrite(comparison_path, comparison)
    print(f"Comparison saved to: {comparison_path}")
    
    # Calculate basic statistics to see if alignment looks better
    def get_face_orientation_score(img):
        """Simple metric to assess face alignment quality"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Use gradient to assess orientation
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate dominant gradient direction
        angles = np.arctan2(grad_y, grad_x)
        hist, bins = np.histogram(angles, bins=36, range=(-np.pi, np.pi))
        dominant_angle = bins[np.argmax(hist)]
        
        return abs(dominant_angle)  # Smaller = more upright
    
    original_score = get_face_orientation_score(original_img)
    aligned_score = get_face_orientation_score(aligned_img)
    
    print(f"Original orientation score: {original_score:.3f}")
    print(f"Aligned orientation score: {aligned_score:.3f}")
    print(f"Improvement: {((original_score - aligned_score) / original_score * 100):.1f}%")

if __name__ == "__main__":
    compare_alignment()