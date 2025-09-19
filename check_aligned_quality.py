#!/usr/bin/env python3
"""
Check aligned face images for proper landmark positioning and visual quality
"""

import cv2
import numpy as np
import os
from pathlib import Path

def check_aligned_image_quality():
    aligned_dir = Path("data/faces/aligned")
    
    if not aligned_dir.exists():
        print("❌ Aligned images directory not found")
        return
    
    aligned_files = list(aligned_dir.glob("*-aligned.png"))
    origin_files = list(aligned_dir.glob("*-origin.png"))
    
    if not aligned_files:
        print("❌ No aligned PNG files found")
        return
    
    print(f"Found {len(aligned_files)} aligned images and {len(origin_files)} origin images")
    
    for aligned_file in aligned_files[:3]:  # Check first 3
        print(f"\n🔍 Analyzing: {aligned_file.name}")
        
        try:
            # Load aligned image
            aligned_img = cv2.imread(str(aligned_file))
            if aligned_img is None:
                print(f"❌ Cannot load {aligned_file.name}")
                continue
                
            h, w = aligned_img.shape[:2]
            print(f"   📐 Dimensions: {w}x{h}")
            
            # Check if image is mostly black/empty
            gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
            non_zero_pixels = cv2.countNonZero(gray)
            total_pixels = w * h
            fill_ratio = non_zero_pixels / total_pixels
            print(f"   🎨 Fill ratio: {fill_ratio:.3f} ({non_zero_pixels}/{total_pixels} non-black pixels)")
            
            # Check for obvious corruption patterns
            if fill_ratio < 0.1:
                print(f"   ⚠️  Image appears mostly black/empty")
            elif fill_ratio > 0.9:
                print(f"   ⚠️  Image appears oversaturated")
            else:
                print(f"   ✅ Fill ratio looks reasonable")
            
            # Check pixel value distribution
            mean_val = np.mean(aligned_img)
            std_val = np.std(aligned_img)
            print(f"   📊 Pixel stats: mean={mean_val:.1f}, std={std_val:.1f}")
            
            # Look for landmark dots (red circles from visualization)
            hsv = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2HSV)
            # Red color range in HSV
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            
            red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            red_pixels = cv2.countNonZero(red_mask)
            
            print(f"   🔴 Red pixels (landmarks): {red_pixels}")
            
            if red_pixels > 10:
                print(f"   ✅ Likely has landmark markers")
            else:
                print(f"   ⚠️  No obvious landmark markers found")
                
            # Check corresponding origin image
            origin_name = aligned_file.name.replace("-aligned.png", "-origin.png")
            origin_file = aligned_dir / origin_name
            
            if origin_file.exists():
                origin_img = cv2.imread(str(origin_file))
                if origin_img is not None:
                    print(f"   📸 Origin image: {origin_img.shape[1]}x{origin_img.shape[0]}")
                    
                    # Compare with origin
                    if np.array_equal(aligned_img, origin_img):
                        print(f"   ⚠️  Aligned and origin images are identical (alignment may have failed)")
                    else:
                        print(f"   ✅ Aligned image differs from origin (alignment applied)")
        
        except Exception as e:
            print(f"❌ Error analyzing {aligned_file.name}: {e}")

if __name__ == "__main__":
    print("🔍 Checking aligned face image quality and landmark positioning...")
    print("=" * 70)
    check_aligned_image_quality()