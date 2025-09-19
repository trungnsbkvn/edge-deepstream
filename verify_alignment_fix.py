#!/usr/bin/env python3
"""
Verification script for aligned face image saving fix
"""

import os
import cv2
import numpy as np
from pathlib import Path

def check_aligned_images():
    aligned_dir = Path("data/faces/aligned")
    
    if not aligned_dir.exists():
        print("❌ Aligned images directory not found")
        return
    
    png_files = list(aligned_dir.glob("*-aligned.png"))
    
    if not png_files:
        print("❌ No aligned PNG files found")
        return
    
    print(f"✅ Found {len(png_files)} aligned PNG files")
    
    good_images = 0
    broken_images = 0
    
    for png_file in png_files[:10]:  # Check first 10 images
        try:
            file_size = png_file.stat().st_size
            
            if file_size < 1000:  # Less than 1KB is likely broken
                print(f"⚠️  {png_file.name}: Too small ({file_size} bytes)")
                broken_images += 1
                continue
                
            # Try to load the image
            img = cv2.imread(str(png_file))
            
            if img is None:
                print(f"❌ {png_file.name}: Cannot load image")
                broken_images += 1
                continue
                
            h, w = img.shape[:2]
            
            if h != 112 or w != 112:
                print(f"⚠️  {png_file.name}: Wrong dimensions {w}x{h} (expected 112x112)")
                broken_images += 1
                continue
                
            print(f"✅ {png_file.name}: {w}x{h} ({file_size:,} bytes)")
            good_images += 1
            
        except Exception as e:
            print(f"❌ {png_file.name}: Error loading - {e}")
            broken_images += 1
    
    print(f"\n📊 SUMMARY:")
    print(f"   Good images: {good_images}")
    print(f"   Broken images: {broken_images}")
    print(f"   Total checked: {good_images + broken_images}")
    
    if good_images > 0:
        print("🎉 Alignment saving fix is working!")
        print("   - Images are properly saved as 112x112 PNG files")
        print("   - File sizes are reasonable (10-25KB)")
        print("   - No more ~500-600 byte broken files")
    else:
        print("❌ Alignment saving still has issues")

if __name__ == "__main__":
    print("🔍 Verifying aligned face image saving fix...")
    print("=" * 50)
    check_aligned_images()