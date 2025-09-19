#!/usr/bin/env python3
"""
More detailed analysis of aligned image corruption
"""

import cv2
import numpy as np
from pathlib import Path

def analyze_corruption():
    aligned_dir = Path("data/faces/aligned")
    aligned_files = list(aligned_dir.glob("*-aligned.png"))
    
    if not aligned_files:
        print("No aligned files found")
        return
        
    img_path = aligned_files[0]  # Check first image
    print(f"Analyzing: {img_path.name}")
    
    img = cv2.imread(str(img_path))
    if img is None:
        print("Cannot load image")
        return
        
    print(f"Shape: {img.shape}")
    print(f"Dtype: {img.dtype}")
    
    # Check each channel
    b, g, r = cv2.split(img)
    
    print(f"Blue channel - min: {b.min()}, max: {b.max()}, mean: {b.mean():.1f}")
    print(f"Green channel - min: {g.min()}, max: {g.max()}, mean: {g.mean():.1f}")
    print(f"Red channel - min: {r.min()}, max: {r.max()}, mean: {r.mean():.1f}")
    
    # Check for patterns that indicate corruption
    # Look for stuck values or unusual patterns
    unique_b = len(np.unique(b))
    unique_g = len(np.unique(g))
    unique_r = len(np.unique(r))
    
    print(f"Unique values - B: {unique_b}, G: {unique_g}, R: {unique_r}")
    
    # Check for dominant colors that might indicate corruption
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
    
    max_b_val = np.argmax(hist_b)
    max_g_val = np.argmax(hist_g)
    max_r_val = np.argmax(hist_r)
    
    max_b_count = hist_b[max_b_val, 0]
    max_g_count = hist_g[max_g_val, 0]
    max_r_count = hist_r[max_r_val, 0]
    
    total_pixels = img.shape[0] * img.shape[1]
    
    print(f"Most common values:")
    print(f"  Blue {max_b_val}: {max_b_count/total_pixels:.3f} of pixels")
    print(f"  Green {max_g_val}: {max_g_count/total_pixels:.3f} of pixels") 
    print(f"  Red {max_r_val}: {max_r_count/total_pixels:.3f} of pixels")
    
    # Check specific patterns that indicate the corruption you described
    # "red dot in middle with broken blue/green pixels"
    center_y, center_x = img.shape[0]//2, img.shape[1]//2
    center_region = img[center_y-5:center_y+5, center_x-5:center_x+5]
    
    print(f"Center region (10x10) around ({center_x}, {center_y}):")
    print(f"  Mean BGR: {center_region.mean(axis=(0,1))}")
    
    # Check if center is predominantly red
    center_red = center_region[:,:,2].mean()
    center_blue = center_region[:,:,0].mean()
    center_green = center_region[:,:,1].mean()
    
    if center_red > center_blue + 50 and center_red > center_green + 50:
        print("  ⚠️  Center appears to have red dot pattern")
    
    # Check for scattered noise patterns in blue/green channels
    # High variance in blue/green with low mean could indicate corruption
    blue_var = np.var(b)
    green_var = np.var(g)
    red_var = np.var(r)
    
    print(f"Channel variance - B: {blue_var:.1f}, G: {green_var:.1f}, R: {red_var:.1f}")
    
    # Save a small crop to examine manually
    crop = img[50:62, 50:62]  # 12x12 crop
    cv2.imwrite(f"debug_crop_{img_path.stem}.png", crop)
    print(f"Saved debug crop: debug_crop_{img_path.stem}.png")

if __name__ == "__main__":
    analyze_corruption()