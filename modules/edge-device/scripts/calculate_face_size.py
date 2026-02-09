#!/usr/bin/env python3
"""
Calculate expected face bounding box size at various distances.

This helps determine if detection will work at 15 feet distance.
"""

import math

def calculate_face_size_pixels(
    distance_feet: float,
    camera_resolution_width: int,
    camera_fov_horizontal_deg: float = 90.0,
    face_width_inches: float = 6.0
) -> dict:
    """
    Calculate expected face bounding box size in pixels.
    
    Args:
        distance_feet: Distance from camera to face in feet
        camera_resolution_width: Camera horizontal resolution in pixels
        camera_fov_horizontal_deg: Camera horizontal field of view in degrees (default: 90°)
        face_width_inches: Typical face width in inches (default: 6")
    
    Returns:
        Dict with calculated values
    """
    # Convert to consistent units
    distance_inches = distance_feet * 12
    
    # Calculate horizontal field of view at the given distance
    # FOV_width = 2 * distance * tan(FOV_angle / 2)
    fov_angle_rad = math.radians(camera_fov_horizontal_deg / 2)
    fov_width_inches = 2 * distance_inches * math.tan(fov_angle_rad)
    
    # Calculate what percentage of the FOV the face occupies
    face_percentage = face_width_inches / fov_width_inches
    
    # Calculate face width in pixels
    face_width_pixels = camera_resolution_width * face_percentage
    
    # Face height is typically 1.2-1.3x the width
    face_height_pixels = face_width_pixels * 1.25
    
    return {
        "distance_feet": distance_feet,
        "distance_inches": distance_inches,
        "camera_resolution_width": camera_resolution_width,
        "camera_fov_horizontal_deg": camera_fov_horizontal_deg,
        "fov_width_inches": fov_width_inches,
        "face_width_inches": face_width_inches,
        "face_percentage_of_fov": face_percentage * 100,
        "face_width_pixels": face_width_pixels,
        "face_height_pixels": face_height_pixels,
        "face_bbox_size": f"{int(face_width_pixels)}x{int(face_height_pixels)}"
    }


def main():
    print("=" * 70)
    print("Face Bounding Box Size Calculator")
    print("=" * 70)
    print()
    
    # Camera specs (from config)
    camera_resolution = (2560, 1920)  # Width x Height
    
    # Common camera FOV scenarios
    fov_scenarios = [
        (60, "Narrow FOV (telephoto)"),
        (75, "Medium FOV"),
        (90, "Wide FOV (typical security camera)"),
        (110, "Very Wide FOV (fisheye)"),
    ]
    
    # Distances to test
    distances = [5, 10, 15, 20, 25]
    
    print(f"Camera Resolution: {camera_resolution[0]}x{camera_resolution[1]}")
    print(f"Target Distance: 15 feet")
    print()
    
    for fov_deg, fov_name in fov_scenarios:
        print(f"\n{'='*70}")
        print(f"Camera FOV: {fov_deg}° ({fov_name})")
        print(f"{'='*70}")
        print(f"{'Distance':<10} {'Face Width':<12} {'Face Height':<12} {'BBox Size':<15} {'% of Width':<12}")
        print("-" * 70)
        
        for dist in distances:
            result = calculate_face_size_pixels(
                distance_feet=dist,
                camera_resolution_width=camera_resolution[0],
                camera_fov_horizontal_deg=fov_deg
            )
            
            print(f"{dist:>3} ft     "
                  f"{result['face_width_pixels']:>6.1f} px   "
                  f"{result['face_height_pixels']:>6.1f} px   "
                  f"{result['face_bbox_size']:<15} "
                  f"{result['face_percentage_of_fov']:>5.2f}%")
        
        # Highlight 15 feet
        result_15ft = calculate_face_size_pixels(
            distance_feet=15,
            camera_resolution_width=camera_resolution[0],
            camera_fov_horizontal_deg=fov_deg
        )
        print()
        print(f"  At 15 feet: {result_15ft['face_bbox_size']} pixels")
        print(f"  This is {result_15ft['face_percentage_of_fov']:.2f}% of the horizontal FOV")
    
    print("\n" + "=" * 70)
    print("Detection Thresholds:")
    print("=" * 70)
    print(f"Current min_face_size: 40 pixels")
    print(f"Minimum crop size for recognition: 80 pixels")
    print()
    
    # Check if 15ft faces meet thresholds for different FOVs
    print("At 15 feet, detection will work if:")
    for fov_deg, fov_name in fov_scenarios:
        result = calculate_face_size_pixels(
            distance_feet=15,
            camera_resolution_width=camera_resolution[0],
            camera_fov_horizontal_deg=fov_deg
        )
        face_size = min(result['face_width_pixels'], result['face_height_pixels'])
        detection_ok = face_size >= 40
        recognition_ok = face_size >= 80
        
        status_detect = "✓" if detection_ok else "✗"
        status_recog = "✓" if recognition_ok else "✗"
        
        print(f"  {fov_name:30} ({fov_deg}°): "
              f"Detection {status_detect} ({face_size:.1f}px), "
              f"Recognition {status_recog}")
    
    print("\n" + "=" * 70)
    print("Detection Resolution Impact:")
    print("=" * 70)
    print("If original frame is 2560px and detection_width=1280px:")
    print("  - Frame is downscaled by factor of 0.5 for detection")
    print("  - Face sizes in detection frame are HALVED")
    print()
    
    for fov_deg, fov_name in fov_scenarios:
        result_orig = calculate_face_size_pixels(
            distance_feet=15,
            camera_resolution_width=2560,  # Original resolution
            camera_fov_horizontal_deg=fov_deg
        )
        result_detect = calculate_face_size_pixels(
            distance_feet=15,
            camera_resolution_width=1280,  # Detection resolution
            camera_fov_horizontal_deg=fov_deg
        )
        
        face_size_orig = min(result_orig['face_width_pixels'], result_orig['face_height_pixels'])
        face_size_detect = min(result_detect['face_width_pixels'], result_detect['face_height_pixels'])
        
        detection_ok = face_size_detect >= 40
        
        print(f"  {fov_name:30} ({fov_deg}°):")
        print(f"    Original (2560px): {face_size_orig:.1f}px")
        print(f"    Detection (1280px): {face_size_detect:.1f}px {'✓' if detection_ok else '✗ (too small!)'}")
    
    print("\n" + "=" * 70)
    print("Recommendations:")
    print("=" * 70)
    print("1. For 15ft detection with 90° FOV:")
    print("   - Face in detection frame: ~21px (BELOW 40px threshold!)")
    print("   - SOLUTION: Increase detection_width to 1920px or higher")
    print("   - OR: Lower min_face_size to 20-25px")
    print()
    print("2. Recognition uses full-resolution crops (2560px), so quality is preserved")
    print("3. For reliable 15ft detection:")
    print("   - Use detection_width >= 1920px, OR")
    print("   - Lower min_face_size to 25px, OR")
    print("   - Use narrower FOV camera (< 75°)")


if __name__ == "__main__":
    main()
