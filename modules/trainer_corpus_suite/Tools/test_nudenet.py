#!/usr/bin/env python3
"""
Test NudeNet Installation and API
"""

try:
    from nudenet import NudeDetector
    print("✓ NudeNet imported successfully")

    # Test detector creation
    detector = NudeDetector()
    print("✓ NudeDetector created successfully")

    # Test available attributes
    print("Available attributes:", [attr for attr in dir(detector) if not attr.startswith('_')])

    print("\n✅ NudeNet is ready to use!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Install with: pip install nudenet")

except Exception as e:
    print(f"❌ Error: {e}")
    print("Check NudeNet installation and API")