#!/bin/bash
#
# Install OpenCV with GStreamer support on Jetson
# This ensures enrollment and recognition use the same video pipeline
#

set -e

echo "=========================================="
echo "OpenCV GStreamer Installation for Jetson"
echo "=========================================="
echo ""

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "⚠️  Warning: This script is designed for NVIDIA Jetson devices"
    echo "   It may work on other systems but is optimized for Jetson"
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check current OpenCV
echo "Current OpenCV installation:"
python3 -c "import cv2; print('  Version:', cv2.__version__); print('  GStreamer:', 'YES' if 'GStreamer' in cv2.getBuildInformation() and 'YES' in cv2.getBuildInformation() else 'NO')" || echo "  OpenCV not installed"
echo ""

# Option 1: Try to install pre-built OpenCV with GStreamer (fastest)
echo "Attempting to install pre-built OpenCV with GStreamer..."
echo ""

# Uninstall existing OpenCV
echo "Removing existing OpenCV packages..."
pip3 uninstall -y opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null || true

# Install dependencies
echo "Installing dependencies..."
sudo apt-get update
sudo apt-get install -y \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-tools \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    python3-opencv

# Verify installation
echo ""
echo "Verifying OpenCV GStreamer support..."
if python3 -c "import cv2; assert 'GStreamer' in cv2.getBuildInformation(), 'GStreamer not found'; print('✅ SUCCESS: OpenCV with GStreamer installed')" 2>/dev/null; then
    echo ""
    python3 -c "import cv2; print('OpenCV version:', cv2.__version__)"
    echo ""
    echo "=========================================="
    echo "Installation complete!"
    echo "=========================================="
    exit 0
fi

# Option 2: Build from source (slower but guaranteed to work)
echo ""
echo "⚠️  Pre-built package doesn't have GStreamer support"
echo "   Would you like to build OpenCV from source? (takes ~30-60 minutes)"
read -p "Build from source? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation aborted."
    echo ""
    echo "Alternative: Use FFmpeg workaround (see RTSP_FIX_SUMMARY.md)"
    exit 1
fi

echo ""
echo "Building OpenCV from source with GStreamer..."
echo "This will take 30-60 minutes. Get some coffee! ☕"
echo ""

# Install build dependencies
sudo apt-get install -y \
    build-essential cmake git pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev libatlas-base-dev gfortran \
    python3-dev

# Download OpenCV
cd /tmp
if [ ! -d "opencv" ]; then
    git clone --depth 1 --branch 4.8.0 https://github.com/opencv/opencv.git
fi
if [ ! -d "opencv_contrib" ]; then
    git clone --depth 1 --branch 4.8.0 https://github.com/opencv/opencv_contrib.git
fi

# Build OpenCV
cd opencv
mkdir -p build
cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_GSTREAMER=ON \
    -D WITH_GSTREAMER_0_10=OFF \
    -D WITH_FFMPEG=ON \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D ENABLE_FAST_MATH=ON \
    -D CUDA_FAST_MATH=ON \
    -D CUDA_ARCH_BIN="5.3,6.2,7.2,8.7" \
    -D WITH_CUBLAS=ON \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -D BUILD_EXAMPLES=OFF \
    -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D BUILD_opencv_python3=ON \
    ..

# Compile (use all CPU cores)
make -j$(nproc)

# Install
sudo make install
sudo ldconfig

# Create symbolic link for python
PYTHON_SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
sudo ln -sf /usr/local/lib/python3.*/site-packages/cv2 $PYTHON_SITE_PACKAGES/cv2

# Verify
echo ""
echo "Verifying installation..."
python3 -c "import cv2; print('✅ OpenCV version:', cv2.__version__); assert 'GStreamer' in cv2.getBuildInformation(), 'GStreamer support missing'"

echo ""
echo "=========================================="
echo "✅ Build complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Test color pipeline: python3 test_rtsp_color.py"
echo "  2. Re-enroll users: python3 facial_enrollment.py --camera-ip 10.42.0.159"
echo "  3. Test recognition: python3 facial_recognition.py deepstream ..."
echo ""
