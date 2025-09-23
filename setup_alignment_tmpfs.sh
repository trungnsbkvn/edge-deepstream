#!/bin/bash

# Setup script for tmpfs alignment directory
# This script ensures the alignment directory exists in tmpfs (/dev/shm)

ALIGNMENT_DIR="/dev/shm/edge-deepstream/aligned"
RECOGNITION_DIR="/dev/shm/edge-deepstream/recognized"

echo "Setting up tmpfs directories..."

# Create directory if it doesn't exist
if [ ! -d "$ALIGNMENT_DIR" ]; then
    echo "Creating alignment directory: $ALIGNMENT_DIR"
    mkdir -p "$ALIGNMENT_DIR"
else
    echo "Alignment directory already exists: $ALIGNMENT_DIR"
fi

if [ ! -d "$RECOGNITION_DIR" ]; then
    echo "Creating recognition directory: $RECOGNITION_DIR"
    mkdir -p "$RECOGNITION_DIR"
else
    echo "Recognition directory already exists: $RECOGNITION_DIR"
fi

# Set proper ownership and permissions
echo "Setting ownership and permissions..."
chown -R $(whoami):$(id -gn) /dev/shm/edge-deepstream
chmod -R 755 /dev/shm/edge-deepstream

# Test write permissions
TEST_FILE="$ALIGNMENT_DIR/test_write.tmp"
if touch "$TEST_FILE" 2>/dev/null; then
    rm -f "$TEST_FILE"
    echo "✓ Alignment directory is writable: $ALIGNMENT_DIR"
else
    echo "✗ ERROR: Cannot write to alignment directory: $ALIGNMENT_DIR"
    exit 1
fi

# Show directory info
echo "Alignment & Recognition directories ready:"
ls -la "$ALIGNMENT_DIR"
ls -la "$RECOGNITION_DIR"

echo "Done! The pipeline can now use tmpfs alignment & recognition directories for better performance."