#!/bin/bash
# Build and install script for vector-and-llm package

set -e

echo "🔨 Building vector-and-llm package..."

# Clean up previous builds
echo "🧹 Cleaning up previous builds..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/

# Install build dependencies
echo "📦 Installing build dependencies..."
pip install --upgrade pip setuptools wheel twine

# Build the package
echo "🏗️ Building package..."
python setup.py sdist bdist_wheel

# Check the package
echo "🔍 Checking package..."
twine check dist/*

echo "✅ Package built successfully!"
echo "📁 Distribution files:"
ls -la dist/

echo ""
echo "🚀 To install locally:"
echo "pip install dist/vector_and_llm-1.0.0-py3-none-any.whl"
echo ""
echo "🚀 To install in development mode:"
echo "pip install -e ."
echo ""
echo "📤 To publish to PyPI (when ready):"
echo "twine upload dist/*"