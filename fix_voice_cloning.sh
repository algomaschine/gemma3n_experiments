#!/bin/bash
# Fix voice cloning dependencies for Python 3.9

echo "🔧 Fixing voice cloning dependencies..."

# Uninstall problematic packages
echo "📦 Removing incompatible packages..."
pip uninstall -y bangla TTS

# Install compatible versions
echo "📦 Installing compatible bangla package..."
pip install "bangla==0.0.2"

# Try different TTS installation approaches
echo "📦 Attempting TTS installation..."

# Method 1: Try standard installation
if pip install TTS>=0.20.0; then
    echo "✅ TTS installed successfully!"
else
    echo "⚠️  Standard installation failed, trying alternatives..."
    
    # Method 2: Install without dependencies and add them manually
    pip install --no-deps TTS
    pip install torch torchaudio numpy scipy librosa soundfile
    
    # Method 3: Try pre-compiled wheels
    if ! python -c "from TTS.api import TTS" 2>/dev/null; then
        echo "🔄 Trying pre-compiled wheels..."
        pip install --find-links https://download.pytorch.org/whl/torch_stable.html TTS
    fi
fi

# Test installation
echo "🧪 Testing voice cloning..."
if python3 -c "from TTS.api import TTS; print('✅ Voice cloning working!')" 2>/dev/null; then
    echo "🎉 Voice cloning is now available!"
else
    echo "❌ Voice cloning still not working"
    echo "💡 You can still use the translator with standard TTS"
    echo "💡 Run with --no-voice-cloning flag to disable voice cloning"
fi

echo "✅ Setup complete!" 