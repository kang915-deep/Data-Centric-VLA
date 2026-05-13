#!/bin/bash

# Exit on error
set -e

echo "🚀 Setting up Data-Centric VLA environment..."

# 1. Update and install basic dependencies
sudo apt-get update && sudo apt-get install -y git git-lfs

# 2. Install Python dependencies
echo "📦 Installing Python packages..."
pip install -r requirements.txt

# 3. Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/raw data/pruned checkpoints logs

# 4. Generate mock data for testing
echo "🧪 Generating mock data..."
python scripts/generate_mock_data.py --output_dir data/raw --num_demos 5

echo "✅ Setup complete! You can now run the pruning or training scripts."
echo "Example: python scripts/prune_dataset.py --input_dir data/raw --output_dir data/pruned --skip_semantic"
