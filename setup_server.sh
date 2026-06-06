#!/bin/bash
# Setup script for Dental API on AWS EC2 (g4dn.xlarge)
# Run this ONCE after first SSH connection

set -e

echo "============================================"
echo "Setting up Dental API Server"
echo "============================================"

# Activate the PyTorch environment from Deep Learning AMI
source /opt/conda/bin/activate pytorch

# Install additional dependencies
echo "Installing Python dependencies..."
pip install -q transformers>=4.51.0 accelerate>=0.28.0 bitsandbytes>=0.46.1 \
    peft>=0.18.0 fastapi>=0.110.0 uvicorn>=0.27.0 qwen-vl-utils Pillow>=10.0.0

# Set HuggingFace token (replace with your actual token!)
echo ""
echo "============================================"
echo "IMPORTANT: Set your HuggingFace token!"
echo "Run: export HF_TOKEN='your_token_here'"
echo "Then: huggingface-cli login --token \$HF_TOKEN"
echo "============================================"
echo ""

# Create a convenient start script
cat > ~/start_api.sh << 'EOF'
#!/bin/bash
source /opt/conda/bin/activate pytorch
cd ~
python dental_api_aws.py
EOF
chmod +x ~/start_api.sh

echo "============================================"
echo "Setup complete!"
echo ""
echo "To start the API:"
echo "  1. Upload dental_api_aws.py to ~/"
echo "  2. Run: ~/start_api.sh"
echo ""
echo "API will be available at:"
echo "  http://<PUBLIC_IP>:8000"
echo "  Swagger docs: http://<PUBLIC_IP>:8000/docs"
echo "============================================"
