
BASE_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/"
swint_ogc_url="${BASE_URL}v0.1.0-alpha/groundingdino_swint_ogc.pth"

# Download each of the four checkpoints using wget
echo "Downloading groundingdino_swint_ogc.pth checkpoint..."
wget $swint_ogc_url || { echo "Failed to download checkpoint from $swint_ogc_url"; exit 1; }

echo "checkpoint downloaded successfully."
