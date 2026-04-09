#!/bin/bash

REGISTRY_URL="postech-a.kr-central-2.kcr.dev"
REGISTRY="${REGISTRY_URL}/chunghyun"
IMAGE_NAME="rist"
VERSION="${1:-latest}"

# Load credentials
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env.registry" ]; then
    source "$SCRIPT_DIR/.env.registry"
else
    echo "Error: $SCRIPT_DIR/.env.registry not found"
    exit 1
fi

TAG="${REGISTRY}/${IMAGE_NAME}:${VERSION}-b200"
echo "Building: ${TAG}"

# Build from project root
cd "$SCRIPT_DIR/.."
docker build -t "$TAG" -f docker/Dockerfile.b200 .

# Push if --push flag
if [[ "$2" == "--push" || "$2" == "-p" ]]; then
    echo "$REGISTRY_PASSWORD" | docker login "$REGISTRY_URL" -u "$REGISTRY_USERNAME" --password-stdin
    docker push "$TAG"
fi
