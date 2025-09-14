#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}"/env.sh

rsync -avz --delete \
    "$SCRIPT_DIR/" "${DEVICE_SSH_NAME}:${DEVICE_WORK_DIR}/${SCRIPT_DIR##*/}" 
if [ $? -ne 0 ]; then
    echo "Error: rsync failed."
    exit 1
fi
echo "Files successfully sent to Banana board. Finished: $(date +%Y-%m-%d_%H-%M-%S)"

exit 0
