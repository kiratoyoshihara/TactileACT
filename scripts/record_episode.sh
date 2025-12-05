if [ -z "$1" ]; then
  echo "Error: Episode name required."
  echo "Usage: ./record_episode.sh [episode_name]"
  echo "Example: ./record_episode.sh episode_001"
  exit 1
fi

EPISODE_NAME="$1"
OUTPUT_DIR="../data/rosbags"  

mkdir -p "$OUTPUT_DIR"

echo "=================================================="
echo "📽️  START RECORDING: $EPISODE_NAME"
echo "📂 Saving to: $OUTPUT_DIR/$EPISODE_NAME"
echo "Stop with [Ctrl + C]"
echo "=================================================="

ros2 bag record -o "${OUTPUT_DIR}/${EPISODE_NAME}" \
    /camera/image_raw \
    /gelsight/image_raw \
    /lbr_state \
    /hande/joint_states \
    /hande/command \
    /lbr_command