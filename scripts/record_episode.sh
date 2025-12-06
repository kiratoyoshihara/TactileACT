if [ -z "$1" ]; then
  echo "Usage: ./record_episode.sh [episode_name]"
  exit 1
fi

EPISODE_NAME="$1"
OUTPUT_DIR="../data/rosbags"
mkdir -p "$OUTPUT_DIR"

echo "=== START RECORDING (ROS 1) ==="

rosbag record -O "${OUTPUT_DIR}/${EPISODE_NAME}.bag" \
    /camera/image_raw \
    /gelsight/image_raw \
    /lbr_state \
    /hande/joint_states \
    /hande/command \
    /lbr_command