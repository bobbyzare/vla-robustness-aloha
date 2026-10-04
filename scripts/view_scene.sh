#!/usr/bin/env bash
# Open an ALOHA simulation scene in MuJoCo's interactive 3D viewer.
#
# Use it to look at the world (robot, table, objects, cameras) and to check
# that the simulator works. Nothing controls the robot here: you move the
# joints yourself with the "Control" sliders in the viewer's right panel.
#
# Usage:
#   conda activate lerobot && unset PYTHONPATH
#   bash scripts/view_scene.sh                 # Transfer Cube (default)
#   bash scripts/view_scene.sh insertion       # Insertion
#
# Needs a display. On WSL2 this is WSLg (Windows 11 shows Linux windows).

set -euo pipefail   # stop on any error, on unset variables, and on failures inside pipes

# 1. Which task? First argument, default "transfer_cube".
TASK="${1:-transfer_cube}"
case "$TASK" in
    transfer_cube|insertion) ;;
    *) echo "Unknown task '$TASK'. Use: transfer_cube | insertion" >&2; exit 1 ;;
esac

# 2. Check we are in the right Python environment.
if ! python -c "import mujoco, gym_aloha" 2>/dev/null; then
    echo "Cannot import mujoco/gym_aloha. Run: conda activate lerobot && unset PYTHONPATH" >&2
    exit 1
fi

# 3. Find where pip installed gym_aloha's scene files (MJCF XML + STL meshes).
ASSETS="$(python -c 'import gym_aloha, os; print(os.path.join(os.path.dirname(gym_aloha.__file__), "assets"))')"
SCENE="$ASSETS/bimanual_viperx_${TASK}.xml"

# 4. The viewer needs a real window. MUJOCO_GL=egl (off-screen rendering for our
#    experiments) would block that, so clear it for this script only.
unset MUJOCO_GL

# 5. Open the viewer. MuJoCo finds the included files (scene.xml, arm files,
#    meshes) relative to the XML file itself, so no `cd` is needed.
echo "Opening $SCENE"
exec python -m mujoco.viewer --mjcf="$SCENE"
