#!/usr/bin/env bash
# Start the JGI integration container and open JupyterLab in your browser (macOS and Linux; on Windows,
# run it from a WSL terminal). Keep this file next to docker-compose.yml in your input_data/ folder.
#
#   macOS:  double-click start.command in Finder, or run ./start.command (if macOS blocks a downloaded copy:
#           System Settings > Privacy & Security > Open Anyway, or run: bash start.command)
#   Linux:  ./start.command
#
#   ./start.command             newest published image (ghcr.io/bkieft-usa/jgi-integration:latest, for any CPU)
#   ./start.command --dev       your local build of docker/Dockerfile-dev (:dev), with dev/docker-compose-dev.yml
#   tag=<tag> ./start.command   a specific published image, e.g. tag=1.2.0 or tag=sha-1a2b3c4
#
# Leave this window open while you work. Press Ctrl+C here (or close the window) to stop the container.
set -euo pipefail

PROJECT="jgi-integration"
CONTAINER="jgi-integration"
JUPYTER_PORT=8888
EXPLORER_PORT=8050

SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")"     # input_data/ (Finder starts .command files in your home folder)

die() { echo; echo "ERROR: $*" >&2; echo; read -r -p "Press Return to close." _ || true; exit 1; }

if [ -f Dockerfile ] && [ -f docker-compose.example.yml ]; then
    die "this is the repository's copy of start.command (${SELF}), reached directly or through a symlink.
Copy the file itself into your input_data folder (cp, not ln -s) and run that copy."
fi

dev=0
for arg in "$@"; do
    case "$arg" in
        --dev) dev=1 ;;
        -h|--help) sed -n '2,13p' "$SELF" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option: $arg (see ./start.command --help)" ;;
    esac
done

# ---- image tag and compose file ----------------------------------------------------------------------
if [ "$dev" -eq 1 ]; then
    tag="${tag:-dev}"
    compose_file="dev/docker-compose-dev.yml"
    pull="never"       # dev images are built locally, never published
else
    tag="${tag:-latest}"
    compose_file="docker-compose.yml"
    pull="always"      # get the newest published image
fi
export tag
[ -f "$compose_file" ] || die "$(pwd)/${compose_file} not found. Keep start.command in your input_data folder."
[ -d config ] || die "no config/ folder in $(pwd). Keep start.command in your input_data folder."
mkdir -p ../output_data

# ---- Docker running? ---------------------------------------------------------------------------------
command -v docker >/dev/null 2>&1 || die "Docker is not installed. Install Docker Desktop (macOS) or Docker Engine (Linux)."
if ! docker info >/dev/null 2>&1; then
    if [ "$(uname -s)" = "Darwin" ]; then
        echo "Starting Docker Desktop..."
        open -a Docker || die "could not start Docker Desktop. Start it yourself, then run this again."
        for _ in $(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 2; done
    fi
    docker info >/dev/null 2>&1 || die "Docker is not running. Start Docker, then run this again."
fi
if [ "$dev" -eq 1 ] && ! docker image inspect "ghcr.io/bkieft-usa/jgi-integration:${tag}" >/dev/null 2>&1; then
    die "no local image ghcr.io/bkieft-usa/jgi-integration:${tag}. Build it from your jgi_integration checkout:
  docker build -t ghcr.io/bkieft-usa/jgi-integration:${tag} -f docker/Dockerfile-dev ."
fi
docker compose version >/dev/null 2>&1 || die "'docker compose' is missing. Update Docker Desktop, or install the docker-compose-plugin on Linux."

# ---- ports free? (our own container is replaced below, so it does not count) -----------------------------
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
for port in "$JUPYTER_PORT" "$EXPLORER_PORT"; do
    rc=0; curl -s -o /dev/null --max-time 2 "http://127.0.0.1:${port}/" || rc=$?
    [ "$rc" -eq 7 ] || die "port ${port} is already in use by another program (another JupyterLab or container?). Stop it and run this again."
done

# ---- start -------------------------------------------------------------------------------------------
compose() { docker compose -p "$PROJECT" -f "$compose_file" "$@"; }

echo "Starting ${CONTAINER} with image ghcr.io/bkieft-usa/jgi-integration:${tag}"
compose up -d --force-recreate --pull "$pull" || die "the container did not start (see the messages above).
If they say 'denied' or 'unauthorized', Docker is sending an old saved login for ghcr.io (the image itself
is public). Remove it with:  docker logout ghcr.io   and run this again."

stop_container() {
    trap - INT TERM HUP
    echo
    echo "Stopping ${CONTAINER}..."
    compose stop >/dev/null 2>&1 || true
    echo "Stopped. You can close this window."
    exit 0
}
trap stop_container INT TERM HUP

printf "Waiting for JupyterLab"
url=""
for _ in $(seq 1 90); do
    if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" != "true" ]; then
        echo; docker logs --tail 30 "$CONTAINER" >&2 || true
        die "the container stopped while starting (its log is above)."
    fi
    # `jupyter server list` prints e.g. http://<container-id>:8888/?token=<token> :: /home/jovyan/work
    token="$(docker exec "$CONTAINER" jupyter server list 2>/dev/null | sed -n 's/.*[?&]token=\([0-9a-zA-Z]*\).*/\1/p' | head -1 || true)"
    if [ -n "$token" ] && curl -s -o /dev/null --max-time 2 "http://127.0.0.1:${JUPYTER_PORT}/api"; then
        url="http://127.0.0.1:${JUPYTER_PORT}/lab?token=${token}"
        break
    fi
    printf "."; sleep 2
done
echo
[ -n "$url" ] || die "JupyterLab did not come up within 3 minutes. Check: docker logs ${CONTAINER}"

if [ "$(uname -s)" = "Darwin" ]; then
    open "$url"
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$url" >/dev/null 2>&1 &
elif command -v wslview >/dev/null 2>&1; then
    wslview "$url" >/dev/null 2>&1 &
fi

cat <<EOF

JupyterLab is running: ${url}
(If no browser tab opened, copy that link into your browser.)

Results explorer: start it from a JupyterLab terminal or the notebook, then open http://localhost:${EXPLORER_PORT}

Leave this window open while you work. Press Ctrl+C here to stop the container.
EOF

# Wait until the container exits on its own, or Ctrl+C / closing the window stops it.
docker wait "$CONTAINER" >/dev/null 2>&1 &
wait $! || true
echo "${CONTAINER} has stopped."
