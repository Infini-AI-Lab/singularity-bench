# A private rootless podman store for the sandbox worker or the evaluator on a host without Docker. Source it in the
# shell that then loads the task images and starts the worker (or the evaluator):
#
#   PODMAN_STORE=/local/disk/podman-store source tasks/rl-swe-agent/worker/podman/podman-store.sh
#   podman load -i IMAGES.tar ...            # the native arm64 task images (arm64_tasks.py, saved with podman save)
#   ...                                      # start the worker or run the evaluator in this shell
#   podman_store_remove                      # afterwards: stop every sandbox and delete the store
#
# PODMAN_STORE is a new directory on node-local disk with room for the images (tens of GB for the training tasks,
# hundreds for the evaluation's SWE-bench images). Everything podman keeps goes under it, so a stale configuration or
# database in the home directory (an older store pinned to another storage driver) cannot interfere, and two jobs on
# one node keep separate stores. The settings are the ones rootless podman needs on a host without subordinate UID
# ranges or a systemd user session:
#   ignore_chown_errors  image layers keep files owned by other users, which do not exist in a single-user mapping
#                        (they become the calling user's)
#   userns host          no further mapping beyond rootless podman's own (container root is the calling user)
#   pidns private        each sandbox its own PID namespace, whatever the site's containers.conf says
#   netns slirp4netns    user-mode networking; sandboxes reach the host's loopback at 10.0.2.2 (podman_env.py)
#   cgroupfs, file       no systemd user session: plain cgroups and a file event log
# shellcheck shell=bash
: "${PODMAN_STORE:?set PODMAN_STORE to a new directory on node-local disk}"
if [ -e "$PODMAN_STORE" ] && [ -n "$(ls -A "$PODMAN_STORE" 2>/dev/null)" ]; then
  echo "podman-store.sh: $PODMAN_STORE is not empty; use a new directory" >&2
  return 1 2>/dev/null || exit 1
fi
# Rootless podman keeps its store in $XDG_DATA_HOME/containers/storage and its run state in $XDG_RUNTIME_DIR.
mkdir -p "$PODMAN_STORE"/config/containers "$PODMAN_STORE"/data "$PODMAN_STORE"/runtime "$PODMAN_STORE"/tmp
chmod 700 "$PODMAN_STORE"/runtime
export XDG_CONFIG_HOME="$PODMAN_STORE/config" XDG_DATA_HOME="$PODMAN_STORE/data" \
  XDG_RUNTIME_DIR="$PODMAN_STORE/runtime" CONTAINERS_STORAGE_CONF="$PODMAN_STORE/config/containers/storage.conf" \
  CONTAINERS_CONF="$PODMAN_STORE/config/containers/containers.conf"
cat > "$CONTAINERS_STORAGE_CONF" <<EOF
[storage]
driver = "overlay"

[storage.options.overlay]
ignore_chown_errors = "true"
EOF
cat > "$CONTAINERS_CONF" <<EOF
[containers]
userns = "host"
pidns = "private"
netns = "slirp4netns"

[engine]
cgroup_manager = "cgroupfs"
events_logger = "file"
image_copy_tmp_dir = "$PODMAN_STORE/tmp"
image_parallel_copies = 8
EOF
podman info --format '{{.Host.Security.Rootless}} {{.Store.GraphDriverName}} {{.Store.GraphRoot}}' || {
  echo "podman-store.sh: podman cannot use the store in $PODMAN_STORE" >&2
  return 1 2>/dev/null || exit 1
}

podman_store_remove() {  # stop and remove every sandbox, then the store itself
  podman rm --all --force --time 10 >/dev/null 2>&1
  podman system reset --force >/dev/null 2>&1
  rm -rf "$PODMAN_STORE"
}
