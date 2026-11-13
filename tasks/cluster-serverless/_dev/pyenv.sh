# Which Python world regeneration runs on. Sourced, not executed.
#
#   CLUSTER_PYTHON   an explicit choice always wins
#   python3          the current environment on PATH
#
# Do not silently switch to a repository or Harbor virtualenv: dependencies
# installed with `python3 -m pip` must be the ones these commands use, too.
_pick_python() {
  if [ -n "${CLUSTER_PYTHON:-}" ]; then echo "$CLUSTER_PYTHON"; return; fi
  command -v python3
}
