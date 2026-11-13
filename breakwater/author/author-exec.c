/* Fixed native-CLI entrypoint. Capture CLI tool events independently of writable
 * trajectories, without ptracing the CLI and disabling its setuid worker calls.
 */
#define _GNU_SOURCE
#include <grp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "stdio_capture.h"

#ifndef AUTHOR_PYTHON
#error "Build with the image's trusted absolute Python interpreter path"
#endif

int main(int argc, char **argv) {
    if (getuid() != 1100 || geteuid() != 0) return 126;
    if (!access("/run/breakwater/external-capture-enabled", F_OK)
            && breakwater_capture_start()) return 126;
    gid_t groups[] = {1100, 1200};
    if (setgroups(2, groups) || setresgid(1100, 1100, 1100)
            || setresuid(1100, 1100, 1100) || close_range(3, ~0U, 0) || clearenv()) return 126;
    char **command = calloc((size_t)argc + 3, sizeof(char *));
    if (!command) return 126;
    command[0] = AUTHOR_PYTHON;
    command[1] = "-I";
    command[2] = "/opt/breakwater-runtime/launch.py";
    for (int i = 1; i < argc; ++i) command[i + 2] = argv[i];
    execv(command[0], command);
    perror("protected author exec");
    return 126;
}
