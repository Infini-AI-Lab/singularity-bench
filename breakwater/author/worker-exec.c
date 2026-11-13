/* Root-owned 4750 root:author. The only privileged transition available to the
 * author is to the fixed worker identity, never to root or an arbitrary user.
 * No command, shell startup file, descriptor or environment is evaluated until
 * after all three uid/gid identities have been dropped.
 */
#define _GNU_SOURCE
#include <grp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <unistd.h>

enum { AUTHOR_UID = 1100, WORKER_UID = 1101, WORKSPACE_GID = 1200 };
static void fail(const char *what) { perror(what); exit(126); }
#include "worker-config.h"
#include "stdio_capture.h"

int main(int argc, char **argv) {
#ifdef GIT_MODE
    /* Internal CLI git calls use the same privilege boundary without a Python
     * interpreter or shell. Preserve argv literally, including empty values. */
    (void)argc;
#else
    if (argc != 2) {
        fputs("worker-exec: requires one command string\n", stderr);
        return 126;
    }
    const char *command = argv[1];
#endif
    if (getuid() != AUTHOR_UID || geteuid() != 0) {
        fputs("worker-exec: requires author uid and root effective uid\n", stderr);
        return 126;
    }
    /* Attach only after the setuid entry has already completed. The author CLI
     * is trusted; worker code cannot signal this root/author process. No evidence
     * directory, collector address, or collector credential enters the worker. */
    int capture = !access("/run/breakwater/external-capture-enabled", F_OK);
    if (capture) {
        if (breakwater_capture_start()) fail("protected stdio capture");
        if (raise(SIGSTOP)) fail("external observation barrier");
    }
    gid_t groups[64];
    size_t group_count = worker_groups(groups, 64);
    if (setgroups(group_count, groups) || setresgid(WORKER_UID, WORKER_UID, WORKER_UID)
            || setresuid(WORKER_UID, WORKER_UID, WORKER_UID)) fail("drop identity");
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)) fail("no_new_privs");
    if (close_range(3, ~0U, 0)) fail("close_range");
    if (clearenv()) fail("clearenv");
    if (setenv("HOME", "/home/worker", 1) || setenv("USER", "worker", 1)
            || setenv("LOGNAME", "worker", 1) || setenv("LANG", "C.UTF-8", 1)
            || setenv("PATH", "/usr/local/bin:/usr/bin:/bin", 1)
            || setenv("SHELL", "/bin/bash", 1)) fail("worker environment");
    worker_environment();
    umask(0002);
#ifdef GIT_MODE
    argv[0] = "git";
    execv("/usr/bin/git", argv);
    fail("exec worker git");
#else
    execl("/bin/bash", "bash", "--noprofile", "--norc", "-c", command, (char *)NULL);
    fail("exec worker shell");
#endif
}
