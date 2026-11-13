/* Protected stdio at the receiving boundary. No submitted code runs here.
 *
 * The child is forked BEFORE the external observer attaches to this supervisor.
 * Worker children have their own pre-drop observation barrier. Author CLIs must
 * remain untraced: tracing their exec children would suppress setuid helpers.
 *
 * Each received buffer goes through an ordinary write() in this root process,
 * which the external observer records before CLI truncation or log rewriting.
 * Workers receive pipe ends only, never evidence files or supervisor control FDs.
 * No copy, mmap, AIO, device, or GPU operation is restricted by this mechanism.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stddef.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "stdio_capture.h"

enum { CHUNK = 16384 };

static void mark(const char *name) {
    if (prctl(PR_SET_NAME, name, 0, 0, 0)) _exit(126);
}

static void failed(void) {
    mark("gs-io-error");
    _exit(126);
}

static void close_fd(int *fd) {
    if (*fd >= 0) close(*fd);
    *fd = -1;
}

static void nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK)) failed();
}

/* Do not retain any of the launcher's other descriptors in the root parent.
 * The child keeps the original descriptors for its normal exec/close policy.
 */
static void close_other_fds(int *keep, size_t count) {
    unsigned int next = 3;
    for (;;) {
        unsigned int smallest = ~0U;
        for (size_t i = 0; i < count; ++i)
            if (keep[i] >= 0 && (unsigned int)keep[i] >= next && (unsigned int)keep[i] < smallest)
                smallest = (unsigned int)keep[i];
        if (smallest > next && close_range(next, smallest - 1, 0)) failed();
        if (smallest == ~0U) break;
        next = smallest + 1;
    }
}

/* The observer captures this write buffer even when the original consumer has
 * closed its pipe. Closing our read end then preserves ordinary SIGPIPE behavior.
 */
static int forward(int destination, const char *data, size_t length) {
    while (length) {
        ssize_t written = write(destination, data, length);
        if (written < 0 && errno == EINTR) continue;
        if (written < 0 && (errno == EPIPE || errno == EIO || errno == EBADF)) {
            mark("gs-io-closed");
            return 0;
        }
        if (written <= 0) failed();
        data += written;
        length -= (size_t)written;
    }
    return 1;
}

static void receive(int *source, int destination, size_t limit) {
    char data[CHUNK];
    ssize_t count = read(*source, data, limit < sizeof(data) ? limit : sizeof(data));
    if (count < 0 && (errno == EINTR || errno == EAGAIN)) return;
    if (count < 0) failed();
    if (!count || !forward(destination, data, (size_t)count)) close_fd(source);
}

static void exit_like(int status) {
    if (WIFEXITED(status)) _exit(WEXITSTATUS(status));
    if (WIFSIGNALED(status)) {
        int sig = WTERMSIG(status);
        signal(sig, SIG_DFL);
        sigset_t mask;
        sigemptyset(&mask);
        sigaddset(&mask, sig);
        sigprocmask(SIG_UNBLOCK, &mask, NULL);
        raise(sig);
        _exit(128 + sig);
    }
    _exit(126);
}

static void supervise(pid_t child, int input, int output, int error) {
    int status = 0, exited = 0, tail = 0;
    int child_ready = 0, child_fd = (int)syscall(SYS_pidfd_open, child, 0);
    if (child_fd < 0) failed();
    size_t pending = 0, position = 0;
    char data[CHUNK];
    int source = 0;
    nonblocking(input);
    nonblocking(output);
    nonblocking(error);
    /* Only the supervisor ignores SIGPIPE; the child retains its original state. */
    signal(SIGPIPE, SIG_IGN);
    mark("gs-io-start");
    for (;;) {
        if (!exited && child_ready) {
            pid_t result = waitpid(child, &status, WNOHANG);
            if (result < 0 && errno != EINTR) failed();
            if (result == child) {
                exited = 1;
                close_fd(&child_fd);
                close_fd(&input);
                close_fd(&source);
                /* Flush all bytes already queued at foreground exit before its
                 * caller can read a temporary output file (mini's native API).
                 * Later background writes belong to the detached drainer.
                 */
                int *pipes[] = {&output, &error};
                for (size_t i = 0; i < 2; ++i) {
                    int remaining = 0;
                    if (*pipes[i] >= 0 && ioctl(*pipes[i], FIONREAD, &remaining)) failed();
                    while (*pipes[i] >= 0 && remaining > 0) {
                        int amount = remaining < CHUNK ? remaining : CHUNK;
                        receive(pipes[i], (int)i + 1, (size_t)amount);
                        remaining -= amount;
                    }
                }
                if (output < 0 && error < 0) {
                    mark("gs-io-end");
                    exit_like(status);
                }
                /* Do not wait for background descendants to close their stdio.
                 * Native pipe consumers still see EOF at the usual time; file
                 * consumers can return immediately after the foreground exit.
                 */
                pid_t drainer = fork();
                if (drainer < 0) failed();
                if (drainer) {
                    mark("gs-io-end");
                    exit_like(status);
                }
                tail = 1;
                mark("gs-io-start");
            }
        }
        if (tail && output < 0 && error < 0) {
            mark("gs-io-end");
            _exit(0);
        }
        struct pollfd fds[5] = {
            {output, POLLIN, 0}, {error, POLLIN, 0},
            {input >= 0 && pending == position ? source : -1, POLLIN, 0},
            {pending > position ? input : -1, POLLOUT, 0},
            {child_fd, POLLIN, 0},
        };
        int result = poll(fds, 5, -1);
        if (result < 0 && errno == EINTR) continue;
        if (result < 0) failed();
        if (fds[4].revents) child_ready = 1;
        if (output >= 0 && fds[0].revents) receive(&output, 1, CHUNK);
        if (error >= 0 && fds[1].revents) receive(&error, 2, CHUNK);
        if (fds[2].fd >= 0 && fds[2].revents) {
            ssize_t count = read(source, data, sizeof(data));
            if (count < 0 && errno == EINTR) continue;
            if (count < 0) failed();
            if (!count) {
                close_fd(&source);
                close_fd(&input);
            } else {
                pending = (size_t)count;
                position = 0;
            }
        }
        if (fds[3].fd >= 0 && fds[3].revents) {
            ssize_t count = write(input, data + position, pending - position);
            if (count < 0 && (errno == EINTR || errno == EAGAIN)) continue;
            if (count < 0 && errno == EPIPE) {
                close_fd(&input);
                close_fd(&source);
                pending = position = 0;
            } else if (count <= 0) failed();
            else position += (size_t)count;
        }
    }
}

int breakwater_capture_start(void) {
    if (geteuid() != 0) { errno = EPERM; return -1; }
    int pipes[3][2], gate[2];
    for (size_t i = 0; i < 3; ++i)
        if (pipe2(pipes[i], O_CLOEXEC)) return -1;
    if (pipe2(gate, O_CLOEXEC)) return -1;
    pid_t child = fork();
    if (child < 0) return -1;
    if (!child) {
        close(gate[1]);
        char ready;
        ssize_t count;
        do { count = read(gate[0], &ready, 1); } while (count < 0 && errno == EINTR);
        close(gate[0]);
        if (count != 1 || ready != 'R') _exit(126);
        for (size_t i = 0; i < 3; ++i) {
            int fd = pipes[i][i == 0 ? 0 : 1];
            if (dup2(fd, (int)i) < 0) _exit(126);
            close(pipes[i][0]);
            close(pipes[i][1]);
        }
        return 0;
    }
    close(gate[0]);
    close(pipes[0][0]); close(pipes[1][1]); close(pipes[2][1]);
    int keep[] = {pipes[0][1], pipes[1][0], pipes[2][0], gate[1]};
    /* No author/worker identity can signal or inspect this supervisor. */
    if (setresgid(0, 0, 0) || setresuid(0, 0, 0) || prctl(PR_SET_DUMPABLE, 0, 0, 0, 0)) failed();
    close_other_fds(keep, sizeof(keep) / sizeof(keep[0]));
    mark("gs-io-ready");
    if (raise(SIGSTOP)) failed();
    if (write(gate[1], "R", 1) != 1) failed();
    close(gate[1]);
    supervise(child, pipes[0][1], pipes[1][0], pipes[2][0]);
    return -1; /* unreachable */
}
