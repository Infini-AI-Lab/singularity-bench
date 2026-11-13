/* Read only root-owned bootstrap output. Values never come from tool arguments. */
#include <errno.h>
#include <fcntl.h>
#include <ctype.h>

static FILE *trusted_file(const char *path) {
    int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0 && errno == ENOENT) return NULL; /* offline/legacy probes */
    if (fd < 0) fail("open worker configuration");
    struct stat info;
    if (fstat(fd, &info) || !S_ISREG(info.st_mode) || info.st_uid != 0
            || (info.st_mode & 0022) || info.st_size > 1048576) {
        fputs("worker configuration is not trusted\n", stderr); exit(126);
    }
    FILE *file = fdopen(fd, "r");
    if (!file) fail("read worker configuration");
    return file;
}

static size_t worker_groups(gid_t *groups, size_t limit) {
    FILE *file = trusted_file("/run/breakwater/worker-groups");
    size_t count = 0;
    unsigned int group;
    if (!file) { groups[0] = WORKSPACE_GID; return 1; }
    while (fscanf(file, "%u", &group) == 1) {
        if (!group || group == AUTHOR_UID || count == limit) {
            fputs("invalid worker groups\n", stderr); exit(126);
        }
        groups[count++] = group;
    }
    if (ferror(file) || !feof(file) || !count) fail("parse worker groups");
    fclose(file);
    return count;
}

static void worker_environment(void) {
    FILE *file = trusted_file("/run/breakwater/worker.env");
    if (!file) return;
    char *entry = NULL;
    size_t capacity = 0;
    ssize_t length;
    while ((length = getdelim(&entry, &capacity, '\0', file)) != -1) {
        char *separator = strchr(entry, '=');
        if (entry[length - 1] != '\0' || !separator || separator == entry) {
            fputs("invalid worker environment\n", stderr); exit(126);
        }
        *separator = '\0';
        for (char *p = entry; *p; p++) {
            if (!isalnum((unsigned char)*p) && *p != '_') {
                fputs("invalid worker variable\n", stderr); exit(126);
            }
        }
        if (setenv(entry, separator + 1, 1)) fail("set worker environment");
    }
    if (ferror(file)) fail("read worker environment");
    free(entry);
    fclose(file);
}
