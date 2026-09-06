// dlssnr-shmctl — read and write the shared-memory header from shell.
//
// This exists because the helper CLI used to poke the header with
//
//     printf '\x01\x00\x00\x00' | dd of="$shm" bs=1 seek=20 count=4 conv=notrunc
//
// which is a hardcoded byte offset into a C++ struct. It was correct for exactly one layout: the
// field at offset 20 was `quit` in the v1 header and is `height` in v2, so the same line that used
// to stop the helper would instead have silently corrupted the frame size. Nothing about the shell
// could have caught that.
//
// Everything here derives its offsets from shm_protocol.h by including it, so the header and the
// tool cannot disagree. Adding a field is now a recompile rather than an arithmetic exercise.
#include "../common/shm_protocol.h"

#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

void Usage() {
    std::fprintf(stderr,
                 "usage: dlssnr-shmctl <shm-path> <command>\n"
                 "\n"
                 "commands:\n"
                 "  status   print the header, one 'key=value' per line\n"
                 "  quit     ask the helper and the layer to stand down\n"
                 "  resume   clear the quit flag and nudge the readers\n"
                 "  reset    re-initialise the whole header to defaults\n");
}

// Maps the header only. The two pixel regions are megabytes and nothing here reads them.
ShmHeader* MapHeader(const char* path, bool create, void** base, int* fdOut) {
    const int flags = create ? (O_RDWR | O_CREAT) : O_RDWR;
    const int fd = open(path, flags, 0600);
    if (fd < 0) return nullptr;

    struct stat st {};
    if (fstat(fd, &st) != 0) {
        close(fd);
        return nullptr;
    }

    if ((size_t) st.st_size < ShmTotalBytes()) {
        if (!create) {  // nothing has ever attached; there is nothing to talk to
            close(fd);
            return nullptr;
        }
        if (ftruncate(fd, (off_t) ShmTotalBytes()) != 0) {
            close(fd);
            return nullptr;
        }
    }

    void* m = mmap(nullptr, kHeaderBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) {
        close(fd);
        return nullptr;
    }

    *base = m;
    *fdOut = fd;
    return (ShmHeader*) m;
}

bool Initialised(const ShmHeader* h) {
    return h->magic.load() == kShmMagic && h->version.load() == kShmVersion;
}

void PrintStatus(const ShmHeader* h) {
    std::printf("initialised=%d\n", Initialised(h) ? 1 : 0);
    std::printf("magic=%#x\nversion=%u\n", h->magic.load(), h->version.load());
    if (!Initialised(h)) return;
    std::printf("seq_req=%u\nseq_resp=%u\n", h->seq_req.load(), h->seq_resp.load());
    std::printf("width=%u\nheight=%u\n", h->width.load(), h->height.load());
    std::printf("quit=%u\nheartbeat=%u\ncontrol_seq=%u\n", h->quit.load(), h->heartbeat.load(),
                h->controlSeq.load());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        Usage();
        return 2;
    }
    const char* path = argv[1];
    const char* cmd = argv[2];

    const bool create = std::strcmp(cmd, "resume") == 0 || std::strcmp(cmd, "reset") == 0;

    void* base = nullptr;
    int fd = -1;
    ShmHeader* h = MapHeader(path, create, &base, &fd);
    if (!h) {
        // No mapping and none wanted: for 'quit' that means nothing is running, which is success.
        return std::strcmp(cmd, "quit") == 0 ? 0 : 1;
    }

    int rc = 0;
    if (std::strcmp(cmd, "status") == 0) {
        PrintStatus(h);
    } else if (std::strcmp(cmd, "quit") == 0) {
        if (Initialised(h)) {
            h->quit.store(1);
            h->controlSeq.fetch_add(1);
        }
    } else if (std::strcmp(cmd, "resume") == 0) {
        if (!Initialised(h)) ShmInitDefaults(h);
        h->quit.store(0);
        h->controlSeq.fetch_add(1);
    } else if (std::strcmp(cmd, "reset") == 0) {
        ShmInitDefaults(h);
    } else {
        Usage();
        rc = 2;
    }

    msync(base, kHeaderBytes, MS_SYNC);
    munmap(base, kHeaderBytes);
    close(fd);
    return rc;
}
