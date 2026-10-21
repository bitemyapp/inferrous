// Host-only CUPTI activity collection for the pinned CUDA 13.4 toolchain.
// This adds no inference kernels and is loaded only by profile_kernels.sh.
#define _POSIX_C_SOURCE 200112L
#include <cupti.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

static FILE *trace;
static pthread_mutex_t trace_mutex = PTHREAD_MUTEX_INITIALIZER;
static uint64_t records, dropped_records, invalid_intervals;

static void check(CUptiResult result, const char *operation) {
    if (result != CUPTI_SUCCESS) {
        const char *message = "unknown CUPTI error";
        cuptiGetResultString(result, &message);
        fprintf(stderr, "kernel trace: %s: %s\n", operation, message);
        _Exit(2);
    }
}

static void CUPTIAPI request_buffer(uint8_t **buffer, size_t *size, size_t *maximum) {
    *size = 8 << 20;
    *maximum = 0;
    void *allocation = NULL;
    if (posix_memalign(&allocation, 64, *size) != 0) {
        fprintf(stderr, "kernel trace: cannot allocate an activity buffer\n");
        _Exit(2);
    }
    *buffer = allocation;
}

static void csv_name(const char *name) {
    fputc('"', trace);
    if (name) {
        for (const char *p = name; *p; ++p) {
            if (*p == '"') fputc('"', trace);
            fputc(*p, trace);
        }
    }
    fputc('"', trace);
}

static void CUPTIAPI complete_buffer(CUcontext context, uint32_t stream, uint8_t *buffer,
                                    size_t size, size_t valid_size) {
    (void)size;
    pthread_mutex_lock(&trace_mutex);
    CUpti_Activity *record = NULL;
    CUptiResult status;
    while ((status = cuptiActivityGetNextRecord(buffer, valid_size, &record)) == CUPTI_SUCCESS) {
        if (record->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) continue;
        const CUpti_ActivityKernel13 *kernel = (const CUpti_ActivityKernel13 *)record;
        ++records;
        if (kernel->start == 0 || kernel->end <= kernel->start) ++invalid_intervals;
        csv_name(kernel->name);
        fprintf(trace, ",%" PRIu64 ",%" PRIu64 ",%u,%u,%u,%d,%d,%d,%d,%d,%d,%u,%d,%d,%u,%u,%" PRIu64 "\n",
                kernel->start, kernel->end, kernel->deviceId, kernel->contextId, kernel->streamId,
                kernel->gridX, kernel->gridY, kernel->gridZ,
                kernel->blockX, kernel->blockY, kernel->blockZ,
                kernel->registersPerThread, kernel->staticSharedMemory, kernel->dynamicSharedMemory,
                kernel->localMemoryPerThread, kernel->graphId, kernel->graphNodeId);
    }
    if (status != CUPTI_ERROR_MAX_LIMIT_REACHED) check(status, "read activity buffer");
    size_t dropped = 0;
    check(cuptiActivityGetNumDroppedRecords(context, stream, &dropped), "read dropped record count");
    dropped_records += dropped;
    pthread_mutex_unlock(&trace_mutex);
    free(buffer);
}

static void finish(void) {
    // The synchronous engine fences its work before returning. Forced flush is
    // blocking and delivers all buffers; reject any incomplete/lost records.
    check(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED), "flush activities");
    check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL), "disable activity collection");
    pthread_mutex_lock(&trace_mutex);
    fprintf(trace, "# records=%" PRIu64 " dropped=%" PRIu64 " invalid=%" PRIu64 "\n",
            records, dropped_records, invalid_intervals);
    int failed = records == 0 || dropped_records != 0 || invalid_intervals != 0 || ferror(trace);
    if (fclose(trace) != 0) failed = 1;
    pthread_mutex_unlock(&trace_mutex);
    if (failed) {
        fprintf(stderr, "kernel trace: empty, incomplete, dropped or unwritten activity records\n");
        _Exit(2);
    }
}

__attribute__((constructor)) static void initialize(void) {
    const char *path = getenv("INFERROUS_KERNEL_TRACE");
    if (!path || !*path) return;
    trace = fopen(path, "wx");
    if (!trace) {
        perror("kernel trace: create output");
        _Exit(2);
    }
    const char *binary_sha = getenv("INFERROUS_TRACE_BINARY_SHA256");
    uint32_t version = 0;
    check(cuptiGetVersion(&version), "read CUPTI version");
    fprintf(trace, "# binary_sha256=%s cupti_api_version=%u\n", binary_sha ? binary_sha : "unknown", version);
    fputs("name,start_ns,end_ns,device,context,stream,grid_x,grid_y,grid_z,block_x,block_y,block_z,registers,static_shared,dynamic_shared,local_per_thread,graph_id,graph_node\n", trace);
    check(cuptiActivityRegisterCallbacks(request_buffer, complete_buffer), "register buffers");
    check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL), "enable kernel activities");
    if (atexit(finish) != 0) {
        fprintf(stderr, "kernel trace: cannot register final flush\n");
        _Exit(2);
    }
}
