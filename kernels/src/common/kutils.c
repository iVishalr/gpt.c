#include <stdarg.h>
#include <common/kutils.h>

void *AllocCheck(void *(*alloc_fn)(const size_t nbytes, const size_t alignment), const size_t nbytes, const size_t alignment) {
    CHECK_ERROR(alloc_fn == NULL, "Expected callback function to be a valid function pointer. Got NULL");
    CHECK_ERROR(nbytes <= 0, "Expected allocation size (nbytes) to be a valid positive integer. Got %zu", nbytes);

    void *ptr = alloc_fn(nbytes, alignment);

    CHECK_ERROR(ptr == NULL, "Failed to allocate aligned memory. Requested allocation size = %zu bytes.", nbytes);
    return ptr;
}