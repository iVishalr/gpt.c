#include "runtime.h"
#include "dispatch.h"

void runtime_init(const device_t device) {
    runtime_init_dispatch(device);
}

void runtime_destroy(const device_t device) {
    runtime_destroy_dispatch(device);
}

void synchronize(const device_t device) {
    synchronize_dispatch(device);
}