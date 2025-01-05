#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

void runtime_init(const device_t device);
void runtime_destroy(const device_t device);
void synchronize(const device_t device);

#ifdef __cplusplus
}
#endif
