#pragma once

#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime_api.h>

inline void hip_check(const hipError_t error, const char *cmd, const char *file,
                      const int line) {
  if (hipSuccess != error) {
    std::fprintf(stderr,
                 "[HIP ERROR] Fatal GPU Exception\n"
                 "\tCommand: %s\n"
                 "\tError: %s (Code: %d)\n"
                 "\tFile: %s:%d\n",
                 cmd, hipGetErrorString(error), static_cast<int>(error), file,
                 line);
    std::abort();
  }
}

#define HIP_CHECK(call) hip_check(call, #call, __FILE__, __LINE__)