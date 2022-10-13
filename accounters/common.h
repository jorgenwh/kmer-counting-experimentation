#ifndef COMMON_H_
#define COMMON_H_

static const uint64_t kEmpty = 0xffffffffffffffff;

struct Table {
  uint64_t *keys;
  uint32_t *values;
};

#endif // COMMON_H_
