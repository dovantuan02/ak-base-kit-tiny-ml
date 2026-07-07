#ifndef CMSIS_COMPILER_H
#define CMSIS_COMPILER_H

#include <stdint.h>

#ifndef __ASM
#define __ASM __asm
#endif

#ifndef __INLINE
#define __INLINE inline
#endif

#ifndef __STATIC_INLINE
#define __STATIC_INLINE static inline
#endif

#ifndef __STATIC_FORCEINLINE
#define __STATIC_FORCEINLINE static inline __attribute__((always_inline))
#endif

#ifndef __NO_RETURN
#define __NO_RETURN __attribute__((__noreturn__))
#endif

#ifndef __USED
#define __USED __attribute__((used))
#endif

#ifndef __WEAK
#define __WEAK __attribute__((weak))
#endif

#ifndef __PACKED
#define __PACKED __attribute__((packed))
#endif

#ifndef __PACKED_STRUCT
#define __PACKED_STRUCT __attribute__((packed))
#endif

#ifndef __PACKED_UNION
#define __PACKED_UNION __attribute__((packed))
#endif

#ifndef __ALIGNED
#define __ALIGNED(x) __attribute__((aligned(x)))
#endif

#ifndef __RESTRICT
#define __RESTRICT __restrict
#endif

#ifndef __CMSIS_GCC
#define __CMSIS_GCC
#endif

#ifndef __UNALIGNED_UINT32_READ
#define __UNALIGNED_UINT32_READ(x) (*(const uint32_t *)__UNALIGNED_UINT32_READ_SAFE(x))
#endif
#ifndef __UNALIGNED_UINT32_WRITE
#define __UNALIGNED_UINT32_WRITE(addr, val) (*(uint32_t *)__UNALIGNED_UINT32_WRITE_SAFE(addr) = (val))
#endif
#ifndef __UNALIGNED_UINT16_READ
#define __UNALIGNED_UINT16_READ(x) (*(const uint16_t *)__UNALIGNED_UINT16_READ_SAFE(x))
#endif
#ifndef __UNALIGNED_UINT16_WRITE
#define __UNALIGNED_UINT16_WRITE(addr, val) (*(uint16_t *)__UNALIGNED_UINT16_WRITE_SAFE(addr) = (val))
#endif

#ifndef __UNALIGNED_UINT32_READ_SAFE
#define __UNALIGNED_UINT32_READ_SAFE(x) (x)
#endif
#ifndef __UNALIGNED_UINT32_WRITE_SAFE
#define __UNALIGNED_UINT32_WRITE_SAFE(addr) (addr)
#endif
#ifndef __UNALIGNED_UINT16_READ_SAFE
#define __UNALIGNED_UINT16_READ_SAFE(x) (x)
#endif
#ifndef __UNALIGNED_UINT16_WRITE_SAFE
#define __UNALIGNED_UINT16_WRITE_SAFE(addr) (addr)
#endif

#ifndef NO_INLINE
#define NO_INLINE __attribute__((noinline))
#endif

#ifndef SECTION_NOINIT
#define SECTION_NOINIT
#endif

#endif
