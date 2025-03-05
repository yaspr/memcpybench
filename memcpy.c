//
#include <string.h>

//
#include "memcpy.h"

//
void memcpy_C(u8 *restrict dst, u8 *restrict src, u64 n)
{
  for (u64 i = 0; i < n; i++)
    dst[i] = src[i];
}

//
void memcpy_openmp(u8 *restrict dst, u8 *restrict src, u64 n)
{
#pragma omp parallel for simd
  for (u64 i = 0; i < n; i++)
    dst[i] = src[i];
}

//
void memcpy_memcpy(u8 *restrict dst, u8 *restrict src, u64 n)
{
  memcpy((void *)dst, (void *)src, n);
}

//
void memcpy_asm(u8 *restrict dst, u8 *restrict src, u64 n)
{
  //
  const u64 _n = n & ~(n & 7);
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"
		    
		    "1:;\n"
		    
		    "mov (%[_src], %%rcx), %%rax;\n"
		    "mov %%rax, (%[_dst], %%rcx);\n"
		    
		    "add $8, %%rcx;\n"
		    "cmp %[_n], %%rcx;\n"
		    "jl 1b;\n"
		    
		    :
		    :
		    [_dst] "r" (dst),
		    [_src] "r" (src),
		    [_n]   "r" (_n)
		    
		    :
		    "cc", "memory", "rcx"
		    );
  
  for (u64 i = _n; i < n; i++)
    dst[i] = src[i];
}

//
void memcpy_SSE_u1(u8 *restrict dst, u8 *restrict src, u64 n)
{
  //
  const u64 _n = n & ~(n & 15);
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"
		    
		    "1:;\n"
		    
		    "movdqa (%[_src], %%rcx), %%xmm0;\n"
		    "movdqa %%xmm0, (%[_dst], %%rcx);\n"
		    
		    "add $16, %%rcx;\n"
		    "cmp %[_n], %%rcx;\n"
		    "jl 1b;\n"
		    
		    :
		    
		    :
		    [_dst] "r" (dst),
		    [_src] "r" (src),
		    [_n]   "r" (_n)
		    
		    :
		    "cc", "memory", "rcx", "xmm0"
		    );

  for (u64 i = _n; i < n; i++)
    dst[i] = src[i];
}

//
void memcpy_SSE_u2(u8 *restrict dst, u8 *restrict src, u64 n)
{
  //
  const u64 _n = n & ~(n & 31);
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"
		    
		    "1:;\n"
		    
		    "movdqa   (%[_src], %%rcx), %%xmm0;\n"
		    "movdqa 16(%[_src], %%rcx), %%xmm1;\n"
		    "movdqa %%xmm0,   (%[_dst], %%rcx);\n"
		    "movdqa %%xmm1, 16(%[_dst], %%rcx);\n"
		    
		    "add $32, %%rcx;\n"
		    "cmp %[_n], %%rcx;\n"
		    "jl 1b;\n"
		    
		    :
		    
		    :
		    [_dst] "r" (dst),
		    [_src] "r" (src),
		    [_n]   "r" (_n)
		    
		    :
		    "cc", "memory", "rcx",
		    "xmm0", "xmm1"
		    );
  
  for (u64 i = _n; i < n; i++)
    dst[i] = src[i];
}

//
void memcpy_SSE_u4(u8 *restrict dst, u8 *restrict src, u64 n)
{
  //
  const u64 _n = n & ~(n & 63);
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"
		    
		    "1:;\n"
		    
		    "movdqa   (%[_src], %%rcx), %%xmm0;\n"
		    "movdqa 16(%[_src], %%rcx), %%xmm1;\n"
		    "movdqa 32(%[_src], %%rcx), %%xmm2;\n"
		    "movdqa 48(%[_src], %%rcx), %%xmm3;\n"
		    "movdqa %%xmm0,   (%[_dst], %%rcx);\n"
		    "movdqa %%xmm1, 16(%[_dst], %%rcx);\n"
		    "movdqa %%xmm2, 32(%[_dst], %%rcx);\n"
		    "movdqa %%xmm3, 48(%[_dst], %%rcx);\n"
		    
		    "add $64, %%rcx;\n"
		    "cmp %[_n], %%rcx;\n"
		    "jl 1b;\n"
		    
		    :
		    
		    :
		    [_dst] "r" (dst),
		    [_src] "r" (src),
		    [_n]   "r" (_n)
		    
		    :
		    "cc", "memory", "rcx",
		    "xmm0", "xmm1", "xmm2", "xmm3"
		    );
  
  for (u64 i = _n; i < n; i++)
    dst[i] = src[i];
}

//
void memcpy_SSE_u8(u8 *restrict dst, u8 *restrict src, u64 n)
{
  //
  const u64 _n = n & ~(n & 127);
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"
		    
		    "1:;\n"
		    
		    "movdqa    (%[_src], %%rcx), %%xmm0;\n"
		    "movdqa  16(%[_src], %%rcx), %%xmm1;\n"
		    "movdqa  32(%[_src], %%rcx), %%xmm2;\n"
		    "movdqa  48(%[_src], %%rcx), %%xmm3;\n"
		    "movdqa  64(%[_src], %%rcx), %%xmm4;\n"
		    "movdqa  80(%[_src], %%rcx), %%xmm5;\n"
		    "movdqa  96(%[_src], %%rcx), %%xmm6;\n"
		    "movdqa 112(%[_src], %%rcx), %%xmm7;\n"
		    "movdqa %%xmm0,    (%[_dst], %%rcx);\n"
		    "movdqa %%xmm1,  16(%[_dst], %%rcx);\n"
		    "movdqa %%xmm2,  32(%[_dst], %%rcx);\n"
		    "movdqa %%xmm3,  48(%[_dst], %%rcx);\n"
		    "movdqa %%xmm4,  64(%[_dst], %%rcx);\n"
		    "movdqa %%xmm5,  80(%[_dst], %%rcx);\n"
		    "movdqa %%xmm6,  96(%[_dst], %%rcx);\n"
		    "movdqa %%xmm7, 112(%[_dst], %%rcx);\n"
		    
		    "add $128, %%rcx;\n"
		    "cmp %[_n], %%rcx;\n"
		    "jl 1b;\n"
		    
		    :
		    
		    :
		    [_dst] "r" (dst),
		    [_src] "r" (src),
		    [_n]   "r" (_n)
		    
		    :
		    "cc", "memory", "rcx",
		    "xmm0", "xmm1", "xmm2", "xmm3",
		    "xmm4", "xmm5", "xmm6", "xmm7"
		    );
  
  for (u64 i = _n; i < n; i++)
    dst[i] = src[i];
}

//
void memcpy_SSE_u8_nt(u8 *restrict dst, u8 *restrict src, u64 n)
{
  //
  const u64 _n = n & ~(n & 127);
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"
		    
		    "1:;\n"
		    
		    "movdqa    (%[_src], %%rcx), %%xmm0;\n"
		    "movdqa  16(%[_src], %%rcx), %%xmm1;\n"
		    "movdqa  32(%[_src], %%rcx), %%xmm2;\n"
		    "movdqa  48(%[_src], %%rcx), %%xmm3;\n"
		    "movdqa  64(%[_src], %%rcx), %%xmm4;\n"
		    "movdqa  80(%[_src], %%rcx), %%xmm5;\n"
		    "movdqa  96(%[_src], %%rcx), %%xmm6;\n"
		    "movdqa 112(%[_src], %%rcx), %%xmm7;\n"
		    "movntdq %%xmm0,    (%[_dst], %%rcx);\n"
		    "movntdq %%xmm1,  16(%[_dst], %%rcx);\n"
		    "movntdq %%xmm2,  32(%[_dst], %%rcx);\n"
		    "movntdq %%xmm3,  48(%[_dst], %%rcx);\n"
		    "movntdq %%xmm4,  64(%[_dst], %%rcx);\n"
		    "movntdq %%xmm5,  80(%[_dst], %%rcx);\n"
		    "movntdq %%xmm6,  96(%[_dst], %%rcx);\n"
		    "movntdq %%xmm7, 112(%[_dst], %%rcx);\n"
		    
		    "add $128, %%rcx;\n"
		    "cmp %[_n], %%rcx;\n"
		    "jl 1b;\n"
		    
		    :
		    
		    :
		    [_dst] "r" (dst),
		    [_src] "r" (src),
		    [_n]   "r" (_n)
		    
		    :
		    "cc", "memory", "rcx",
		    "xmm0", "xmm1", "xmm2", "xmm3",
		    "xmm4", "xmm5", "xmm6", "xmm7"
		    );
  
  for (u64 i = _n; i < n; i++)
    dst[i] = src[i];
}

//
void memcpy_AVX_u1(u8 *restrict dst, u8 *restrict src, u64 n)
{
  //
  const u64 _n = n & ~(n & 31);
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"
		    
		    "1:;\n"
		    
		    "vmovdqa (%[_src], %%rcx), %%ymm0;\n"
		    "vmovdqa %%ymm0, (%[_dst], %%rcx);\n"
		    
		    "add $32, %%rcx;\n"
		    "cmp %[_n], %%rcx;\n"
		    "jl 1b;\n"
		    
		    :
		    
		    :
		    [_dst] "r" (dst),
		    [_src] "r" (src),
		    [_n]   "r" (_n)
		    
		    :
		    "cc", "memory", "rcx", "ymm0"
		    );

  for (u64 i = _n; i < n; i++)
    dst[i] = src[i];
}

//
void memcpy_AVX_u2(u8 *restrict dst, u8 *restrict src, u64 n)
{
  //
  const u64 _n = n & ~(n & 63);
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"
		    
		    "1:;\n"
		    
		    "vmovdqa   (%[_src], %%rcx), %%ymm0;\n"
		    "vmovdqa 32(%[_src], %%rcx), %%ymm1;\n"
		    "vmovdqa %%ymm0,   (%[_dst], %%rcx);\n"
		    "vmovdqa %%ymm1, 32(%[_dst], %%rcx);\n"
		    
		    "add $64, %%rcx;\n"
		    "cmp %[_n], %%rcx;\n"
		    "jl 1b;\n"
		    
		    :
		    
		    :
		    [_dst] "r" (dst),
		    [_src] "r" (src),
		    [_n]   "r" (_n)
		    
		    :
		    "cc", "memory", "rcx",
		    "ymm0", "ymm1"
		    );

  for (u64 i = _n; i < n; i++)
    dst[i] = src[i];
}

//
void memcpy_AVX_u4(u8 *restrict dst, u8 *restrict src, u64 n)
{
  //
  const u64 _n = n & ~(n & 127);
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"
		    
		    "1:;\n"
		    
		    "vmovdqa   (%[_src], %%rcx), %%ymm0;\n"
		    "vmovdqa 32(%[_src], %%rcx), %%ymm1;\n"
		    "vmovdqa 64(%[_src], %%rcx), %%ymm2;\n"
		    "vmovdqa 96(%[_src], %%rcx), %%ymm3;\n"
		    "vmovdqa %%ymm0,   (%[_dst], %%rcx);\n"
		    "vmovdqa %%ymm1, 32(%[_dst], %%rcx);\n"
		    "vmovdqa %%ymm2, 64(%[_dst], %%rcx);\n"
		    "vmovdqa %%ymm3, 96(%[_dst], %%rcx);\n"
		    
		    "add $128, %%rcx;\n"
		    "cmp %[_n], %%rcx;\n"
		    "jl 1b;\n"
		    
		    :
		    
		    :
		    [_dst] "r" (dst),
		    [_src] "r" (src),
		    [_n]   "r" (_n)
		    
		    :
		    "cc", "memory", "rcx",
		    "ymm0", "ymm1", "ymm2", "ymm3" 
		    );

  for (u64 i = _n; i < n; i++)
    dst[i] = src[i];
}

//
void memcpy_AVX_u8(u8 *restrict dst, u8 *restrict src, u64 n)
{
  //
  const u64 _n = n & ~(n & 255);
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"
		    
		    "1:;\n"
		    
		    "vmovdqa    (%[_src], %%rcx), %%ymm0;\n"
		    "vmovdqa  32(%[_src], %%rcx), %%ymm1;\n"
		    "vmovdqa  64(%[_src], %%rcx), %%ymm2;\n"
		    "vmovdqa  96(%[_src], %%rcx), %%ymm3;\n"
		    "vmovdqa 128(%[_src], %%rcx), %%ymm4;\n"
		    "vmovdqa 160(%[_src], %%rcx), %%ymm5;\n"
		    "vmovdqa 192(%[_src], %%rcx), %%ymm6;\n"
		    "vmovdqa 224(%[_src], %%rcx), %%ymm7;\n"
		    "vmovdqa %%ymm0,    (%[_dst], %%rcx);\n"
		    "vmovdqa %%ymm1,  32(%[_dst], %%rcx);\n"
		    "vmovdqa %%ymm2,  64(%[_dst], %%rcx);\n"
		    "vmovdqa %%ymm3,  96(%[_dst], %%rcx);\n"
		    "vmovdqa %%ymm4, 128(%[_dst], %%rcx);\n"
		    "vmovdqa %%ymm5, 160(%[_dst], %%rcx);\n"
		    "vmovdqa %%ymm6, 192(%[_dst], %%rcx);\n"
		    "vmovdqa %%ymm7, 224(%[_dst], %%rcx);\n"
		    
		    "add $256, %%rcx;\n"
		    "cmp %[_n], %%rcx;\n"
		    "jl 1b;\n"
		    
		    :
		    
		    :
		    [_dst] "r" (dst),
		    [_src] "r" (src),
		    [_n]   "r" (_n)
		    
		    :
		    "cc", "memory", "rcx",
		    "ymm0", "ymm1", "ymm2", "ymm3",
		    "ymm4", "ymm5", "ymm6", "ymm7"
		    );
  
  for (u64 i = _n; i < n; i++)
    dst[i] = src[i];
}

//
void memcpy_AVX_u8_nt(u8 *restrict dst, u8 *restrict src, u64 n)
{
  //
  const u64 _n = n & ~(n & 255);
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"
		    
		    "1:;\n"
		    
		    "vmovdqa    (%[_src], %%rcx), %%ymm0;\n"
		    "vmovdqa  32(%[_src], %%rcx), %%ymm1;\n"
		    "vmovdqa  64(%[_src], %%rcx), %%ymm2;\n"
		    "vmovdqa  96(%[_src], %%rcx), %%ymm3;\n"
		    "vmovdqa 128(%[_src], %%rcx), %%ymm4;\n"
		    "vmovdqa 160(%[_src], %%rcx), %%ymm5;\n"
		    "vmovdqa 192(%[_src], %%rcx), %%ymm6;\n"
		    "vmovdqa 224(%[_src], %%rcx), %%ymm7;\n"
		    "vmovntdq %%ymm0,    (%[_dst], %%rcx);\n"
		    "vmovntdq %%ymm1,  32(%[_dst], %%rcx);\n"
		    "vmovntdq %%ymm2,  64(%[_dst], %%rcx);\n"
		    "vmovntdq %%ymm3,  96(%[_dst], %%rcx);\n"
		    "vmovntdq %%ymm4, 128(%[_dst], %%rcx);\n"
		    "vmovntdq %%ymm5, 160(%[_dst], %%rcx);\n"
		    "vmovntdq %%ymm6, 192(%[_dst], %%rcx);\n"
		    "vmovntdq %%ymm7, 224(%[_dst], %%rcx);\n"
		    
		    "add $256, %%rcx;\n"
		    "cmp %[_n], %%rcx;\n"
		    "jl 1b;\n"
		    
		    :
		    
		    :
		    [_dst] "r" (dst),
		    [_src] "r" (src),
		    [_n]   "r" (_n)
		    
		    :
		    "cc", "memory", "rcx",
		    "ymm0", "ymm1", "ymm2", "ymm3",
		    "ymm4", "ymm5", "ymm6", "ymm7"
		    );
  
  for (u64 i = _n; i < n; i++)
    dst[i] = src[i];
}

//
void memcpy_AVX_u16(u8 *restrict dst, u8 *restrict src, u64 n)
{
  //
  const u64 _n = n & ~(n & 511);
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"
		    
		    "1:;\n"
		    
		    "vmovdqa    (%[_src], %%rcx), %%ymm0;\n"
		    "vmovdqa  32(%[_src], %%rcx), %%ymm1;\n"
		    "vmovdqa  64(%[_src], %%rcx), %%ymm2;\n"
		    "vmovdqa  96(%[_src], %%rcx), %%ymm3;\n"
		    "vmovdqa 128(%[_src], %%rcx), %%ymm4;\n"
		    "vmovdqa 160(%[_src], %%rcx), %%ymm5;\n"
		    "vmovdqa 192(%[_src], %%rcx), %%ymm6;\n"
		    "vmovdqa 224(%[_src], %%rcx), %%ymm7;\n"
		    "vmovdqa 256(%[_src], %%rcx), %%ymm8;\n"
		    "vmovdqa 288(%[_src], %%rcx), %%ymm9;\n"
		    "vmovdqa 320(%[_src], %%rcx), %%ymm10;\n"
		    "vmovdqa 352(%[_src], %%rcx), %%ymm11;\n"
		    "vmovdqa 384(%[_src], %%rcx), %%ymm12;\n"
		    "vmovdqa 416(%[_src], %%rcx), %%ymm13;\n"
		    "vmovdqa 448(%[_src], %%rcx), %%ymm14;\n"
		    "vmovdqa 480(%[_src], %%rcx), %%ymm15;\n"
		    
		    "vmovdqa %%ymm0,    (%[_dst], %%rcx);\n"
		    "vmovdqa %%ymm1,  32(%[_dst], %%rcx);\n"
		    "vmovdqa %%ymm2,  64(%[_dst], %%rcx);\n"
		    "vmovdqa %%ymm3,  96(%[_dst], %%rcx);\n"
		    "vmovdqa %%ymm4, 128(%[_dst], %%rcx);\n"
		    "vmovdqa %%ymm5, 160(%[_dst], %%rcx);\n"
		    "vmovdqa %%ymm6, 192(%[_dst], %%rcx);\n"
		    "vmovdqa %%ymm7, 224(%[_dst], %%rcx);\n"
		    "vmovdqa %%ymm0, 256(%[_dst], %%rcx);\n"
		    "vmovdqa %%ymm1, 288(%[_dst], %%rcx);\n"
		    "vmovdqa %%ymm2, 320(%[_dst], %%rcx);\n"
		    "vmovdqa %%ymm3, 352(%[_dst], %%rcx);\n"
		    "vmovdqa %%ymm4, 384(%[_dst], %%rcx);\n"
		    "vmovdqa %%ymm5, 416(%[_dst], %%rcx);\n"
		    "vmovdqa %%ymm6, 448(%[_dst], %%rcx);\n"
		    "vmovdqa %%ymm7, 480(%[_dst], %%rcx);\n"
		    
		    "add $512, %%rcx;\n"
		    "cmp %[_n], %%rcx;\n"
		    "jl 1b;\n"
		    
		    :
		    
		    :
		    [_dst] "r" (dst),
		    [_src] "r" (src),
		    [_n]   "r" (_n)
		    
		    :
		    "cc", "memory", "rcx",
		    "ymm0", "ymm1", "ymm2", "ymm3",
		    "ymm4", "ymm5", "ymm6", "ymm7",
		    "ymm8", "ymm9", "ymm10", "ymm11",
		    "ymm12", "ymm13", "ymm14", "ymm15"

		    );
  
  for (u64 i = _n; i < n; i++)
    dst[i] = src[i];
}

//
void memcpy_AVX_u16_nt(u8 *restrict dst, u8 *restrict src, u64 n)
{
  //
  const u64 _n = n & ~(n & 511);
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"
		    
		    "1:;\n"
		    
		    "vmovdqa    (%[_src], %%rcx), %%ymm0;\n"
		    "vmovdqa  32(%[_src], %%rcx), %%ymm1;\n"
		    "vmovdqa  64(%[_src], %%rcx), %%ymm2;\n"
		    "vmovdqa  96(%[_src], %%rcx), %%ymm3;\n"
		    "vmovdqa 128(%[_src], %%rcx), %%ymm4;\n"
		    "vmovdqa 160(%[_src], %%rcx), %%ymm5;\n"
		    "vmovdqa 192(%[_src], %%rcx), %%ymm6;\n"
		    "vmovdqa 224(%[_src], %%rcx), %%ymm7;\n"
		    "vmovdqa 256(%[_src], %%rcx), %%ymm8;\n"
		    "vmovdqa 288(%[_src], %%rcx), %%ymm9;\n"
		    "vmovdqa 320(%[_src], %%rcx), %%ymm10;\n"
		    "vmovdqa 352(%[_src], %%rcx), %%ymm11;\n"
		    "vmovdqa 384(%[_src], %%rcx), %%ymm12;\n"
		    "vmovdqa 416(%[_src], %%rcx), %%ymm13;\n"
		    "vmovdqa 448(%[_src], %%rcx), %%ymm14;\n"
		    "vmovdqa 480(%[_src], %%rcx), %%ymm15;\n"
		    
		    "vmovntdq %%ymm0,    (%[_dst], %%rcx);\n"
		    "vmovntdq %%ymm1,  32(%[_dst], %%rcx);\n"
		    "vmovntdq %%ymm2,  64(%[_dst], %%rcx);\n"
		    "vmovntdq %%ymm3,  96(%[_dst], %%rcx);\n"
		    "vmovntdq %%ymm4, 128(%[_dst], %%rcx);\n"
		    "vmovntdq %%ymm5, 160(%[_dst], %%rcx);\n"
		    "vmovntdq %%ymm6, 192(%[_dst], %%rcx);\n"
		    "vmovntdq %%ymm7, 224(%[_dst], %%rcx);\n"
		    "vmovntdq %%ymm0, 256(%[_dst], %%rcx);\n"
		    "vmovntdq %%ymm1, 288(%[_dst], %%rcx);\n"
		    "vmovntdq %%ymm2, 320(%[_dst], %%rcx);\n"
		    "vmovntdq %%ymm3, 352(%[_dst], %%rcx);\n"
		    "vmovntdq %%ymm4, 384(%[_dst], %%rcx);\n"
		    "vmovntdq %%ymm5, 416(%[_dst], %%rcx);\n"
		    "vmovntdq %%ymm6, 448(%[_dst], %%rcx);\n"
		    "vmovntdq %%ymm7, 480(%[_dst], %%rcx);\n"
		    
		    "add $512, %%rcx;\n"
		    "cmp %[_n], %%rcx;\n"
		    "jl 1b;\n"
		    
		    :
		    
		    :
		    [_dst] "r" (dst),
		    [_src] "r" (src),
		    [_n]   "r" (_n)
		    
		    :
		    "cc", "memory", "rcx",
		    "ymm0", "ymm1", "ymm2", "ymm3",
		    "ymm4", "ymm5", "ymm6", "ymm7",
		    "ymm8", "ymm9", "ymm10", "ymm11",
		    "ymm12", "ymm13", "ymm14", "ymm15"

		    );
  
  for (u64 i = _n; i < n; i++)
    dst[i] = src[i];
}

//
void memcpy_AVX512_u1(u8 *restrict dst, u8 *restrict src, u64 n)
{
  //
  const u64 _n = n & ~(n & 63);
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"
		    
		    "1:;\n"
		    
		    "vmovdqa64 (%[_src], %%rcx), %%zmm0;\n"
		    "vmovdqa64 %%zmm0, (%[_dst], %%rcx);\n"
		    
		    "add $64, %%rcx;\n"
		    "cmp %[_n], %%rcx;\n"
		    "jl 1b;\n"
		    
		    :
		    
		    :
		    [_dst] "r" (dst),
		    [_src] "r" (src),
		    [_n]   "r" (_n)
		    
		    :
		    "cc", "memory", "rcx", "zmm0"
		    );

  for (u64 i = _n; i < n; i++)
    dst[i] = src[i];
}

//
void memcpy_AVX512_u2(u8 *restrict dst, u8 *restrict src, u64 n)
{
  //
  const u64 _n = n & ~(n & 127);
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"
		    
		    "1:;\n"
		    
		    "vmovdqa64   (%[_src], %%rcx), %%zmm0;\n"
		    "vmovdqa64 64(%[_src], %%rcx), %%zmm1;\n"
		    "vmovdqa64 %%zmm0,   (%[_dst], %%rcx);\n"
		    "vmovdqa64 %%zmm1, 64(%[_dst], %%rcx);\n"
		    
		    "add $128, %%rcx;\n"
		    "cmp %[_n], %%rcx;\n"
		    "jl 1b;\n"
		    
		    :
		    
		    :
		    [_dst] "r" (dst),
		    [_src] "r" (src),
		    [_n]   "r" (_n)
		    
		    :
		    "cc", "memory", "rcx",
		    "zmm0", "zmm1"
		    );

  for (u64 i = _n; i < n; i++)
    dst[i] = src[i];
}

//
void memcpy_AVX512_u4(u8 *restrict dst, u8 *restrict src, u64 n)
{
  //
  const u64 _n = n & ~(n & 255);
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"
		    
		    "1:;\n"
		    
		    "vmovdqa64    (%[_src], %%rcx), %%zmm0;\n"
		    "vmovdqa64  64(%[_src], %%rcx), %%zmm1;\n"
		    "vmovdqa64 128(%[_src], %%rcx), %%zmm2;\n"
		    "vmovdqa64 192(%[_src], %%rcx), %%zmm3;\n"
		    "vmovdqa64 %%zmm0,    (%[_dst], %%rcx);\n"
		    "vmovdqa64 %%zmm1,  64(%[_dst], %%rcx);\n"
		    "vmovdqa64 %%zmm2, 128(%[_dst], %%rcx);\n"
		    "vmovdqa64 %%zmm3, 192(%[_dst], %%rcx);\n"
		    
		    "add $256, %%rcx;\n"
		    "cmp %[_n], %%rcx;\n"
		    "jl 1b;\n"
		    
		    :
		    
		    :
		    [_dst] "r" (dst),
		    [_src] "r" (src),
		    [_n]   "r" (_n)
		    
		    :
		    "cc", "memory", "rcx",
		    "zmm0", "zmm1", "zmm2", "zmm3"
		    );

  for (u64 i = _n; i < n; i++)
    dst[i] = src[i];
}

//
void memcpy_AVX512_u8(u8 *restrict dst, u8 *restrict src, u64 n)
{
  //
  const u64 _n = n & ~(n & 511);
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"
		    
		    "1:;\n"
		    
		    "vmovdqa64    (%[_src], %%rcx), %%zmm0;\n"
		    "vmovdqa64  64(%[_src], %%rcx), %%zmm1;\n"
		    "vmovdqa64 128(%[_src], %%rcx), %%zmm2;\n"
		    "vmovdqa64 192(%[_src], %%rcx), %%zmm3;\n"
		    "vmovdqa64 256(%[_src], %%rcx), %%zmm4;\n"
		    "vmovdqa64 320(%[_src], %%rcx), %%zmm5;\n"
		    "vmovdqa64 384(%[_src], %%rcx), %%zmm6;\n"
		    "vmovdqa64 448(%[_src], %%rcx), %%zmm7;\n"
		    "vmovdqa64 %%zmm0,    (%[_dst], %%rcx);\n"
		    "vmovdqa64 %%zmm1,  64(%[_dst], %%rcx);\n"
		    "vmovdqa64 %%zmm2, 128(%[_dst], %%rcx);\n"
		    "vmovdqa64 %%zmm3, 192(%[_dst], %%rcx);\n"
		    "vmovdqa64 %%zmm4, 256(%[_dst], %%rcx);\n"
		    "vmovdqa64 %%zmm5, 320(%[_dst], %%rcx);\n"
		    "vmovdqa64 %%zmm6, 384(%[_dst], %%rcx);\n"
		    "vmovdqa64 %%zmm7, 448(%[_dst], %%rcx);\n"
		    
		    "add $512, %%rcx;\n"
		    "cmp %[_n], %%rcx;\n"
		    "jl 1b;\n"
		    
		    :
		    
		    :
		    [_dst] "r" (dst),
		    [_src] "r" (src),
		    [_n]   "r" (_n)
		    
		    :
		    "cc", "memory", "rcx",
		    "zmm0", "zmm1", "zmm2", "zmm3",
		    "zmm4", "zmm5", "zmm6", "zmm7"
		    );

  for (u64 i = _n; i < n; i++)
    dst[i] = src[i];
}

//
void memcpy_AVX512_u8_nt(u8 *restrict dst, u8 *restrict src, u64 n)
{
  //
  const u64 _n = n & ~(n & 511);
  
  __asm__ volatile (
		    "xor %%rcx, %%rcx;\n"
		    
		    "1:;\n"
		    
		    "vmovdqa64    (%[_src], %%rcx), %%zmm0;\n"
		    "vmovdqa64  64(%[_src], %%rcx), %%zmm1;\n"
		    "vmovdqa64 128(%[_src], %%rcx), %%zmm2;\n"
		    "vmovdqa64 192(%[_src], %%rcx), %%zmm3;\n"
		    "vmovdqa64 256(%[_src], %%rcx), %%zmm4;\n"
		    "vmovdqa64 320(%[_src], %%rcx), %%zmm5;\n"
		    "vmovdqa64 384(%[_src], %%rcx), %%zmm6;\n"
		    "vmovdqa64 448(%[_src], %%rcx), %%zmm7;\n"
		    "vmovntdq %%zmm0,    (%[_dst], %%rcx);\n"
		    "vmovntdq %%zmm1,  64(%[_dst], %%rcx);\n"
		    "vmovntdq %%zmm2, 128(%[_dst], %%rcx);\n"
		    "vmovntdq %%zmm3, 192(%[_dst], %%rcx);\n"
		    "vmovntdq %%zmm4, 256(%[_dst], %%rcx);\n"
		    "vmovntdq %%zmm5, 320(%[_dst], %%rcx);\n"
		    "vmovntdq %%zmm6, 384(%[_dst], %%rcx);\n"
		    "vmovntdq %%zmm7, 448(%[_dst], %%rcx);\n"
		    
		    "add $512, %%rcx;\n"
		    "cmp %[_n], %%rcx;\n"
		    "jl 1b;\n"
		    
		    :
		    
		    :
		    [_dst] "r" (dst),
		    [_src] "r" (src),
		    [_n]   "r" (_n)
		    
		    :
		    "cc", "memory", "rcx",
		    "zmm0", "zmm1", "zmm2", "zmm3",
		    "zmm4", "zmm5", "zmm6", "zmm7"
		    );

  for (u64 i = _n; i < n; i++)
    dst[i] = src[i];
}
