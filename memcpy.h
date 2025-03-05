#pragma once

#include "types.h"

void memcpy_C(u8 *restrict dst, u8 *restrict src, u64 n);
void memcpy_openmp(u8 *restrict dst, u8 *restrict src, u64 n);
void memcpy_memcpy(u8 *restrict dst, u8 *restrict src, u64 n);

void memcpy_asm(u8 *restrict dst, u8 *restrict src, u64 n);

void memcpy_SSE_u1(u8 *restrict dst, u8 *restrict src, u64 n);
void memcpy_SSE_u2(u8 *restrict dst, u8 *restrict src, u64 n);
void memcpy_SSE_u4(u8 *restrict dst, u8 *restrict src, u64 n);
void memcpy_SSE_u8(u8 *restrict dst, u8 *restrict src, u64 n);
void memcpy_SSE_u8_nt(u8 *restrict dst, u8 *restrict src, u64 n);

void memcpy_AVX_u1(u8 *restrict dst, u8 *restrict src, u64 n);
void memcpy_AVX_u2(u8 *restrict dst, u8 *restrict src, u64 n);
void memcpy_AVX_u4(u8 *restrict dst, u8 *restrict src, u64 n);
void memcpy_AVX_u8(u8 *restrict dst, u8 *restrict src, u64 n);
void memcpy_AVX_u8_nt(u8 *restrict dst, u8 *restrict src, u64 n);
void memcpy_AVX_u16(u8 *restrict dst, u8 *restrict src, u64 n);
void memcpy_AVX_u16_nt(u8 *restrict dst, u8 *restrict src, u64 n);

void memcpy_AVX512_u1(u8 *restrict dst, u8 *restrict src, u64 n);
void memcpy_AVX512_u2(u8 *restrict dst, u8 *restrict src, u64 n);
void memcpy_AVX512_u4(u8 *restrict dst, u8 *restrict src, u64 n);
void memcpy_AVX512_u8(u8 *restrict dst, u8 *restrict src, u64 n);
void memcpy_AVX512_u8_nt(u8 *restrict dst, u8 *restrict src, u64 n);
void memcpy_AVX512_u16(u8 *restrict dst, u8 *restrict src, u64 n);
void memcpy_AVX512_u16_nt(u8 *restrict dst, u8 *restrict src, u64 n);
