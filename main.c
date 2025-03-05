#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "types.h"
#include "utils.h"
#include "memcpy.h"

#define NB_SAMPLES 33

ascii *memcpy_functions_names[] = {
  
  "memcpy_C",
  "memcpy_openmp",
  "memcpy_memcpy",

  "memcpy_asm",
  
  "memcpy_SSE_u1",
  "memcpy_SSE_u2",
  "memcpy_SSE_u4",
  "memcpy_SSE_u8",
  "memcpy_SSE_u8_nt",
  
  "memcpy_AVX_u1",
  "memcpy_AVX_u2",
  "memcpy_AVX_u4",
  "memcpy_AVX_u8",
  "memcpy_AVX_u8_nt",
  "memcpy_AVX_u16",
  "memcpy_AVX_u16_nt",
  
#if defined(__AVX512F__)

  "memcpy_AVX512_u1",
  "memcpy_AVX512_u2",
  "memcpy_AVX512_u4",
  "memcpy_AVX512_u8",
  "memcpy_AVX512_u8_nt",
  
#endif
  
  NULL };

void (*memcpy_functions_list[])(u8 *restrict, u8 *restrict, u64) = {

  memcpy_C,
  memcpy_openmp,
  memcpy_memcpy,

  memcpy_asm,
  
  memcpy_SSE_u1,
  memcpy_SSE_u2,
  memcpy_SSE_u4,
  memcpy_SSE_u8,
  memcpy_SSE_u8_nt,
    
  memcpy_AVX_u1,
  memcpy_AVX_u2,
  memcpy_AVX_u4,
  memcpy_AVX_u8,
  memcpy_AVX_u8_nt,
  memcpy_AVX_u16,
  memcpy_AVX_u16_nt,
    
#if defined(__AVX512F__)
  
  memcpy_AVX512_u1,
  memcpy_AVX512_u2,
  memcpy_AVX512_u4,
  memcpy_AVX512_u8,
  memcpy_AVX512_u8_nt,
  
#endif
  
  NULL };
  
int main(int argc, char **argv)
{
  if (argc < 3)
    return printf("Usage: %s [size in bytes] [repetitions]\n", argv[0]), -1;
  
  //Number of array bytes
  u64 n = atoll(argv[1]);
  u64 r = atoll(argv[2]);
  
  //Total size in KiB (2 arrays)
  f64 s_kib = (f64)(n << 1) / (1024.0);

  //Total size in MiB (2 arrays)
  f64 s_mib = (f64)(n << 1) / (1024.0 * 1024.0);

  //Total size in GiB (2 arrays)
  f64 s_gib = (f64)(n << 1) / (1024.0 * 1024.0 * 1024);
  
  //
  u8 *restrict src = NULL;
  u8 *restrict dst = NULL;
  
  //
  u64 i = 0;
  void (*memcpy_ptr)(u8 *restrict, u8 *restrict, u64) = memcpy_functions_list[0];
  
  //
  f64 t1      = 0.0;
  f64 t2      = 0.0;
  f64 elapsed = 0.0;
  f64 samples[NB_SAMPLES];
  
  //
  u8 compare_state = 0;

#if defined(__AVX512F__)
  
  printf("Info: AVX512F detected! AVX512 benchmarks were added.\n");
  
#endif
  
  //Header
  printf("%30s; %16s; %16s; %6s; %16s; %16s; %16s; %16s; %16s; %16s; %16s\n",
	 "title",
	 "KiB",
	 "MiB",
	 "check",
	 "min (us)",
	 "max (us)",
	 "med (us)",
	 "mean (us)",
	 "CV",
	 "MRR",
	 "MiB/s");
  
  //Go through all implementations
  while (memcpy_ptr)
    {
      for (u64 j = 0; j < NB_SAMPLES; j++)
	{
	  //Allocate arrays
	  dst = aligned_alloc(64, n);
	  src = aligned_alloc(64, n);
      
	  //Initialize arrays (first touch)
	  memset(dst, 0, n);
	  memset(src, 1, n);
      
	  //Repeat the measurement if it fails
	  do
	    {
	      t1 = omp_get_wtime();
	  
	      //Ensure kernel measurability through repetition
	      for (u64 i = 0; i < r; i++)
		memcpy_ptr(dst, src, n);
	  
	      t2 = omp_get_wtime();
	  
	      //
	      elapsed = (t2 - t1) / (f64)r;
	    }
	  while (elapsed <= 0.0);
	  
	  samples[j] = elapsed;

	  //Check if the copy operation is performed properly
	  compare_state = memcmp(dst, src, n);
	  
	  free(src);
	  free(dst);
	}

      sort(samples, NB_SAMPLES);
      
      f64 min    = samples[0];
      f64 max    = samples[NB_SAMPLES - 1];
      f64 med    = samples[(NB_SAMPLES + 1) >> 1];
      f64 mean   = compute_mean(samples, NB_SAMPLES);
      f64 stddev = compute_stddev(samples, mean, NB_SAMPLES);

      //Stability 
      f64 cv     = (stddev * 100.0) / mean; //Coefficient of variation
      f64 mrr    = (max - min) / min; //Minimum relative range
      
      //GiB/s
      f64 bw = (s_gib) / (min);

      printf("%30s; %16.3lf; %16.3lf; %6s; %16.3lf; %16.3lf; %16.3lf; %16.3lf; %16.3lf; %16.3lf; %16.3lf\n",
	     memcpy_functions_names[i],
	     s_kib,
	     s_mib,
	     (compare_state) ? "FAIL" : "PASS",
	     min  * 1e6,
	     max  * 1e6,
	     med  * 1e6,
	     mean * 1e6,
	     cv,
	     mrr,
	     bw);
      
      //Move to the next kernel implementation to benchmark
      memcpy_ptr = memcpy_functions_list[++i];
    }
  
  //
  return 0;
}
