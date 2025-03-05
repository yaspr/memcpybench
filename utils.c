//
#include <math.h>

//
#include "types.h"

//
void sort(f64 *restrict a, u64 n)
{
  for (u64 i = 0; i < n; i++)
    for (u64 j = i + 1; j < n; j++)
      if (a[i] > a[j])
	{
	  f64 tmp = a[i];

	  a[i] = a[j];
	  a[j] = tmp;
	}
}

//
f64 compute_mean(f64 *restrict a, u64 n)
{
  f64 m = 0.0;

  for (u64 i = 0; i < n; i++)
    m += a[i];

  return m / (f64)n;
}

//
f64 compute_stddev(f64 *restrict a, f64 m, u64 n)
{
  f64 d = 0.0;

  for (u64 i = 0; i < n; i++)
    d += (a[i] - m) * (a[i] - m);

  d /= (f64)(n - 1);

  return sqrt(d);
}

