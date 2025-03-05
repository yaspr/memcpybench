#pragma once

//
void sort(f64 *restrict a, u64 n);
f64 compute_mean(f64 *restrict a, u64 n);
f64 compute_stddev(f64 *restrict a, f64 m, u64 n);
