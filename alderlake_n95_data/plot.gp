set term png size 1900,1000 noenhanced font "Terminal,10"

set grid

set datafile separator ";"

set style data histogram
set style fill solid border -1
set boxwidth 0.9

set auto x
set xtic rotate by -45 scale 0

set xlabel "Benchmark"
set ylabel "GiB/s"

set multiplot layout 1, 1 rowsfirst

set title "AlderLake N95"
list=system('ls -v *.csv')
plot for [csv in list] csv u 11:xticlabels(stringcolumn(1)) t split(csv, ".")[1]." MiB"

undef multiplot
