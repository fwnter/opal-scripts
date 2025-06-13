
../opal-scripts/experiment-scripts/ex1_pur.dacapo.sh | tee ../jfr/dacapo/ex1_purity.log | sed -n -E -f ../opal-scripts/filter.sed > ../jfr/dacapo/ex1_purity.sanitized.log
../opal-scripts/experiment-scripts/ex2_im.dacapo.sh | tee ../jfr/dacapo/experiment2_immutability.log | sed -n -E -f ../opal-scripts/filter.sed > ../jfr/dacapo/experiment2_immutability.sanitized.log