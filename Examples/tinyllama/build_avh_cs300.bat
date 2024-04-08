@ECHO OFF
REM cbuild -O cprj tinyllama.csolution.yml --toolchain GCC -c tinyllama.Release+VHT-Corstone-300 -r
REM cbuild -O cprj tinyllama.csolution.yml --toolchain CLANG -c tinyllama.Release+VHT-Corstone-300 

REM cbuild -r --update-rte -O cprj tinyllama.csolution.yml --toolchain AC6 -c tinyllama.Release+AVH_Corstone-300
cbuild -O cprj tinyllama.csolution.yml --toolchain AC6 -c tinyllama.Release+AVH_Corstone-300

REM dump_vht_m55.bat

