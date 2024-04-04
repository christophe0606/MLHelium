@ECHO OFF
REM cbuild -O cprj simple.csolution.yml --toolchain GCC -c simple.Release+VHT-Corstone-300 -r
REM cbuild -O cprj simple.csolution.yml --toolchain CLANG -c simple.Release+VHT-Corstone-300 

REM cbuild -r --update-rte -O cprj simple.csolution.yml --toolchain AC6 -c simple.Release+AVH_Corstone-300
cbuild -O cprj simple.csolution.yml --toolchain AC6 -c simple.Release+AVH_Corstone-300

REM dump_vht_m55.bat

