@ECHO OFF
REM cbuild -O cprj simple.csolution.yml -r --toolchain GCC -c simple.Release+MPS3-Corstone-300 
REM cbuild -O cprj simple.csolution.yml -r --toolchain CLANG -c simple.Release+MPS3-Corstone-300 

REM cbuild -O cprj -r --update-rte simple.csolution.yml --toolchain AC6 -c simple.Release+MPS3_Corstone-300 
cbuild -O cprj -r simple.csolution.yml --toolchain AC6 -c simple.Release+MPS3_Corstone-300 

