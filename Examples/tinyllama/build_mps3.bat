@ECHO OFF
REM cbuild -O cprj tinyllama.csolution.yml -r --toolchain GCC -c tinyllama.Release+MPS3-Corstone-300 
REM cbuild -O cprj tinyllama.csolution.yml -r --toolchain CLANG -c tinyllama.Release+MPS3-Corstone-300 

REM cbuild -O cprj -r --update-rte tinyllama.csolution.yml --toolchain AC6 -c tinyllama.Release+MPS3_Corstone-300 
cbuild -O cprj tinyllama.csolution.yml --toolchain AC6 -c tinyllama.Release+MPS3_Corstone-300 

