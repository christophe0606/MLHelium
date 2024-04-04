@ECHO OFF

pyocd load --target cortex_m -u L85986697A cprj\out\simple\MPS3_Corstone-300\Release\simple.axf
pyocd reset --target cortex_m -u L85986697A
