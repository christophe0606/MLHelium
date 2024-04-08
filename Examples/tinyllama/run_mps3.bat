@ECHO OFF

pyocd load --target cortex_m -u L85986697A cprj\out\tinyllama\MPS3_Corstone-300\Release\tinyllama.axf
pyocd reset --target cortex_m -u L85986697A
