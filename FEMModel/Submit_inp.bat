@echo off
pushd %~n1
for %%i in (*.inp) do (
cmd/c abaqus job=%%~ni cpus=4 double int  ask_delete=OFF
) 