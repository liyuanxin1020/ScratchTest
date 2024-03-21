## Scratch model  
The code to create the scratch model is in [FEMModel](FEMModel)  
Run [ScratchVPL-240320.py](ScratchTest/FEMModel/ScratchVPL-240320.py) through abaqus 2020 to get the scratch model  
Export the inp file of the model,run [VPL-Generate-Inp-240320.py](ScratchTest/FEMModel/VPL-Generate-Inp-240320.py) through pycharm 2023 to get the batch calculation inp file  
Put [Submit_inp.bat](ScratchTest/FEMModel/Submit_inp.bat) and inp file in the same folder, and the dual-machine bat file can be submitted for calculation   
The calculated odb file can be extracted by running [VPL-GetData-240320.py](ScratchTest/FEMModel/VPL-GetData-240320.py) through abaqus 2020   
The limited metadata in this paper is obtained from this

## Data

The finite element simulation data is in [FEMData](ScratchTest/FEMData)  
[FEM12.csv](ScratchTest/FEMData/FEM12.csv) is 12 sets of data for solving parameters, corresponding to Table 1  
[FEM50.csv](ScratchTest/FEMData/FEM50.csv) is the 50 sets of data verified, corresponding to Table 2  
The experimental data is in [TestData](ScratchTest/TestData)  
[RawData](ScratchTest/TestData/RawData) contains data for [Tensile](ScratchTest/TestData/RawData/Tensile),[ScratchForce](ScratchTest/TestData/RawData/ScratchForce) and [ScratchProfile](ScratchTest/TestData/RawData/ScratchProfile)  
Plug [TestData.csv](ScratchTest/TestData/TestData.csv) into [Solve_FEM_ns.py](ScratchTest/FEMData/Solve_FEM_ns.py) run, you can get [TestSolve.csv](ScratchTest/TestData/TestSolve.csv)  


## Code

[Find_knt.py](ScratchTest/FEMData/Find_knt.py) is used to determine the parameters
[Verify_FEM_knt.py](ScratchTest/FEMData/Verify_FEM_knt.py) is forward verified by finite data
[Solve_FEM_ns.py](ScratchTest/FEMData/Solve_FEM_ns.py) is solved in reverse using finite data
[Verify_FEM_Hs.py](ScratchTest/FEMData/Verify_FEM_Hs.py) is a comparison of normalized scratch hardness
[Verify_FEM_He.py](ScratchTest/FEMData/Verify_FEM_He.py) is a comparison of scratch elastic recovery




## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.

