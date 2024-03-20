## Scratch model
The code to create the scratch model is in [FEMModel](ScratchTest/FEMModel).
-Run [ScratchVPL-240320.py](ScratchTest/FEMModel/ScratchVPL-240320.py) through abaqus 2020 to get the scratch model, export the inp file of the model.
Run [VPL-Generate-Inp-240320.py](ScratchTest/FEMModel/VPL-Generate-Inp-240320.py) through pycharm 2023 to get the batch calculation inp file, 
put [Submit_inp](ScratchTest/FEMModel/Submit_inp) and inp file in the same folder, and the dual-machine bat file can be submitted for calculation. 
The calculated odb file can be extracted by running [VPL-GetData-240320.py](ScratchTest/FEMModel/VPL-GetData-240320.py) through abaqus 2020. 
The limited metadata in this paper is obtained from this.

## Data

The finite element simulation data is in [FEMData](ScratchTest/FEMData).
[FEM12.csv](ScratchTest/FEMData/FEM12.csv) is 12 sets of data for solving parameters, corresponding to Table 1.
[FEM50.csv](ScratchTest/FEMData/FEM50.csv) is the 50 sets of data verified, corresponding to Table 2.
The experimental data is in [TestData](ScratchTest/TestData)
[RawData](ScratchTest/TestData/RawData) contains data for [Tensile](ScratchTest/TestData/RawData/Tensile),[ScratchForce](ScratchTest/TestData/RawData/ScratchForce) and [ScratchProfile](ScratchTest/TestData/RawData/ScratchProfile)
Plug [TestData.csv](ScratchTest/TestData/TestData.csv) into [Solve_FEM_ns.py](ScratchTest/FEMData/Solve_FEM_ns.py) run, you can get [TestSolve.csv](ScratchTest/TestData/TestSolve.csv)


## Code

Code 1 is used to determine the parameters
Code 2 is forward verified by finite metadata
Code 3 is solved in reverse using finite metadata
Code 4 is a comparison of normalized scratch hardness
Code 5 is a comparison of scratch elastic recovery

All the code is in the folder [src](src). The code depends on the deep learning package [DeepXDE](https://github.com/lululxvi/deepxde) v1.1.2. If you use DeepXDE>1.1.2, you need to set `standardize=True` in `dde.data.MfDataSet()`.

- [data.py](src/data.py): The classes are used to read the data file. Remember to uncomment certain line in `ExpData` to scale `dP/dh`.
- [nn.py](src/nn.py): The main functions of multi-fidelity neural networks.
- [model.py](src/model.py): The fitting function method. Some parameters are hard-coded in the code, and you should modify them for different cases.
- [fit_n.py](src/fit_n.py): Fit strain-hardening exponent.
- [mfgp.py](src/mfgp.py): Multi-fidelity Gaussian process regression.
- 


## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.


## Data

All the data is in the folder [data](data).

## Code

All the code is in the folder [src](src). The code depends on the deep learning package [DeepXDE](https://github.com/lululxvi/deepxde) v1.1.2. If you use DeepXDE>1.1.2, you need to set `standardize=True` in `dde.data.MfDataSet()`.

- [data.py](src/data.py): The classes are used to read the data file. Remember to uncomment certain line in `ExpData` to scale `dP/dh`.
- [nn.py](src/nn.py): The main functions of multi-fidelity neural networks.
- [model.py](src/model.py): The fitting function method. Some parameters are hard-coded in the code, and you should modify them for different cases.
- [fit_n.py](src/fit_n.py): Fit strain-hardening exponent.
- [mfgp.py](src/mfgp.py): Multi-fidelity Gaussian process regression.

## Cite this work

If you use this code for academic research, you are encouraged to cite the following paper:

```
@article{Lu7052,
  author  = {Lu, Lu and Dao, Ming and Kumar, Punit and Ramamurty, Upadrasta and Karniadakis, George Em and Suresh, Subra},
  title   = {Extraction of mechanical properties of materials through deep learning from instrumented indentation},
  volume  = {117},
  number  = {13},
  pages   = {7052--7062},
  year    = {2020},
  doi     = {10.1073/pnas.1922210117},
  journal = {Proceedings of the National Academy of Sciences}
}
```

## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
