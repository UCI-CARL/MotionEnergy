<div align="center">
	<img src="http://socsci.uci.edu/~jkrichma/CARL-Logo-small.jpg" width="300"/>
</div>

# MotionEnergy

This is a CUDA implementation of Simoncelli and Heeger's Motion Energy model
([Simoncelli & Heeger, 1998](http://dx.doi.org/10.1016/S0042-6989(97)00183-1)).

The code comes with both a Python interface (in `pyME`) and a C/C++ interface (in `cppME`).

If you use this code in a scholarly publication, please cite as follows:
> Beyeler, M., Dutt, N., Krichmar, J.L. (2014).
> Efficient Spiking Neural Network Model of Pattern Motion Selectivity in Visual Cortex
> Neuroinformatics 12(3):435-454, [doi:10.1007/s12021-014-9220-y](http://dx.doi.org/10.1007/s12021-014-9220-y)

Or use the following BibTex:
```
@article{Beyeler2014,
	author = {M. Beyeler and N. Dutt and J. L. Krichmar},
	title = {Efficient Spiking Neural Network Model of Pattern Motion Selectivity in Visual Cortex},
	journal = {Neuroinformatics},
	year = {2014},
	volume = {12},
	number = {3},
	pages = {435--454},
	doi = {10.1007/s12021-014-9220-y}
}
```


## Installation

1. Fork MotionEnergy by clicking on the [`Fork`](https://github.com/UCI-CARL/MotionEnergy#fork-destination-box) box
   in the top-right corner.

2. Clone the repo, where `YourUsername` is your actual GitHub user name:
   ```
   $ git clone https://github.com/YourUsername/MotionEnergy
   $ cd MotionEnergy
   ```

3. Choose whether you want to use the Python interface or the C/C++ interface.
   - Python: There is no package install yet. See the file `pyME/run_dir_V1.py` for an example script.

   - C++: The installation depends on your platform.

     - Linux / Mac OS X:

       -# By default, MotionEnergy gets installed to `/opt/CARL/ME`.
          You can change this by exporting an environment variable called `ME_LIB_DIR`:
          ```
          $ export ME_LIB_DIR=/path/to/your/preferred/dir
          ```

       -# Then compile and install:
          ```
          $ cd cppME
          $ make
          $ sudo -E make install
          ```
          Note the `-E` flag, which will cause `sudo` to remember the `ME_LIB_DIR`.

     - Windows: Simply open the solution file `motion_energy.sln` in Visual Studio.
