# Example problems and Jupyter Notebooks for the parallel MODFLOW 6 class

This repository includes example problems and Jupyter Notebooks that will be used in the parallel MODFLOW class.


### Running the examples locally

With the parallel version of MODFLOW 6 (`mf6p`) in your path, open a terminal in an example directory (for example, `par_gwf01-2d`) and run using `mpiexec`.

1. _par_gwf01-1d_

        mpiexec -n 2 mf6p -p

    This example can be run on 1 or 2 processors.

2. _par_gwf01-2d_

        mpiexec -n 2 mf6p -p

    This example can be run on 1 or 2 processors.

3. _par_gwf01-3d_

        mpiexec -n 2 mf6p -p

    This example can be run on 1 or 2 processors.
