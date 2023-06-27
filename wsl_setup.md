# WSL and Parallel MODFLOW

Instructions are provided here for setting up Ubuntu on the Windows Subsystem for Linux (WSL) to work with MODFLOW.

## Installing WSL
On a Windows machine it is relatively easy to get parallel MODFLOW compiled and running in a WSL Ubuntu virtual machine.

Install a latest version of Ubuntu.
```
  wsl --install -d Ubuntu-22.04
```

You will be asked to provide a username and password.  You'll need to remember this information for some future `sudo` operations.

## Firewall Considerations

If you are on a networked computer behind a firewall, you may need to take additional steps at this point to configure WSL to handle SSL intercept requirements.

## Installing Miniconda

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b
eval "$(/home/langevin/miniconda3/bin/conda shell.bash hook)"
conda init
```

At this point, you will need to restart the shell to allow the miniconda installation to complete.

## Cloning MODFLOW Resources from GitHub
The next step is to clone the GitHub repositories for MODFLOW 6 and the MODFLOW 6 parallel class.

```
git clone https://github.com/jdhughes-usgs/parallel-modflow6-class.git
git clone https://github.com/MODFLOW-USGS/modflow6.git
```

## Installing the `mf6pro` Conda Environment
Next, you will create a conda environment, called `mf6pro`, that will be used for thie class.  The `mf6pro` environment will have all of the software needed to compile serial and parallel versions of MODFLOW 6, and the Python packages needed to pre- and post-process MODFLOW models.

The steps for creating the `mf6pro` conda environment are as follows

```
conda env create -f ./parallel-modflow6-class/environment/flopy_environment.yml -f ./parallel-modflow6-class/environment/mf6_environment.yml
```

In order to run jupyter notebooks, it will also be necessary to install jupyter.  Activate the `mf6pro` conda environment and then install jupyter from the conda-forge channel, as shown with the following commands.

```
conda activate mf6pro
conda install -c conda-forge jupyter jupyterlab
```

## Building MODFLOW

```
cd modflow6
meson setup builddir -Ddebug=false -Dparallel=true --prefix=$(pwd) --libdir=bin
meson install -C builddir
meson test --verbose --no-rebuild -C builddir
```

To make this new MODFLOW 6 executable available for future simulations, add a symbolic link to the newly compiled version of mf6 (./bin/mf6) from the mf6pro bin folder (~/miniconda3/envs/mf6pro/bin).

```
ln ./bin/mf6 ~/miniconda3/envs/mf6pro/bin/mf6
```