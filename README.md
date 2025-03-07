## Quick Start
As a user, clone the repo over https using:
```
git clone --recursive https://github.com/LukeV37/Pileup_Project.git
```

As a developer, clone the repo over ssh using:
```
git clone --recursive git@github.com:LukeV37/Pileup_Project.git
```

Ensure dependencies are installed (see below), and build the submodules:
```
./build_submodules.sh
```

Please be patient while submodules build...

See `software/README.md` for more information regarding submodules.


## How To Generate Datasets

First, generate events in MadGraph:
```
cd madgraph
# Generate Signal
./run_dihiggs.sh
# Generate Background
./run_4b.sh
```
For more information on pythia, please see `madgraph/README.md`.

Then, shower in Pythia8:
```
cd pythia
./generate_diHiggs.sh {diHiggs|4b} {mu} {String Modifier} {MinJetpT}
```
The root file generated by the pythia simulation will be found in `pythia/output`. 

For more information on pythia, please see `pythia/README.md`.

## Folder structure

- `madgraph/`: Generate Events in LHE format using MadGraph5
  - `run_dihiggs.sh`: Generate signal process
  - `run_4b.sh`: Generate background process 
- `pythia/`: Shower Events to ROOT files using Pythia8
  - `generate_diHiggs.sh`
- `models/`: Train a model to predict Efrac and Mfrac of jets
  - `run_preprocessing.sh`: Convert ROOT file to Pytorch Tensors and perform basic cuts
  - `train_model.sh`: Train the model on the preprocessed data to predict Efrac and Mfrac
- `analysis/`: Distinguish between diHiggs and 4b using Efrac and Mfrac
  - `run_preprocessing.sh`: Convert ROOT file to Pytorch Tensors and perform basic cuts
  - `train_model.sh`: Train the model on the preprocessed data to distinguish between diHiggs and 4b
- `software/`: Git references to external packages
  - `install_scripts/`: Folder containing scripts to auto-install

## Dependencies
Tested on Ubuntu 22.04, EL9 lxplus machine, and even WSL2.

Required Dependencies:
<ul>
  <li>python3</li>
  <li>ROOTv6</li>
  <li>autoconf</li>
  <li>libtool</li>
  <li>automake</li>
  <li>tcl</li>
  <li>gfortran</li>
  <li>g++</li>
</ul>

ROOTv6 can be installed following instructions [here](https://root.cern/install/).

Other packages can be intalled with `sudo apt install {package}`

## Misc.

>[!WARNING]
> [ROOT](https://root.cern/install/) must be installed on the system before `./install.sh` script can be run. \
> Use `root-config --cflags --libs` to see if you have a successful ROOT install.

>[!NOTE]
> On GPURIG2, please add the following lines to your `.bashrc` file in your home directory:
> ```bash
> export ROOTSYS=/usr/local/rootpy27/root_install
> export PATH=$ROOTSYS/bin:$PATH
> ```
