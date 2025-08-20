# SBEL GPU DEM-Engine
__A dual-GPU DEM solver with complex grain geometry support__

### 🚨 DEM-Engine now works on both Linux (including WSL) and Windows! 🚨

**⚠️ Windows users are still encouraged to use it via WSL for a more well-tested experience. ⚠️**

<p>
  <img width="380" src="https://i.imgur.com/DKGlM14.jpg">
  <img width="380" src="https://i.imgur.com/A3utANi.gif">
</p>

<p>
  <img width="380" src="https://i.imgur.com/YOEbAd8.gif">
  <img width="380" src="https://i.imgur.com/4R25TPX.gif">
</p>

## Quick links

<li><a href="#description">Overview, movies of demos, and where to get help</a></li>

<li><a href="#pyDEME">Use DEME with Python</a></li>

<li><a href="#compilation">How to compile from source</a></li>

<li><a href="#examples">Numerical examples and use cases</a></li>

<!-- <li><a href="#ccontainer">Container</a></li> -->

<li><a href="#install-as-library">Install as C++ library</a></li>

<li><a href="#postprocessing">Postprocessing recommendation</a></li>

<li><a href="#licensing">Licensing</a></li>

<li><a href="#limitations">Limitations</a></li>

<li><a href="#citation">Cite this work</a></li>


<h2 id="description">Description</h2>

DEM-Engine, nicknamed _DEME_, does Discrete Element Method simulations:

- Using up to two GPUs at the same time (works great on consumer _and_ data center GPUs).
- With the particles having complex shapes represented by clumped spheres.
- With support for customizable contact force models (want to add a non-standard cohesive force, or an electrostatic repulsive force? You got this).
- With an emphasis on computational efficiency. As a rule of thumb, using 3-sphere clump elements, simulating 1 million elements for 1 million time steps takes around 1 hour on two RTX 3080s.
- Supporting a wide range of problems with flexible API designs. Deformable meshes and grain breakage can be simulated by leveraging the explicit controls given to the user.
- With support for co-simulation with other C/C++ packages, such as [Chrono](https://github.com/projectchrono/chrono).

Currently _DEME_ is a C++ package with an API design similar to Chrono's, and should be easy to learn for existing Chrono users.

You can find the movies of some of _DEME_'s demos [here](https://uwmadison.app.box.com/s/u4m9tee3k1vizf097zkq3rgv54orphyv).

You are welcome to discuss _DEME_ on [Project Chrono's forum](https://groups.google.com/g/projectchrono). 

<h2 id="pyDEME">PyDEME</h2>

_DEME_ is now available as a Python package, _pyDEME_. It is quick to install and pick up its usage by trying this Python version. If you want to maximize the performance and use the cutting-edge features, you can instead <a href="#compilation">install the C++ version of _DEME_ from source</a>.

To install _pyDEME_, use a Linux machine, install CUDA if you do not already have it. Useful installation instructions may be found [here](https://developer.nvidia.com/cuda-downloads). 

Some additional troubleshooting tips for getting CUDA ready:

- I recommend getting the newest CUDA. But note that the recent releases CUDA 12.1, 12.2 and 12.3 appear to cause troubles with jitify and you should not use them with DEME.

Once CUDA is ready, you can `pip` install _pyDEME_. In your conda environement, do
```
conda create -n pyDEME python=3.11
conda activate pyDEME
conda install cmake
pip3 install DEME
```

~~You can also install pyDEME via `conda install`:~~ (Please don't use `conda install` for now, it is not yet behaving correctly)

~~`conda create -n pyDEME python=3.11`~~

~~`conda activate pyDEME`~~

~~`conda install -c projectchrono pydeme`~~

`pyDEME` can be replaced with an environement name of your choice. Other Python versions other than 3.11 should work as well.

Then [Python scripts](https://github.com/projectchrono/DEM-Engine/tree/pyDEME_demo/src/demo) can be executed in this environment. To understand the content of each Python demo, refer to the explanations of the C++ demos with the same names in <a href="#examples">Numerical examples</a> section.

If you use _pyDEME_ in conjunction with PyChrono, you should `import pyDEME` first, then PyChrono.

<h2 id="compilation">Compilation</h2>

You can also build C++ _DEME_ from source. It allows for potentially more performance and more tailoring.

### Linux and WSL

First, [install CUDA](https://developer.nvidia.com/cuda-downloads). The newest release is recommended.

Once CUDA is ready, clone this project and then:

```
git submodule init
git submodule update
```

This will pull the submodule NVIDIA/jitify so that we can do runtime compilation. 

Then, one typical choice is to make a build directory in it. Then in the build directory, use `cmake` to configure the compilation. An example:

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
```

You may want to use [this information](https://askubuntu.com/questions/1203635/installing-latest-cmake-on-ubuntu-18-04-3-lts-run-via-wsl-openssl-error) if you need to update cmake to the newest. 

We suggest that you install a `cmake` GUI such as `ccmake`, and `ninja_build` generator, to better help you configure the project. In this case, the example above can be done like this alternatively:

```
mkdir build
cd build
ccmake -G Ninja ..
```

You generally do not have to change the build options in the GUI, but preferably you can change `CMAKE_BUILD_TYPE` to `Release`, and if you need to install this package as a library you can specify a `CMAKE_INSTALL_PREFIX`. 

Some additional troubleshooting tips for generating the project:

- If some dependencies such as CUB are not found, then you probably need to manually set `$PATH` and `$LD_LIBRARY_PATH`. An example is given below for a specific version of CUDA, note it may be different on your machine or cluster. You should also inspect if `nvidia-smi` and `nvcc --version` give correct returns.
```
export CPATH=/usr/local/cuda-12.8/targets/x86_64-linux/include${CPATH:+:${CPATH}}
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}
export PATH=/usr/local/cuda-12.8/lib64/cmake${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-12.8
```

Finally, build the project.

```
ninja
```

Some additional troubleshooting tips for building the project:

- If you see some grammatical errors during compilation, such as `filesystem` not being a member of `std` or arguments not expanded with `...`, then manually setting the flag `TargetCXXStandard` to `STD_CXX17` might help.
- If CUB is not found, then you may manually set it in the `ccmake` GUI as `/usr/local/cuda/lib64/cmake/cub`. It may be a slightly different path on your machine or cluster.
- If `libcudacxx` is not found, then you may manually set it in the `ccmake` GUI as `/usr/local/cuda-12.8/targets/x86_64-linux/lib/cmake/libcudacxx`. Depending on your CUDA version it may be a slightly different path on your machine or cluster. You may also try to find these packages using `find`. 

### Windows

The process is similar to [the installation of Chrono](https://api.projectchrono.org/tutorial_install_chrono.html), which you can use as reference. The steps depend on your choice of tools, and what listed here are our recommendation.

- Follow the guide in the **Linux and WSL** section to install CUDA on Windows.
- Use [Sourcetree](https://www.sourcetreeapp.com/) to clone the repo's main branch, with the _recursive_ option on.
- Use [CMake GUI](https://cmake.org/download/) to configure the project. You should get all the dependency fields filled automatically; if not, manually fill them. Example CUB directory: `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/cmake/cub`. Use the default `x64` architecture and `MSVC` compiler. If you need to install this package as a library later, you can specify a `CMAKE_INSTALL_PREFIX`.
- After configuration and generation, find the file `Chrono-DEM-Engine.sln` in the build directory, open it using Visual Studio, select `Release` build type and build the project.

<h2 id="examples">Numerical examples</h2>

After the build process is done, you can start trying out the demos. On Windows, these demos may be in the `bin\Release` directory instead.

- `./bin/DEMdemo_SingleSphereCollide` can be used to test a correct installation. If it runs outputting a lot of texts (those are debug messages; the user do not have to worry about the content) and stops without an error in the end, the installation is probably good.
- An all-rounder beginner example featuring a bladed mixer interacting with complex shaped particles: `./bin/DEMdemo_Mixer`.
- A place to learn how prescribed motions work in this package, using either analytical boundaries or particle-represented boundaries: `./bin/DEMdemo_Centrifuge` and `./bin/DEMdemo_Sieve`.
- A few representative engineering experiments reproduced in DEM simulations, which potentially serve as starting points for your own DEM scripts: `/bin/DEMdemo_BallDrop`, `./bin/DEMdemo_ConePenetration`, `/bin/DEMdemo_RotatingDrum`, `./bin/DEMdemo_Repose`, `./bin/DEMdemo_Plow`.
- `./bin/DEMdemo_WheelDP` shows how to load a checkpointed configuration file to instantly generate a settled granular terrain, then run a drawbar-pull test on it. This demo therefore requires you to first finish the two GRCPrep demos to obtain the terrain checkpoint file. The granular terrain in these demos features DEM particles with a variety of sizes and shapes.
- `./bin/DEMdemo_WheelDPSimplified` is a simplified version of the previous drawbar-pull test which has no prerequisite. The terrain is simpilified to be made of only one type of irregular-shaped particles. It serves as a quick starting point for people who want to create similar experiments.
- `./bin/DEMdemo_Indentation` is a more advanced examples showing the usage of the custom additional properties (called _wildcards_) that you can associate with the simulation entities, and use them in the force model and/or change them in simulation then deposit them into the output files. _Wildcards_ have more use cases especially if coupled together with a custom force model, as shown in some of the follwing demos.
- `./bin/DEMdemo_Electrostatic` simulates a pile of complex-shaped and charged granular particles interacting with a mesh that is also charged. Its purpose is to show how to define a non-local force (electrostatic force) which takes effect even when the bodies are not in contact, using a custom force model file. This idea can be extended to modeling a custom cohesion force etc.
- `./bin/DEMdemo_FlexibleMesh` simulates a deforming mesh interacting with DEM particles. The intention is to show that the user can extract the force pairs acting on a mesh, then update the mesh with deformation information. _DEME_ does not care how this deformation is calculated. Presumably the user can feed the forces to their own solid mechanics solver to get the deformation. _DEME_ does not come with a built-in linear solver so for simplicity, in this demo the mesh deformation is instead prescribed.
- `./bin/DEMdemo_GameOfLife` is a fun game-of-life simulator built with the package, showing the flexibility in terms of how you can use this tool.
- `./bin/DEMdemo_Fracture_Box` simulates a concrete bar breaking using a custom force model that creates inter-particle bonds and lets them break under certain conditions. This is a showcase for advanced usage of custom models that involves per-contact wildcard variables.
- It is a good idea to read the comment lines at the top of the demo files to understand what they each does.

[The documentations for _DEME_](https://api.projectchrono.org/) are hosted on Chrono website (work in progress).

Some additional troubleshooting tips for running the demos:

- If errors similar to `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` are encountered while you run the demos, or (rarely) the simulations proceed without detecting any contacts, then please make sure the CUDA installation is the same version as when the code is compiled.
- Another cause for the simulations proceeding without detecting any contacts, could be the force kernel silently failed. This could also lead to a **too-many-geometries-in-bin** crash. The cause is usually that the force kernel was launched with too many threads per block, therefore not enough registers can be leveraged. This can be avoided by calling `SetForceCalcThreadsPerBlock` prior to the start of simulation with the argument being smaller choices like 128. **Note that you should only try this if you are using a custom force model**.
- Used your own force model but got runtime compilation error like `expression must have pointer-to-object type but it has type "float"`, or `unknown variable "delta_time"`? Check out what we did in demo `DEMdemo_Electrostatic`. You may need to manually specify what material properties are pairwise and what contact wildcards you have using `SetMustPairwiseMatProp` and `SetPerContactWildcards`.
- Just running provided demos or a script that used to work, but the jitification of the force model failed or the simulation fails at the first kernel call (probably in `DEMCubContactDetection.cu`)? Then did you pull a new version and just re-built in-place? A new update may modify the force model, and the force model in _DEME_ are given as text files so might not be automatically copied over when the project is re-built. I am sorry for the trouble it might cause, but you can do a clean re-build from an empty directory and it should fix the problem. Do not forget to first commit your own branches' changes and relocate the data you generated in the build directory. Another solution is to copy everything in `src/DEM` to the `DEM` directory in the build directory, then everything in `src/kernel` to the `kernel` directory in the build directory, then try again.

<!-- <h2 id="ccontainer">Using DEME in Container</h2>

_DEME_ is now [hosted on DockerHub](https://hub.docker.com/r/uwsbel/dem-engine) for those who want to run it in a container. It can potentially save your time that would otherwise be spent on getting the dependencies right, and for you to test out if _DEME_ is what you needed.

On a Linux machine, [install CUDA](https://developer.nvidia.com/cuda-downloads). Then install [docker](https://docs.docker.com/desktop/install/linux-install/) depending on your OS. Then [install Nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for running GPU-based containers. Things to note about installing the prerequisites:

- Got `Docker daemon permission denied` error? Maybe try [this page](https://stackoverflow.com/questions/48957195/how-to-fix-docker-got-permission-denied-issue).
- You can install the newest CUDA for running the container. However, I still recommend CUDA 12.0 or a CUDA 11 distro.

After those are done, you can launch the container by doing this in a Linux command prompt: `docker run -it --gpus all uwsbel/dem-engine:latest`

Then the source code along with a pre-built _DEME_ can be found in `/DEM-Engine` and `/DEM-Engine/build`. See **Examples** section for running example simulations. 

Starting from this point, you can start adding new scripts or modify existing ones for your own simulations. You can also build and run your newly added code, and commit the modified container as needed. If you encounter problems when re-building the project in the container, then you may refer to the troubleshooting tips in the **Installation** section for help, or turn to the [forum](https://groups.google.com/g/projectchrono).

Note that the container imagine is not updated as often for bug-fixes and new features as the GitHub repo.  -->

<h2 id="limitations">Limitations</h2>

_DEME_ is designed to simulate the interaction among clump-represented particles, the interaction between particles and mesh-represented bodies, as well as the interaction between particles and analytical boundaries. _DEME_ does not resolve mesh&ndash;mesh or mesh&ndash;analytical contacts.

- It is able to handle mesh-represented bodies with relatively simple physics, for example a meshed plow moving through granular materials with a prescribed velocity, or several meshed projectiles flying and hitting the granular ground. 
- However, if the bodies' physics are complex multibody problems, say it is a vehicle that has joint-connected parts and a motor with certain driving policies, or the meshed bodies have collisions among themselves that needs to be simulated, then _DEME_ alone does not have the infrastructure to handle them. But you can install _DEME_ as a library and do coupled simulations with other tools such as [Chrono](https://github.com/projectchrono/chrono), where _DEME_ is exclusively tasked with handling the granular materials and the influence they exert on the outside world (with high efficiency, of course). See the following section.

<h2 id="install-as-library">Install as library</h2>

### Linux and WSL

Set `CMAKE_INSTALL_PREFIX` flag in `cmake` GUI to your desired installation path and then 

```
ninja install
```

### Windows

Set `CMAKE_INSTALL_PREFIX` flag in `cmake` GUI to your desired installation path, then open a command line window in Visual Studio (may be found in Tools -> Command Line), and use the following command to install
```
cmake --install .
```

### Example scripts in Chrono-projects

We provide examples of linking against both [Chrono](https://github.com/projectchrono/chrono) and _DEME_ for co-simulations in [chrono-projects](https://github.com/projectchrono/chrono-projects/tree/feature/DEME). 

**For now, you need to checkout the `feature/DEME` branch of chrono-projects after cloning the code to access the scripts!**

You need to build `chrono-projects` linking against a Chrono installation (Chrono installation guide is [here](https://api.projectchrono.org/tutorial_install_chrono_linux.html); note you have to `make install` to install Chrono, not just build it), then link against _DEME_. The steps for building `chrono-projects`:

- Start by linking against Chrono. Set `Chrono_DIR`. It should be in `<your_Chrono_install_dir>/lib/cmake`. Then configure the project;
- Make sure `ENABLE_PROJECTS` to `ON` and configure the project;
- Linkage against Chrono is done, now move on to link against _DEME_. Set `ENABLE_DEME_TESTS` to `ON`. Then configure the project;
- Set `DEME_DIR` when prompted. It should be in `<your_DEME_install_dir>/lib64/cmake/DEME`. Then configure the project.
- You may see the `ChPF_DIR` option being prompted. If you did not build _DEME_ with USE_CHPF being ON, you can ignore this option. Otherwise CMake will refuse to generate, and you need to set this option to be `<your_DEME_install_dir>/lib64/cmake/ChPF`.

Then build the project and you should be able to run the demo scripts that demonstrate the co-simulation between _DEME_ and Chrono.

More documentations on using this package for co-simulations are being added.

<h2 id="postprocessing">Postprocessing recommendation</h2>

_DEME_ mainly outputs clumps or component spheres in `csv` format, so you can postprocess the raw data however you desire. The following is simply an example work flow which gives a quick and interactive rendering scene of the outputted component spheres (`WriteSphereFile`) using [ParaView](https://www.paraview.org/). 

- Directly load the outputted `vtk` mesh files in ParaView and show them.
- Load the `csv` component sphere files. Use filter **Table To Points**, and set X, Y and Z coordinates to be read from the _X_, _Y_ and _Z_ columns of the `csv` files, then click **Apply**. After that, apply **Glyph** filter. Set **Glyph Type** to be _Sphere_ (you may reduce **Theta** and **Phi Resolition** to save memory), then set the option **Scale Array** to be the _r_ column of the `csv` file, set **Scale Factor** to be 2 (_DEME_ outputs the radii but in ParaView we should scale the spheres by diameter). Finally, set **Glyph Mode** to _All Points_, then click **Apply**.

<h2 id="licensing">Licensing</h2>

This project should be treated as the collective intellectual property of the Author(s) and the University of Wisconsin - Madison. The following copyright statement should be included in any new or modified source files
```
Copyright (c) 2021, Simulation-Based Engineering Laboratory
Copyright (c) 2021, University of Wisconsin - Madison

SPDX-License-Identifier: BSD-3-Clause
```

New authors should add their name to the file `CONTRIBUTORS.md` rather than to individual copyright headers.

#### Notes on code included from Project Chrono

This project exists independently of Chrono so developers should be sure to include the appropriate BSD license header on any code which is sourced from Chrono::GPU(DEM) or other parts of Chrono.

> #### SAMPLE header for files sourced from Chrono

> ```
> Copyright (c) 2021, SBEL GPU Development Team
> Copyright (c) 2021, University of Wisconsin - Madison
> 
> SPDX-License-Identifier: BSD-3-Clause
> 
> 
> This file contains modifications of the code authored by the Project Chrono 
> Development Team. The original license can be found below:
>
> Copyright (c) 2016, Project Chrono Development Team
> All rights reserved.
> 
> Use of this source code is governed by a BSD-style license that can be found
> in the LICENSE file at the top level of the distribution and at
> http://projectchrono.org/license-chrono.txt. A copy of the license is below.
>
> Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
> 
>  - Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. 
>  - Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution. 
>  - Neither the name of the nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission. 
> 
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
> ```

<h2 id="citation">Citation</h2>

See [the paper that explains the design and usage of _DEME_](https://www.sciencedirect.com/science/article/pii/S001046552400119X?via%3Dihub) and cite
```bibtex
@article{zhang_2024_deme,
title = {Chrono {DEM-Engine}: A Discrete Element Method dual-{GPU} simulator with customizable contact forces and element shape},
journal = {Computer Physics Communications},
volume = {300},
pages = {109196},
year = {2024},
issn = {0010-4655},
doi = {https://doi.org/10.1016/j.cpc.2024.109196},
author = {Ruochun Zhang and Bonaventura Tagliafierro and Colin {Vanden Heuvel} and Shlok Sabarwal and Luning Bakke and Yulong Yue and Xin Wei and Radu Serban and Dan Negruţ},
keywords = {Discrete Element Method, GPU computing, Physics-based simulation, Scientific package, BSD3 open-source},
}
```

See [the paper on using _DEME_ for simulating rover dynamics](https://link.springer.com/article/10.1007/s00366-023-01921-9) and cite
```bibtex
@article{ruochunGRC-DEM2023,
      title={A {GPU}-accelerated simulator for the {DEM} analysis of granular systems composed of clump-shaped elements}, 
      author={Ruochun Zhang and Colin {Vanden Heuvel} and Alexander Schepelmann and Arno Rogg and Dimitrios Apostolopoulos and Samuel Chandler and Radu Serban and Dan Negrut},
      year={2024},
      journal={Engineering with Computers},
      doi={https://doi.org/10.1007/s00366-023-01921-9}
}
```
