# SBEL GPU DEM-Engine
__A dual-GPU DEM solver with complex grain geometry support__

<p>
  <img width="380" src="https://i.imgur.com/DKGlM14.jpg">
  <img width="380" src="https://i.imgur.com/c9DWwqk.gif">
</p>

<p>
  <img width="380" src="https://i.imgur.com/YOEbAd8.gif">
  <img width="380" src="https://i.imgur.com/4R25TPX.gif">
</p>

## Quick links

<li><a href="#description">Overview, movies of demos, and where to get help</a></li>

<li><a href="#pyDEME">Use DEME with Python</a></li>

<li><a href="#installation">How to compile from source</a></li>

<li><a href="#examples">Numerical examples and use cases</a></li>

<li><a href="#ccontainer">Container</a></li>

<li><a href="#library">Install as C++ library</a></li>

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

#### _pyDEME_ is BEING TESTED, many methods are not yet wrapped and the scripts may not work yet. For now it is recommended to <a href="#installation">install _DEME_ from source</a>.

_DEME_ is now available as a Python package, _pyDEME_.

To install _pyDEME_, use a Linux machine, install CUDA if you do not already have it. Useful installation instructions may be found [here](https://developer.nvidia.com/cuda-downloads). 

Some additional troubleshooting tips for getting CUDA ready:

- I recommend just getting CUDA 12.0, or a CUDA 11 distro. CUDA 12.1 and 12.2 appears to cause troubles with jitify.
- On WSL this code may be buildable (and [this](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) is the guide for installing CUDA on WSL), but may not run. This is due to the [many limitations on unified memory and pinned memory support](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications) on WSL. A native Linux machine or cluster is recommended.

Once CUDA is ready, you can `pip` install _pyDEME_. In your conda environement, do
```
conda create -n pyDEME python=3.11
conda activate pyDEME
conda install cmake
pip3 install DEME
```
~~You can also install pyDEME via `conda install`:~~ (Please don't use `conda install` for now, it is not yet behaving correctly)
```
conda create -n pyDEME python=3.11
conda activate pyDEME
conda install -c projectchrono pydeme
```
`pyDEME` can be replaced with an environement name of your choice. Other Python versions other than 3.11 should work as well.

Then [Python scripts](https://github.com/projectchrono/DEM-Engine/tree/pyDEME_demo/src/demo) can be executed in this environment. To understand the content of each Python demo, refer to the explanations of the C++ demos with the same names in **Examples** section.

<h2 id="installation">Compilation</h2>

You can also build C++ _DEME_ from source. It allows for potentially more performance and more tailoring.

On a Linux machine, [install CUDA](https://developer.nvidia.com/cuda-downloads). I recommend CUDA 12.0.

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

- For now, I suggest using CUDA version 12.0 or below. CUDA 12.1 does not seem to work well with the jitified kernels in _DEME_.
- If some dependencies such as CUB are not found, then you probably need to manually set `$PATH` and `$LD_LIBRARY_PATH`. An example is given below for a specific version of CUDA, note it may be different on your machine or cluster. You should also inspect if `nvidia-smi` and `nvcc --version` give correct returns.
```
export CPATH=/usr/local/cuda-12.0/targets/x86_64-linux/include${CPATH:+:${CPATH}}
export PATH=/usr/local/cuda-12.0/bin${PATH:+:${PATH}}
export PATH=/usr/local/cuda-12.0/lib64/cmake${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-12.0
```

Finally, build the project.

```
ninja
```

Some additional troubleshooting tips for building the project:

- If you see some grammatical errors during compilation, such as `filesystem` not being a member of `std` or arguments not expanded with `...`, then manually setting the flag `TargetCXXStandard` to `STD_CXX17` might help.
- If CUB is not found, then you may manually set it in the `ccmake` GUI as `/usr/local/cuda/lib64/cmake/cub`. It may be a slightly different path on your machine or cluster.
- If `libcudacxx` is not found, then you may manually set it in the `ccmake` GUI as `/usr/local/cuda-12.0/targets/x86_64-linux/lib/cmake/libcudacxx`. Depending on your CUDA version it may be a slightly different path on your machine or cluster. You may also try to find these packages using `find`.

<h2 id="examples">Examples</h2>

After the build process is done, you can start trying out the demos.

- `./src/demo/DEMdemo_SingleSphereCollide` can be used to test a correct installation. If it runs outputting a lot of texts (those are debug messages; the user do not have to worry about the content) and stops without an error in the end, the installation is probably good.
- An all-rounder beginner example featuring a bladed mixer interacting with complex shaped particles: `./src/demo/DEMdemo_Mixer`.
- A place to learn how prescribed motions work in this package, using either analytical boundaries or particle-represented boundaries: `./src/demo/DEMdemo_Centrifuge` and `./src/demo/DEMdemo_Sieve`.
- A few representative engineering experiments reproduced in DEM simulations, which potentially serve as starting points for your own DEM scripts: `/src/demo/DEMdemo_BallDrop`, `./src/demo/DEMdemo_ConePenetration`, `/src/demo/DEMdemo_RotatingDrum`, `./src/demo/DEMdemo_Repose`, `./src/demo/DEMdemo_Plow`.
- `./src/demo/DEMdemo_WheelDP` shows how to load a checkpointed configuration file to instantly generate a settled granular terrain, then run a drawbar-pull test on it. This demo therefore requires you to first finish the two GRCPrep demos to obtain the terrain checkpoint file. The granular terrain in these demos features DEM particles with a variety of sizes and shapes.
- `./src/demo/DEMdemo_WheelDPSimplified` is a simplified version of the previous drawbar-pull test which has no prerequisite. The terrain is simpilified to be made of only one type of irregular-shaped particles. It serves as a quick starting point for people who want to create similar experiments.
- `./src/demo/DEMdemo_Indentation` is a more advanced examples showing the usage of the custom additional properties (called _wildcards_) that you can associate with the simulation entities, and use them in the force model and/or change them in simulation then deposit them into the output files. _Wildcards_ have more use cases especially if coupled together with a custom force model, as shown in some of the follwing demos.
- `./src/demo/DEMdemo_Electrostatic` simulates a pile of complex-shaped and charged granular particles interacting with a mesh that is also charged. Its purpose is to show how to define a non-local force (electrostatic force) which takes effect even when the bodies are not in contact, using a custom force model file. This idea can be extended to modeling a custom cohesion force etc.
- `./src/demo/DEMdemo_FlexibleMesh` simulates a deforming mesh interacting with DEM particles. The intention is to show that the user can extract the force pairs acting on a mesh, then update the mesh with deformation information. _DEME_ does not care how this deformation is calculated. Presumably the user can feed the forces to their own solid mechanics solver to get the deformation. _DEME_ does not come with a built-in linear solver so for simplicity, in this demo the mesh deformation is instead prescribed.
- `./src/demo/DEMdemo_GameOfLife` is a fun game-of-life simulator built with the package, showing the flexibility in terms of how you can use this tool.
- `./src/demo/DEMdemo_SolarSystem` simulates our solar system. It is yet another fun simulation that is not strictly DEM per se, but shows how to define a mid-to-long-ranged force (gravitational force) using a custom force model file.
- It is a good idea to read the comment lines at the top of the demo files to understand what they each does.

[The documentations for _DEME_](https://api.projectchrono.org/) are hosted on Chrono website (work in progress).

Some additional troubleshooting tips for running the demos:

- If errors similar to `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` are encountered while you run the demos, or (rarely) the simulations proceed without detecting any contacts, then please make sure the CUDA installation is the same version as when the code is compiled.
- Another cause for the simulations proceeding without detecting any contacts, could be the force kernel silently failed. This is usually because it was launched with too many threads per block, therefore not enough registers can be leveraged. This can be avoided by calling `SetForceCalcThreadsPerBlock` prior to the start of simulation with the argument being 256 or even smaller choices like 128.
- Used your own force model but got runtime compilation error like `expression must have pointer-to-object type but it has type "float"`, or `unknown variable "delta_time"`? Check out what we did in demo `DEMdemo_Electrostatic`. You may need to manually specify what material properties are pairwise and what contact wildcards you have using `SetMustPairwiseMatProp` and `SetPerContactWildcards`.
- Just running provided demos or a script that used to work, but the jitification of the force model failed or the simulation fails at the first kernel call (probably in `DEMCubContactDetection.cu`)? Then did you pull a new version and just re-built in-place? A new update may modify the force model, and the force model in _DEME_ are given as text files so might not be automatically copied over when the project is re-built. I am sorry for the trouble it might cause, but you can do a clean re-build from an empty directory and it should fix the problem. Do not forget to first commit your own branches' changes and relocate the data you generated in the build directory. Another solution is to copy everything in `src/DEM` to the `DEM` directory in the build directory, then everything in `src/kernel` to the `kernel` directory in the build directory, then try again.

<h2 id="ccontainer">Using DEME in Container</h2>

_DEME_ is now [hosted on DockerHub](https://hub.docker.com/r/uwsbel/dem-engine) for those who want to run it in a container. It can potentially save your time that would otherwise be spent on getting the dependencies right, and for you to test out if _DEME_ is what you needed.

On a Linux machine, [install CUDA](https://developer.nvidia.com/cuda-downloads). Then install [docker](https://docs.docker.com/desktop/install/linux-install/) depending on your OS. Then [install Nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for running GPU-based containers. Things to note about installing the prerequisites:

- Got `Docker daemon permission denied` error? Maybe try [this page](https://stackoverflow.com/questions/48957195/how-to-fix-docker-got-permission-denied-issue).
- You can install the newest CUDA for running the container. However, I still recommend CUDA 12.0 or a CUDA 11 distro.

After those are done, you can launch the container by doing this in a Linux command prompt: `docker run -it --gpus all uwsbel/dem-engine:latest`

Then the source code along with a pre-built _DEME_ can be found in `/DEM-Engine` and `/DEM-Engine/build`. See **Examples** section for running example simulations. 

Starting from this point, you can start adding new scripts or modify existing ones for your own simulations. You can also build and run your newly added code, and commit the modified container as needed. If you encounter problems when re-building the project in the container, then you may refer to the troubleshooting tips in the **Installation** section for help, or turn to the [forum](https://groups.google.com/g/projectchrono).

Note that the container imagine is not updated as often for bug-fixes and new features as the GitHub repo. 

<h2 id="limitations">Limitations</h2>

_DEME_ is designed to simulate the interaction among clump-represented particles, the interaction between particles and mesh-represented bodies, as well as the interaction between particles and analytical boundaries. _DEME_ does not resolve mesh&ndash;mesh or mesh&ndash;analytical contacts.

- It is able to handle mesh-represented bodies with relatively simple physics, for example a meshed plow moving through granular materials with a prescribed velocity, or several meshed projectiles flying and hitting the granular ground. 
- However, if the bodies' physics are complex multibody problems, say it is a vehicle that has joint-connected parts and a motor with certain driving policies, or the meshed bodies have collisions among themselves that needs to be simulated, then _DEME_ alone does not have the infrastructure to handle them. But you can install _DEME_ as a library and do coupled simulations with other tools such as [Chrono](https://github.com/projectchrono/chrono), where _DEME_ is exclusively tasked with handling the granular materials and the influence they exert on the outside world (with high efficiency, of course). See the following section.

<h2 id="library">Install as C++ library</h2>

Set the `CMAKE_INSTALL_PREFIX` flag in `cmake` GUI to your desired installation path and then 

```
ninja install
```

We provide examples of linking against both [Chrono](https://github.com/projectchrono/chrono) and _DEME_ for co-simulations in [chrono-projects](https://github.com/projectchrono/chrono-projects/tree/feature/DEME). You need to checkout the `feature/DEME` branch after cloning the code.

You need to build `chrono-projects` linking against a Chrono installation (Chrono installation guide is [here](https://api.projectchrono.org/tutorial_install_chrono_linux.html); note you have to `make install` to install Chrono, not just build it), then link against _DEME_. The steps for building `chrono-projects`:

- Start by linking against Chrono. Set `Chrono_DIR`. It should be in `<your_Chrono_install_dir>/lib/cmake`. Then configure the project;
- Make sure `ENABLE_PROJECTS` to `ON` and configure the project;
- Linkage against Chrono is done, now move on to link against _DEME_. Set `ENABLE_DEME_TESTS` to `ON`. Then configure the project;
- Set `ChPF_DIR` when prompted. It should be in `<your_DEME_install_dir>/lib64/cmake/ChPF`. Then configure the project;
- Set `DEME_DIR` when prompted. It should be in `<your_DEME_install_dir>/lib64/cmake/DEME`. Then configure the project.

Then build the project and you should be able to run the demo scripts that demonstrate the co-simulation between _DEME_ and Chrono.

More documentations on using this package for co-simulations are being added.

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

See [a related _DEME_ paper](https://arxiv.org/abs/2307.03445) and cite
```bibtex
@article{zhang2023gpuaccelerated,
      title={A GPU-accelerated simulator for the DEM analysis of granular systems composed of clump-shaped elements}, 
      author={Ruochun Zhang and Colin Vanden Heuvel and Alexander Schepelmann and Arno Rogg and Dimitrios Apostolopoulos and Samuel Chandler and Radu Serban and Dan Negrut},
      year={2023},
      eprint={2307.03445},
      archivePrefix={arXiv},
      primaryClass={cs.CE}
}
```
