# SBEL GPU DEM-Engine
_A Duo-GPU DEM solver with complex grain geometry support_

### Description

DEM-Engine, nicknamed _DEME_, does Discrete Element Method simulations:

- using up to two GPUs at the same time (works great on consumer _and_ data center GPUs);
- with the particles having complex shapes represented by clumped spheres;
- with support for customizable contact force models (want to add a non-standard cohesive force, or an electrostatic repulsive force? You got this);
- with emphasis on computational efficiency;
- with support for co-simulation with other C/C++ packages, such as [Chrono](https://github.com/projectchrono/chrono).

<p>
  <img width="380" src="https://i.imgur.com/mLMjuTc.jpg">
  <img width="380" src="https://i.imgur.com/PRbd0nJ.jpg">
</p>

Currently _DEME_ is a C++ package. We are building a Python wrapper for it.

### Licensing

This project should be treated as the collective intellectual property of the Author(s) and the University of Wisconsin - Madison. The following copyright statement should be included in any new or modified source files
```
Copyright (c) 2021, Simulation-Based Engineering Laboratory
Copyright (c) 2021, University of Wisconsin - Madison

SPDX-License-Identifier: BSD-3-Clause
```

New authors should add their name to the file `CONTRIBUTORS.md` rather than to individual copyright headers.

### Installation

On a Linux machine, install CUDA if you do not already have it. Useful installation instructions may be found [here](https://developer.nvidia.com/cuda-downloads) and for WSL users, [here](https://docs.nvidia.com/cuda/wsl-user-guide/index.html). Make sure `nvidia-smi` and `nvcc --version` give correct returns. Sometimes after installation, `nvcc` and CUDA necessary libraries are not in `$PATH`, and you may have to manually include them. Depending on the version of CUDA you are using, an example:
```
export PATH=$PATH:/usr/local/cuda-12.1/bin:/usr/local/cuda-12.1/lib64/:/usr/local/cuda-12.1/lib64/cmake/
```

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

Finally, build the project.

```
ninja
```

Some additional troubleshooting tips for building the project:

- If you see some grammatical errors during compilation, such as `filesystem` not being a member of `std` or arguments not expanded with `...`, then manually setting the flag `TargetCXXStandard` to `STD_CXX17` might help.

After the build process is done, you can start trying out the demos.

- An all-rounder beginner example featuring a bladed mixer interacting with complex shaped particles: `./src/demo/DEMdemo_Mixer`
- A fun game-of-life simulator built with the package, showing the flexibility in terms of how you can use this tool: `./src/demo/DEMdemo_GameOfLife`
- A place to learn how prescribed motions work in this package, using either analytical boundaries or particle-represented boundaries: `./src/demo/DEMdemo_Centrifuge` and `./src/demo/DEMdemo_Sieve`
- More advanced examples showing the usage of the custom additional properties (called _wildcards_) that you can associate with the simulation entities, and use them in the force model and/or change them in simulation then deposit them into the output files: `./src/demo/DEMdemo_Indentation` 

More documentations on this package's usage are being added.

_DEME_ is able to handle mesh-represented bodies with relatively simple physics and their interaction with granular materials, for example a plow with a prescribed velocity, or some projectiles that are launched and then hit the ground. However, if the bodies' physics are complex, say it is a vehicle that has joint-connected parts and a motor with certain driving policies, or the meshed bodies have collisions among themselves that needs to be simulated, then _DEME_ alone is perhaps not enough to manage them. But you can export _DEME_ as a library and use it along with other simulation tools, where _DEME_ is exclusively tasked with handling the granular material (with high efficiency, of course). See the following section.

### Install as C++ library

Set the `CMAKE_INSTALL_PREFIX` flag in `cmake` GUI to your desired installation path and then 

```
ninja install
```

We provide examples of linking against both [Chrono](https://github.com/projectchrono/chrono) and _DEME_ for co-simulations in [chrono-projects](https://github.com/projectchrono/chrono-projects/tree/feature/DEME).

More documentations on using this package for  are being added.

#### Notes on code included from Project Chrono

This project exists independently of Chrono so developers should be sure to include the appropriate BSD license header on any code which is sourced from Chrono::FSI, Chrono::GPU(DEM), or other parts of Chrono.

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
