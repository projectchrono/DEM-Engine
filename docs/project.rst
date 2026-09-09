Project information
===================

Community
---------

Discuss DEME on the `Project Chrono forum <https://groups.google.com/g/projectchrono>`_.
`Demo videos <https://uwmadison.app.box.com/s/u4m9tee3k1vizf097zkq3rgv54orphyv>`_
illustrate typical applications.

Licensing and contributors
--------------------------

DEME uses the :download:`BSD-3-Clause license <../LICENSE.md>`.
The project is the collective intellectual property of its authors and the
University of Wisconsin–Madison. Add authors to
:download:`the contributor list <CONTRIBUTORS.md>` rather than individual
copyright headers. Include this notice in new or modified source files:

.. code-block:: text

   Copyright (c) 2021, Simulation-Based Engineering Laboratory
   Copyright (c) 2021, University of Wisconsin - Madison

   SPDX-License-Identifier: BSD-3-Clause

Code sourced from Chrono must also retain the appropriate original license.
The following is the source-header example previously provided in the README:

.. code-block:: text

   Copyright (c) 2021, SBEL GPU Development Team
   Copyright (c) 2021, University of Wisconsin - Madison

   SPDX-License-Identifier: BSD-3-Clause


   This file contains modifications of the code authored by the Project Chrono
   Development Team. The original license can be found below:

   Copyright (c) 2016, Project Chrono Development Team
   All rights reserved.

   Use of this source code is governed by a BSD-style license that can be found
   in the LICENSE file at the top level of the distribution and at
   http://projectchrono.org/license-chrono.txt. A copy of the license is below.

   Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    - Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    - Neither the name of the nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Citation
--------

The DEM-Engine design and usage paper (Computer Physics Communications, 2024):

.. code-block:: bibtex

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

The clump-based granular and rover simulation paper (Engineering with Computers, 2024):

.. code-block:: bibtex

   @article{ruochunGRC-DEM2023,
         title={A {GPU}-accelerated simulator for the {DEM} analysis of granular systems composed of clump-shaped elements},
         author={Ruochun Zhang and Colin {Vanden Heuvel} and Alexander Schepelmann and Arno Rogg and Dimitrios Apostolopoulos and Samuel Chandler and Radu Serban and Dan Negrut},
         year={2024},
         journal={Engineering with Computers},
         doi={https://doi.org/10.1007/s00366-023-01921-9}
   }
