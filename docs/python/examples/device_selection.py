"""Select DEME worker devices explicitly."""

import deme


# The first ID selects the dynamic worker; the second selects the
# kinematic/contact-detection worker.
solver = deme.DEMSolver([2, 3])
print("DEME worker devices [dT, kT]:", solver.GetGPUDeviceIDs())
