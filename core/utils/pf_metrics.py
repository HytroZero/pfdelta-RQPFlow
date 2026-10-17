"""Independent branch-based complex power mismatch for numerical solvers."""
import torch
from core.utils.ac_power_flow import branch_power_injections


def official_pbl(system, output):
    """Return mean/max per-bus |delta P + j delta Q| in per-unit.

    Uses endpoint branch powers rather than the solver's sparse Y-bus residual.
    Both branch-end shunts, transformer taps/shifts and bus shunts are included.
    """
    vm, angle, p_net, q_net = (torch.as_tensor(output[k], dtype=torch.float64)
                              for k in ('voltage', 'angle', 'p_net', 'q_net'))
    a = torch.as_tensor(system.edge_attr, dtype=torch.float64)
    f, t = torch.as_tensor(system.edge_index, dtype=torch.long)
    p, q = branch_power_injections(vm, angle, a[:, 0], a[:, 1], a[:, 6], a[:, 7], f, t,
                                  conductance_from=a[:, 2], susceptance_from=a[:, 3],
                                  conductance_to=a[:, 4], susceptance_to=a[:, 5])
    shunt = torch.as_tensor(system.shunt, dtype=torch.float64)
    mismatch = torch.hypot(p+vm.square()*shunt[:, 0]-p_net,
                           q-vm.square()*shunt[:, 1]-q_net)
    return float(mismatch.mean()), float(mismatch.max())
