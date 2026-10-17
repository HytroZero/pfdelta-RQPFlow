"""FP64 fixed-bus AC equations, sparse analytic Jacobian and PFDelta adapter.

The numerical backend is SciPy/CPU. Torch residuals share the input contract,
but use the independent branch-power implementation for differentiable training.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import torch


class InvalidPFInput(ValueError):
    pass


def array(value, dtype=np.float64):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


@dataclass
class PFSystem:
    bus_type: np.ndarray
    known: np.ndarray
    shunt: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray

    def __post_init__(self):
        self.bus_type = array(self.bus_type, np.int64).copy()
        self.known = array(self.known).copy()
        self.shunt = array(self.shunt).copy()
        self.edge_index = array(self.edge_index, np.int64).copy()
        self.edge_attr = array(self.edge_attr).copy()
        self.n = len(self.bus_type)
        if (not self.n or self.known.shape != (self.n, 2)
                or self.shunt.shape != (self.n, 2)
                or self.edge_index.shape != (2, len(self.edge_attr))
                or self.edge_attr.shape != (len(self.edge_attr), 8)):
            raise InvalidPFInput("Expected bus (n,2), shunt (n,2), and branch (m,8) arrays")
        if not all(np.isfinite(a).all() for a in (self.known, self.shunt, self.edge_attr)):
            raise InvalidPFInput("Non-finite input")
        if not np.isin(self.bus_type, [1, 2, 3]).all():
            raise InvalidPFInput("Bus types must be PQ=1, PV=2, slack=3")
        if self.edge_index.size and (self.edge_index.min() < 0 or self.edge_index.max() >= self.n):
            raise InvalidPFInput("Branch endpoint outside bus array")
        self.pq = np.flatnonzero(self.bus_type == 1)
        self.free = np.flatnonzero(self.bus_type != 3)
        self.slack = np.flatnonzero(self.bus_type == 3)
        if np.any(self.known[self.bus_type != 1, 1] <= 0):
            raise InvalidPFInput("Prescribed voltage must be positive")
        a = self.edge_attr
        if np.any((a[:, 0] == 0) & (a[:, 1] == 0)) or np.any(a[:, 6] < 0):
            raise InvalidPFInput("Zero impedance or negative transformer tap")
        f, t = self.edge_index
        graph = sp.coo_matrix((np.ones(len(f)), (f, t)), shape=(self.n, self.n))
        _, labels = connected_components(graph, directed=False)
        for island in np.unique(labels):
            if np.count_nonzero((labels == island) & (self.bus_type == 3)) != 1:
                raise InvalidPFInput("Each electrical island requires exactly one slack bus")
        self.islands = labels
        self.reference_angle = np.zeros(self.n)
        for slack in self.slack:
            self.reference_angle[labels == labels[slack]] = self.known[slack, 0]
        self.p_spec = np.where(self.bus_type == 1, -self.known[:, 0], self.known[:, 0])
        self.q_spec = -self.known[:, 1]
        y = 1 / (a[:, 0] + 1j * a[:, 1])
        tap = np.where(a[:, 6] == 0, 1, a[:, 6]) * np.exp(1j * a[:, 7])
        self.yff = (y + a[:, 2] + 1j * a[:, 3]) / np.abs(tap)**2
        self.ytt = y + a[:, 4] + 1j * a[:, 5]
        self.yft = -y / tap.conj()
        self.ytf = -y / tap
        self.Y = sp.coo_matrix((np.concatenate((self.yff, self.yft, self.ytf, self.ytt,
                                               self.shunt[:, 0] + 1j*self.shunt[:, 1])),
                              (np.concatenate((f, f, t, t, np.arange(self.n))),
                               np.concatenate((f, t, f, t, np.arange(self.n))))),
                              shape=(self.n, self.n)).tocsr()
        self.size = len(self.free) + len(self.pq)

    @classmethod
    def from_data(cls, data):
        bus, branch = data['bus'], data['bus', 'branch', 'bus']
        if getattr(data, 'num_graphs', 1) != 1:
            raise InvalidPFInput("Use solve_batch for a multi-graph Batch")
        # PFDelta normally removes offline branches. Also support explicit status.
        attr, index = branch.edge_attr, branch.edge_index
        status = getattr(branch, 'edge_status', None)
        if status is not None:
            mask = status.reshape(-1).bool()
            attr, index = attr[mask], index[:, mask]
        return cls(bus.bus_type, bus.x, bus.shunt, index, attr)

    def initial_state(self, v0=None):
        if v0 is None:
            return np.r_[self.reference_angle[self.free], np.zeros(len(self.pq))]
        v0 = array(v0, np.complex128)
        if v0.shape != (self.n,) or not np.isfinite(v0).all() or np.any(np.abs(v0) == 0):
            raise InvalidPFInput("V0 must be a finite nonzero complex voltage vector")
        return np.r_[np.angle(v0)[self.free], np.log(np.abs(v0[self.pq]))]

    def voltage(self, x):
        angle = self.reference_angle.copy()
        vm = np.ones(self.n)
        vm[self.bus_type != 1] = self.known[self.bus_type != 1, 1]
        angle[self.free] = x[:len(self.free)]
        with np.errstate(over='ignore', invalid='ignore'):
            vm[self.pq] = np.exp(x[len(self.free):])
            return vm * np.exp(1j * angle)

    def residual(self, x):
        v = self.voltage(x)
        with np.errstate(over='ignore', invalid='ignore'):
            s = v * (self.Y @ v).conj()
        return np.r_[s.real[self.free] - self.p_spec[self.free],
                     s.imag[self.pq] - self.q_spec[self.pq]]

    def jacobian(self, x):
        v = self.voltage(x)
        diag_v = sp.diags(v)
        s = v * (self.Y @ v).conj()
        cross = diag_v @ self.Y.conjugate() @ sp.diags(v.conj())
        da = 1j * (sp.diags(s) - cross)
        du = sp.diags(s) + cross
        return sp.bmat([[da.real.tocsr()[self.free][:, self.free], du.real.tocsr()[self.free][:, self.pq]],
                        [da.imag.tocsr()[self.pq][:, self.free], du.imag.tocsr()[self.pq][:, self.pq]]], format='csr')

    def output(self, x):
        v = self.voltage(x)
        angle = self.reference_angle.copy()
        angle[self.free] = x[:len(self.free)]
        s = v * (self.Y @ v).conj()
        p = self.p_spec.copy()
        q = self.q_spec.copy()
        p[self.slack] = s.real[self.slack]
        q[self.bus_type != 1] = s.imag[self.bus_type != 1]
        f, t = self.edge_index
        sf = v[f] * (self.yff*v[f] + self.yft*v[t]).conj()
        st = v[t] * (self.ytf*v[f] + self.ytt*v[t]).conj()
        mismatch = np.hypot(s.real-p, s.imag-q)
        return dict(voltage=np.abs(v), angle=angle, p_net=p, q_net=q,
                    branch_from=sf, branch_to=st, pbl_mean=float(mismatch.mean()),
                    pbl_max=float(mismatch.max()))

    def torch_residual(self, x):
        from core.utils.ac_power_flow import branch_power_injections
        def tensor(a, dtype=torch.float64):
            return torch.as_tensor(a, dtype=dtype, device=x.device)
        angle = tensor(self.reference_angle).index_copy(0, tensor(self.free, torch.long), x[:len(self.free)])
        vm = tensor(np.where(self.bus_type == 1, 1., self.known[:, 1]))
        vm = vm.index_copy(0, tensor(self.pq, torch.long), x[len(self.free):].exp())
        a = tensor(self.edge_attr)
        f, t = tensor(self.edge_index, torch.long)
        p, q = branch_power_injections(vm, angle, a[:, 0], a[:, 1], a[:, 6], a[:, 7], f, t,
                                      conductance_from=a[:, 2], susceptance_from=a[:, 3],
                                      conductance_to=a[:, 4], susceptance_to=a[:, 5])
        p = p + vm.square()*tensor(self.shunt[:, 0])
        q = q - vm.square()*tensor(self.shunt[:, 1])
        return torch.cat(((p-tensor(self.p_spec))[tensor(self.free, torch.long)],
                          (q-tensor(self.q_spec))[tensor(self.pq, torch.long)]))


def equilibrate(j, rounds=3, floor=1e-12):
    """Three alternating infinity-norm Ruiz scaling rounds; never densifies J."""
    a = j.copy().tocsr()
    dr, dc = np.ones(j.shape[0]), np.ones(j.shape[1])
    for _ in range(rounds):
        r = 1 / np.sqrt(np.maximum(abs(a).max(axis=1).toarray().ravel(), floor))
        a = sp.diags(r) @ a
        c = 1 / np.sqrt(np.maximum(abs(a).max(axis=0).toarray().ravel(), floor))
        a = (a @ sp.diags(c)).tocsr()
        dr *= r
        dc *= c
    return a, dr, dc
