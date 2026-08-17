from mpmath import mp, mpf, mpc
import scipy
import numpy as _np

I = mpc("0", "1")
ZERO = mpf("0")
ONE = mpf("1")
TWO = mpf("2")

def zeros(dim):
    return _np.array(mp.zeros(dim)).reshape(dim, dim) * ONE

def eye(dim):
    return _np.array(mp.eye(dim)).reshape(dim, dim) * ONE

class BesselModel:
    """
    Model for matrix-valued correlation function in which spectral weights are
    given by Bessel functions. This model does not have a particularly strong
    physical motivation, but it readily accomodates a general (n, n) matrix.
    """
    def __init__(self, nt, nops, nstates=200, dE=0.01):
        self.nt = nt
        self.nops = nops
        self.nstates = nstates
        self.energy = mpf(str(dE)) * (_np.arange(nstates) + 1)
        self.correlator = self.evaluate_correlator()

    def _z(self, idx_state, idx_operator):
        """
        Evaluates spectral weights defined to have the form Z_{na} = J_a(5 E_n).
        """
        phase = mp.one
        return mp.besselj(idx_operator, 5*self.energy[idx_state]) * phase

    def evaluate_correlator(self):
        """
        Evaluates the correlation function assuming decaying contributions only.
        """
        correlator = _np.array([mp.zero for _ in range(self.nt*self.nops*self.nops)])
        correlator = correlator.reshape((self.nt, self.nops, self.nops))
        for n in range(self.nstates):
            En = self.energy[n]
            z = _np.empty((self.nops, self.nops), dtype=object)
            for a in range(self.nops):
                zna = self._z(n, a)
                for b in range(self.nops):
                    znb = self._z(n, b)
                    z[a,b] = zna * mp.conjugate(znb)
            for t in range(self.nt):
                correlator[t] += z * mp.exp(-mp.fmul(En, t))
        return correlator

    def rho_smeared(self, lmbda):
        """
        Evaluate the smeared spectral function in the complex
        transfer-matrix-eigenvalue plane.
        """
        result = _np.array([mp.zero for _ in range(2*2)])
        result = result.reshape((2, 2))
        for k in range(self.nstates):
            Ek = self.energy[k]
            lmbda_km = mp.exp(-Ek)  # Positive real energy axis in (0,1)
            zk0 = self._z(k, 0)
            zk1 = self._z(k, 1)
            z = _np.array([
                [zk0*mp.conjugate(zk0), zk0*mp.conjugate(zk1)],
                [zk1*mp.conjugate(zk0), zk1*mp.conjugate(zk1)]])
            result += z / (lmbda - lmbda_km)
        return result

    def rho_smeared_physical(self, z):
        """
        Evaluate the smeared spectral function in the complex energy plane.
        """
        result = _np.array([mp.zero for _ in range(2*2)])
        result = result.reshape((2, 2))
        for k in range(self.nstates):
            Ek = self.energy[k]
            zk0 = self._z(k, 0)
            zk1 = self._z(k, 1)
            amplitude_matrix = _np.array([
                [zk0*mp.conjugate(zk0), zk0*mp.conjugate(zk1)],
                [zk1*mp.conjugate(zk0), zk1*mp.conjugate(zk1)]])
            result += amplitude_matrix / (z - Ek)
        return result

class StaggeredModel:
    """
    Model for a staggered thermal correlation function.
    """
    def __init__(self, nt, nstates, decaying=True, thermal=False):
        assert (thermal is True) or (decaying is True), "Must have _some_ states!"
        self.decaying = decaying
        self.thermal = thermal
        self.nt = nt
        self.nstates = nstates
        self.correlator = self.evaluate_correlator()
        shape = self.correlator.shape
        assert shape[1] == shape[2]
        self.nops = shape[2]

    def _ztilde(self, idx_state, idx_operator):
        """Evaluate the spectral weights."""
        if idx_operator == 0:
            if (idx_state % 2) == 0:
                return ONE
            return mpf("0.1")
        elif idx_operator == 1:
            if idx_state == 0:
                return mpf("-0.1")
            if (idx_state % 2) == 1:
                return ONE
            return mpf("0.1")
        else:
            raise ValueError

    def _energy_decay(self, k):
        """Evaluate the decaying energies."""
        return mpf("0.1") * (k+1)

    def _energy_osc(self, k):
        """Evaluate the opposite-parity/"oscillating" energies."""
        return mpf("0.2") * (k+1)

    def evaluate_correlator(self):
        """Evaluate the correlation function."""
        correlator = _np.array([mp.zero for _ in range(self.nt*2*2)])
        correlator = correlator.reshape((self.nt, 2, 2))
        for k in range(self.nstates):

            # Decaying states
            Ek = self._energy_decay(k)
            zk0 = self._ztilde(k, 0)/mp.sqrt(mp.fmul(TWO, Ek))
            zk1 = self._ztilde(k, 1)/mp.sqrt(mp.fmul(TWO, Ek))
            z = _np.array([
                [zk0*mp.conjugate(zk0), zk0*mp.conjugate(zk1)],
                [zk1*mp.conjugate(zk0), zk1*mp.conjugate(zk1)]])

            for t in range(self.nt):
                if self.decaying:
                    correlator[t] += z * mp.exp(-Ek*t)
                if self.thermal:
                    correlator[t] += z * mp.exp(-Ek*(self.nt-t))

            # Oscillating states
            Ek = self._energy_osc(k)
            zk0 = self._ztilde(k, 0)/mp.sqrt(mp.fmul(TWO, Ek))
            zk1 = self._ztilde(k, 1)/mp.sqrt(mp.fmul(TWO, Ek))
            z = _np.array([
                [zk0*mp.conjugate(zk0), zk0*mp.conjugate(zk1)],
                [zk1*mp.conjugate(zk0), zk1*mp.conjugate(zk1)]])
            for t in range(self.nt):
                if self.decaying:
                    correlator[t] += (-1)**t * z * mp.exp(-Ek*t)
                if self.thermal:
                    correlator[t] += (-1)**t * z * mp.exp(-Ek*(self.nt-t))

        return correlator

    def rho_smeared(self, lmbda):
        """
        Evaluate the smeared spectral function in the complex
        transfer-matrix-eigenvalue plane.
        """
        result = _np.array([mp.zero for _ in range(2*2)])
        result = result.reshape((2, 2))

        for k in range(self.nstates):

            # Decaying states
            Ek = self._energy_decay(k)
            lmbda_km = mp.exp(-Ek)  # Positive real axis in (0,1)
            lmbda_kp = mp.exp(+Ek)  # Positive real axis in (1,infty)
            zk0 = self._ztilde(k, 0)/mp.sqrt(mp.fmul(TWO, Ek))
            zk1 = self._ztilde(k, 1)/mp.sqrt(mp.fmul(TWO, Ek))
            z = _np.array([
                [zk0*mp.conjugate(zk0), zk0*mp.conjugate(zk1)],
                [zk1*mp.conjugate(zk0), zk1*mp.conjugate(zk1)]])
            if self.decaying:
                result += z / (lmbda - lmbda_km)
            if self.thermal:
                result += z / (lmbda - lmbda_kp) * (1/lmbda_kp)**self.nt

            # Oscillating states
            Ek = self._energy_osc(k)
            lmbda_km = -mp.exp(-Ek)  # Negative real axis in (-1,0)
            lmbda_kp = -mp.exp(+Ek)  # Negative real axis in (infty,1)
            zk0 = self._ztilde(k, 0)/mp.sqrt(mp.fmul(TWO, Ek))
            zk1 = self._ztilde(k, 1)/mp.sqrt(mp.fmul(TWO, Ek))
            z = _np.array([
                [zk0*mp.conjugate(zk0), zk0*mp.conjugate(zk1)],
                [zk1*mp.conjugate(zk0), zk1*mp.conjugate(zk1)]])
            if self.decaying:
                result += z / (lmbda - lmbda_km)
            if self.thermal:
                result += z / (lmbda - lmbda_kp) * (-1/lmbda_kp)**self.nt
        return result

    def rho_smeared_physical(self, z, choice):
        """
        Evaluate the smeared spectral function in the complex energy plane.
        """
        assert choice in ('decay', 'oscillating')

        result = _np.array([mp.zero for _ in range(2*2)])
        result = result.reshape((2, 2))

        for k in range(self.nstates):
            if choice == 'decay':
                # Decaying states
                Ek = self._energy_decay(k)
            elif choice == 'osc':
                # Oscillating states
                Ek = self._energy_osc(k)
            else:
                raise ValueError(f"Unexpected choice '{choice}'")

            zk0 = self._ztilde(k, 0)/mp.sqrt(mp.fmul(TWO, Ek))
            zk1 = self._ztilde(k, 1)/mp.sqrt(mp.fmul(TWO, Ek))
            amplitude_matrix = _np.array([
                [zk0*mp.conjugate(zk0), zk0*mp.conjugate(zk1)],
                [zk1*mp.conjugate(zk0), zk1*mp.conjugate(zk1)]])
            result += amplitude_matrix / (z - Ek)
        return result


def gaussian(x, mu, sigma, use_mp=True):
    """
    Evaluate a unit-normalized Gaussian function with mean mu
    and standard deviation sigma at the point x.
    """
    if use_mp:
        if not hasattr(x, '__iter__'):
            # ((x-mu)/sigma)**2
            tmp = mp.power(mp.fdiv(mp.fsub(x, mu), sigma), mpf("2"))
            # exp(-0.5*(x-mu)**2/sigma**2)
            tmp = mp.exp(mp.fmul(mp.fneg(mpf('1/2')), tmp))
            # 1/sqrt(2*pi)/sigma
            norm = mp.fdiv(mp.one, mp.sqrt(mp.fmul(mpf("2"), mp.pi)))
            norm = mp.fdiv(norm, sigma)
            return mp.fmul(tmp, norm)
        else:
            return [gaussian(xi, mu, sigma) for xi in x]
    else:
        return _np.exp(-(x-mu)**2/(2*sigma**2)) / _np.sqrt(2*_np.pi*sigma**2)


def J2(dim):
    """
    Computes the J structure [[0, i*I],[-i*I, 0]] for the Hamburger momentum problem.
    """
    zero = zeros(dim)
    identity = eye(dim)
    return _np.block([
        [zero, I*identity],
        [-I*identity, zero]
    ])


def compute_hankel(corr, test=True):
    """
    Computes the Hankel matrix associated with a correlator, G_{m,n} = C[n+m].

    Parameters
    ----------
    corr : (nt, dim, dim), ndarray
        Euclidean-time correlation function

    Returns
    -------
    h : (dim*nt//2, dim*nt//2), ndarray
        The Hankel matrix

    Notes
    -----
    A valid correlator must have a positive semi-definite Hankel matrix.
    """
    nt = len(corr)
    arr = []
    for i in range(nt//2):
        arr.append([corr[i+j] for j in range(nt//2)])
    arr = _np.block(arr)
    if test:
        # Check that the Hankel matrix is positive definite
        _ = mp.cholesky(mp.matrix(arr))
    return arr


def compute_b(z, npts, dim):
    """
    Computes the block column matrix b(z) = [I, z*I, z**2*I, ..., z**(n-1)*I]
    """
    identity = eye(dim)
    return _np.vstack([mp.power(z, n) * identity for n in range(npts)])


def compute_c(z, npts, correlator):
    """
    Compute the block column matrix
    c(z) = -[0, C0, C(1)+z*C(0), ..., C(n-2) + z*C(n-3) + ... + z**(n-2)*C[0]]
    """
    dim = correlator.shape[1]
    assert correlator.shape[1] == correlator.shape[2]

    blocks = [zeros(dim)]
    for nterms in range(1, npts):
        # print(f"nterms {nterms}")
        block = zeros(dim)
        for nidx in range(nterms):
            midx = nterms-1-nidx
            # print(f"C[{nidx}] * z**{midx}")
            block += correlator[nidx] * mp.power(z, midx)
        blocks.append(block)
    c = -_np.vstack(blocks)
    return c


def compute_bc(z, correlator):
    """
    Computes the block column matrices b(z) and c(z).
    """
    nt = correlator.shape[0]
    dim = correlator.shape[1]
    assert correlator.shape[1] == correlator.shape[2]

    b = compute_b(z, nt//2, dim)
    c = compute_c(z, nt//2, correlator)
    return b, c


def inv(A):
    """
    Computes the inverse of a matrix A.
    """
    _A = mp.matrix(A)
    # Note: mp.matrix inversion via **-1
    return _np.array(_A**-1, dtype=object).reshape(A.shape)


def sqrtm(A):
    """
    Computes the square root of a matrix A.
    """
    _A = _np.matrix(A, dtype=complex)
    return scipy.linalg.sqrtm(_A)


def norm(vec):
    """
    Compute the p=2 norm of a vector.
    """
    return mp.sqrt(_np.sum([vn * mp.conjugate(vn) for vn in vec])).real


def compute_coefficient_matrix(z, correlator, test=True):
    """
    Computes the coefficient matrix associated with the fractional linear
    transformation describing the general solution to the Hamburger moment
    problem.

    Parameters
    ----------
    z : complex / mpc
        Point in the complex transfer-matrix-eigenvalue plane
    correlator : (nt, dim, dim), ndarray
        Euclidean-time correlation function

    Returns
    -------
    A : (2*dim, 2*dim), ndarray
        Coefficient matrix, best understood as a (2, 2) block matrix with each
        block of size (dim, dim).

    Notes
    -----
    Theorem (Kovalishina):
    The general solution w(z) of FMI(H) is representable as a fractional linear
    transformation of an arbitrary Nevanlinna pair [p(z), q(z)]^T:

    w(z) = [alpha(z)p(z) + beta(z)q(z)]/[gamma(z)p(z) + delta(z)q(z)]^{-1}

    whose coefficient matrix

    \mathfrak{A} = [[alpha(z), beta(z)], [gamma(z), delta(z)]]

    is constructed from the matrix S_H >0:

    \mathfrak{A} = I - i*z*J2 [b^\dagger(zbar), -c^\dagger(zbar)] S_H^{-1} [b(0), -c(0)]^T

    and is a matrix-valued function that is J2-expanding in the upper half-plane,
    J2-unitary on the real axis, of full rank, and has a pole of order n at z=\infty.
    """
    dim = correlator.shape[1]
    assert correlator.shape[1] == correlator.shape[2]

    hankel = compute_hankel(correlator)
    bz, cz = compute_bc(mp.conjugate(z), correlator)
    b0, c0 = compute_bc(ZERO, correlator)

    bc_col = _np.vstack([bz.conjugate().T, -cz.conjugate().T])
    bc_row = _np.hstack([b0, -c0])

    identity = eye(2*dim)
    j2 = J2(dim)

    # Could also do with linear solve
    A = identity - 1j * z * j2 @ bc_col @ inv(hankel) @ bc_row
    return A


def compute_weyl_matrix(z, correlator):
    """
    Computes the Weyl matrix,
    $W(z) = (\mathfrac{A}^\dagger)^{-1} J_2 \mathfrak{A}^{-1}$
    """
    dim = correlator.shape[1]
    assert correlator.shape[1] == correlator.shape[2]
    A = compute_coefficient_matrix(z, correlator)
    j2 = J2(dim)
    Ainv = inv(A)
    weyl = Ainv.conjugate().T @ j2 @ Ainv
    return weyl


def unpack_two_by_two(arr):
    """
    Unpacks an array into (2,2) blocks.

    Parameters
    ----------
    arr : (2*dim, 2*dim), ndarray

    Returns
    -------
    [[a,b], [c, d]] : (dim, dim) blocks
    """
    assert arr.shape[0] == arr.shape[1]
    assert (arr.shape[0] % 2) == 0
    dim = arr.shape[0]//2
    a = arr[0:dim, 0:dim]
    b = arr[0:dim, dim:]
    c = arr[dim:, 0:dim]
    d = arr[dim:, dim:]
    return [[a,b],[c,d]]


def unpack_weyl(weyl, debug=False):
    """
    Unpacks the elements of the (2,2) block Weyl matrix
    W(z) = [[-R(z), S(z)], [S(z)^\dagger, -T(z)]].
    """
    (a,b), (c,d) = unpack_two_by_two(weyl)
    R, S = -a, b
    Sdagger, T = c, -d
    if debug:
        # Check that Sdagger @ Rinv @ S - T is positive definite
        arr = Sdagger @ inv(R) @ S - T
        _ = mp.cholesky(mp.matrix(arr))
    return R, S, Sdagger, T
