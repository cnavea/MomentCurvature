import math
import numpy as np
import math as m
from typing import Literal

#test

class Column:
    def __init__(self, trans, long, unconfined, confined, d, h, n_slices):
        self.transReinf = trans
        self.longReinf = long
        self.unconfinedConc = unconfined
        self.confinedConc = confined
        self.innerD = (self.transReinf.ringRadius * 2) + (self.transReinf.diameter)
        self.outerD = d
        self.outerR = self.outerD / 2.0
        self.innerR = self.innerD / 2.0
        self.height = h
        self.cover = (self.outerD - self.innerD) / 2.0
        self.xi, self.area_core, self.area_cover = self._precompute_slice_geometry(n_slices)
          #xi - matrix of midpoints of slices
          #area_core - matrix of the areas of each slice in the core
          #area_cover - matrix of the areas of each slice in the cover concrete

    def _precompute_slice_geometry(self, n_slices: int):
        """
        Geometry-only precompute for a circular section discretized into horizontal slices.
        Returns a cache dict you can reuse for many (eps0, phi) evaluations.
        """
        dx = self.outerD / float(n_slices)                #width of slices
        r = self.outerR                                   #radius of section
        xi = np.linspace(-r + dx/2, r - dx/2, n_slices)   #slice midpoints from center (x=0)

        
        h_outer = 2.0 * np.sqrt(r**2 - xi**2)               # outer chord heights of whole section
        
        core_mask = np.abs(xi) <= self.innerR               # array of true/false for each slice (true if xi within confinement)
        h_core = np.zeros_like(h_outer)                     # array of zeros for all xi, but we will only adjust the ones in core
        h_core[core_mask] = 2.0 * np.sqrt(self.innerR ** 2 - xi[core_mask] ** 2)  #assignes these values inside the confinement region
        h_cover = h_outer - h_core                           # unconfined chord heights

        area_core = h_core * dx             #vector of the areas of each core slice
        area_cover = h_cover * dx           #vector of the areas of each cover slice

        return xi, area_core, area_cover

    def axial_forces(self, eps0: float, phi: float):
        """
        axial force (kips) using precomputed slice geometry.
        returns the force of each slice of concrete, each steel rebar, and the total sum of all forces
        """
        # strain at all slice midpoints (vector)
        eps_profile = eps0 - phi * self.xi              #   [ε] = ε0 + ϕ[x]

        # vectorize the material stress calls (your stress() is scalar)
        sig_con = self.confinedConc.stress(eps_profile)
        sig_unCon = self.unconfinedConc.stress(eps_profile)

        # concrete force (kips)
        F_conc = sig_con * self.area_core + sig_unCon * self.area_cover

        # steel bars (small array; vectorization is fine but a sum over rows is also OK)
        F_steel = np.array([])
        if getattr(self.longReinf, "coords", None) is not None:
            xb = np.asarray(self.longReinf.coords)[:, 0]
            eps_b = eps0 - phi * xb           #   [ε] = ε0 + ϕ[x]
            sig_b = self.longReinf.stress(eps_b)
            F_steel = sig_b * self.longReinf.area

        F_total = np.sum(F_conc) + np.sum(F_steel)
        return (np.real_if_close(F_conc, tol=1e-12),
                np.real_if_close(F_steel, tol=1e-12),
                float(np.real_if_close(F_total, tol=1e-12)))

class RebarMaterial:
    def __init__(self, E, Fy, Fu, size, eps_u=0.12, eps_h=0.012, spacing=None, n_bars=None, r=None,
                 start: Literal["left","right","top","bottom"] = "left"):
        self.E = E
        self.Fy = Fy
        self.Fu = Fu
        self.eps_y = self.Fy / self.E
        self.eps_h = eps_h
        self.eps_u = eps_u
        self.diameter = size / 8
        self.area = m.pi * (self.diameter ** 2 ) / 4
        self.spacing = spacing
        self.n_bars = n_bars
        self.ringRadius = r
        self.start = start
        if n_bars is not None:
            self.coords = self._generate_bar_coords()

    def _generate_bar_coords(self) -> np.ndarray:
        #starting angle
        start_angles = {"left":math.pi, "right":0.0, "top":math.pi/2, "bottom":-math.pi/2}
        theta0 = start_angles.get(self.start, math.pi)

        dtheta = 2 * math.pi / self.n_bars

        coords = np.zeros((self.n_bars, 2))
        for i in range(self.n_bars):
            theta = theta0 + i * dtheta
            coords[i,0] = self.ringRadius * math.cos(theta)
            coords[i,1] = self.ringRadius * math.sin(theta)
        return coords

    def stress (self, eps):
        """
        Array-aware steel stress (ksi).
        Accepts float or array-like; returns same shape.
        """
        arr = np.asarray(eps, dtype=float)
        sig = np.zeros_like(arr, dtype=float)
        sgn = np.sign(arr)
        
        # masks
        m_el  = abs(arr) < self.eps_y                                  # linear lastic
        m_pl  = (abs(arr) >= self.eps_y) & (abs(arr) < self.eps_h)     # yield plateau
        m_h   = (abs(arr) >= self.eps_h) & (abs(arr) < self.eps_u)     # hardening
        # (everything else -> zero; e.g., arr >= eps_u or arr < 0 if you want zero in tension)

        # elastic
        sig[m_el] = arr[m_el] * self.E
        # plateau
        sig[m_pl] = sgn[m_pl]*self.Fy
        # hardening (cubic fit)
        dh = abs(arr[m_h]) - self.eps_h
        sig[m_h] = sgn[m_h]*( self.Fy
                     + 710.0  * dh
                     - 6564.0 * dh**2
                     + 20227.0* dh**3 )

        return sig.item() if sig.ndim == 0 else sig


class ConcreteMaterial:
    def __init__(self, fc, rebar=None, Ec=None):
        self.rebar = rebar
        if self.rebar is not None: 
            self.dc = self.rebar.ringRadius
            self.fl = 0 or (0.95 * 2 * self.rebar.area * self.rebar.Fy / ((self.dc * 2) + (self.rebar.diameter)))              #lateral confinement pressure
        else: self.fl = 0
        self.fc = fc
            #Ke = 0.95 for circular cross-sections
        
                   
        # compute dependent parameters once
        self.fcc = self._confined_strength()
        self.eps_cc = self._confined_strain()
        self.Ec = Ec or 57 * ( self.fc * 1000 ) ** 0.5
        self.Esec = self.fcc / self.eps_cc
        self.r_value = self.Ec / (self.Ec - self.Esec)
        self.eps_u = self._ultimate_strain()
        self.fr = 0.24*math.sqrt(self.fc)        #modulus of rupture (cracking)
        self.eps_cr = self.fr/self.Ec

    def _confined_strength(self):
        return self.fc * (2.254 * (1 + 7.94 * self.fl / self.fc) ** 0.5
                          - 2 * self.fl / self.fc - 1.254)

    def _confined_strain(self):
        return 0.002 * (1 + 5 * (self.fcc / self.fc - 1))

    def _ultimate_strain(self):
        if self.rebar == None:
            return 0.004
        row = 4 * self.rebar.area / (self.dc * self.rebar.spacing)
        return 0.004 + (1.4 * row * self.rebar.Fy * self.rebar.eps_u) / self.fcc

    def stress(self, eps):
        """
        Vectorized Mander confined concrete stress.
        eps : float | array_like
            Negative strain = Compression
        Returns: float or np.ndarray matching input shape.
        """
        eps_arr = np.asarray(eps, dtype=float)
        sig = np.zeros_like(eps_arr, dtype=float)
        
        # Valid compression range mask
        maskMander = (eps_arr < 0) & (eps_arr >= -self.eps_u)
        maskTension = (eps_arr >= 0) & (eps_arr < self.fr/self.Ec)
        
        x = -eps_arr[maskMander] / self.eps_cc #mander needs positive strains to work
        #mander formula
        sig_comp = -(self.fcc * self.r_value * x) / (self.r_value - 1.0 + np.power(x, self.r_value)) #turns back to negative
        sig[maskMander] = sig_comp
        sig[maskTension] = eps_arr[maskTension] * self.Ec
        #if input was scalar, then return a scalar
        return sig.item() if sig.ndim == 0 else sig

# ---------------------------------------------------------------
# 3) Hybrid Newton → Bisection solve for eps0 at fixed curvature
# ---------------------------------------------------------------
def find_eps0(col: Column, phi: float, P: float,prevEps0: float,
                      tol_force: float = 1e-3,
                      max_newton: int = 7,
                      max_bisect: int = 60) -> float:
    """
    Solve F(eps0,phi)=P. Try a few Newton steps (clamped in bracket),
    then fall back to bisection to guarantee convergence.
    """
    def total_axial_force(eps0: float, phi: float) -> float:
        """Total axial force (kips) from concrete + steel at (eps0, phi)."""
        _, _, Ftot = col.axial_forces(eps0, phi)
        return Ftot
    
    def residual(eps0: float, phi: float, P: float) -> float:
        return total_axial_force(eps0, phi) - P
    
    def dF_deps0(eps0: float, phi: float, h: float = 1e-6) -> float:
        """Finite-difference derivative dF/deps0 (kips per unit strain)."""
        f1 = total_axial_force(eps0 + h, phi)
        f0 = total_axial_force(eps0 - h, phi)
        return (f1 - f0) / (2.0 * h)

    eps_lo = prevEps0 - 1.0e-3
    eps_hi = prevEps0 + 1.0e-3
    
    # Ensure bracket (expand hi if needed)
    fL = residual(eps_lo, phi, P)
    fR = residual(eps_hi, phi, P)

    NewtonTries = 0
    BisectTries = 0
    tries = 0
    while fL * fR > 0 and tries < 30:
        eps_hi += 0.2 * abs(eps_hi)
        eps_lo -= 0.2 * abs(eps_lo)
        fR = residual(eps_hi,phi,P)
        fL = residual(eps_lo,phi,P)
        tries += 1

    # If already good enough at an endpoint
    if abs(fL) < tol_force: return eps_lo
    if abs(fR) < tol_force: return eps_hi

    
    # --- Newton phase (few iterations, clamped to [eps_lo, eps_hi]) ---
    eps = 0.5 * (eps_lo + eps_hi)
    for _ in range(max_newton):
        NewtonTries += 1
        f = residual(eps, phi, P)
        if abs(f) < tol_force:
            return eps,tries,NewtonTries,BisectTries
        dfdx = dF_deps0(eps, phi)
        if dfdx == 0 or not np.isfinite(dfdx):
            break  # derivative unusable → go bisection
        step = -f / dfdx
        eps_new = eps + step
        # clamp step to bracket (avoid runaway)
        if eps_new <= eps_lo or eps_new >= eps_hi:
            eps_new = 0.5 * (eps_lo + eps_hi)
        # update bracket with sign test
        f_new = residual(eps_new, phi, P)
        if fL * f_new <= 0:
            eps_hi, fR = eps_new, f_new
        else:
            eps_lo, fL = eps_new, f_new
        eps = eps_new

        if abs(f_new) < tol_force:
            return eps,tries,NewtonTries,BisectTries
    
    # --- Bisection phase (robust) ---
    for _ in range(max_bisect):
        BisectTries += 1
        mid = 0.5 * (eps_lo + eps_hi)
        fM = residual(mid, phi, P)
        if abs(fM) < tol_force:
            return mid,tries,NewtonTries,BisectTries
        if fL * fM <= 0:
            eps_hi, fR = mid, fM
        else:
            eps_lo, fL = mid, fM
    return  0.5 * (eps_lo + eps_hi),tries,NewtonTries,BisectTries














