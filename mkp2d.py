"""Výsledek numerického cvičení: MKP pro rovinnou deformaci v NumPy/SciPy.

Kód sleduje značení kapitoly 6 učebního textu: referenční prvek, triangulace
(coord, elem), kvadratura, izoparametrická transformace, matice B a D_elast,
K = B^T D_elast B a vektor objemových sil. Žádný cyklus přes prvky.

    from mkp2d import *
    ref = referencni_prvek(*P1)
    coord, elem = sit_obdelnik(8, 2, "P1")
    u, K, b, free = vyres(ref, coord, elem, F=(0, -1e-3))

Síť z Gmsh (po gmsh.model.mesh.generate) převede z_gmsh():

    coord, elem, skupiny = z_gmsh()
    u, *_ = vyres(ref, coord, elem["Triangle 3"], F, gamma0=skupiny["Gamma0"])
"""
import numpy as np
import sympy as sp
from scipy import sparse
from scipy.sparse.linalg import spsolve

xi1, xi2, xi3 = sp.symbols("xi1 xi2 xi3")
XI = (xi1, xi2, xi3)


# ---------------------------------------------------------------- 6.2 a 6.4
def baze_z_predpisu(predpis, uzly):
    """Bazické funkce phi^_p = sum_k C[k, p] m_k splňující (6.1) pro předpis s členy m_k (2D)."""
    m = sp.Matrix(predpis)
    V = sp.Matrix([list(m.subs({xi1: a, xi2: b})) for a, b in uzly])    # V[q, k] = m_k(N^_q)
    return (V.inv().T * m).applyfunc(sp.factor)                         # V C = I, C = V^-1


def referencni_prvek(phi, uzly, body, vahy):
    """Bazické funkce a jejich derivace v kvadraturních bodech.

    phi  : seznam výrazů SymPy v proměnných xi1, xi2 (ve 3D i xi3)
    uzly : body N^_1, ..., N^_np referenčního prvku (ve stejném pořadí)
    body, vahy : kvadratura A^_q, w^_q na referenčním prvku
    """
    phi = sp.Matrix(phi)
    xi = XI[:len(uzly[0])]
    V = sp.Matrix([[f.subs(dict(zip(xi, N))) for N in uzly] for f in phi])
    if V != sp.eye(len(phi)):                       # podmínka (6.1)
        raise ValueError("Bazické funkce nesplňují phi_p(N_q) = delta_pq.")
    dphi = phi.jacobian(xi)
    vycisli = lambda M: np.array(
        [[[float(v) for v in M.subs(dict(zip(xi, A))).row(p)]
          for p in range(M.rows)] for A in body])
    return dict(
        phi=phi, uzly=np.array(uzly, float),
        Phi=vycisli(phi)[:, :, 0],                  # Phi[q, p]
        dPhi=vycisli(dphi),                         # dPhi[q, p, j]
        body=np.array(body, float), vahy=np.array(vahy, float),
        rohy=3 if len(uzly) in (3, 6) else 4,
    )


g = 1 / np.sqrt(3)
P1 = ([1 - xi1 - xi2, xi1, xi2],
      [(0, 0), (1, 0), (0, 1)],
      [(1 / 3, 1 / 3)], [1 / 2])
Q1 = ([(1 - xi1) * (1 - xi2) / 4, (1 + xi1) * (1 - xi2) / 4,
       (1 + xi1) * (1 + xi2) / 4, (1 - xi1) * (1 + xi2) / 4],
      [(-1, -1), (1, -1), (1, 1), (-1, 1)],
      [(-g, -g), (g, -g), (g, g), (-g, g)], [1, 1, 1, 1])


# ---------------------------------------------------------------- 6.1
def sit_obdelnik(nx, ny, typ, L=4.0, H=1.0):
    """Pravidelná síť obdélníku (0,L)x(0,H); vrcholy levotočivě."""
    x, y = np.meshgrid(np.linspace(0, L, nx + 1), np.linspace(0, H, ny + 1))
    coord = np.column_stack([x.ravel(), y.ravel()])
    c = np.arange(len(coord)).reshape(ny + 1, nx + 1)
    n1, n2, n3, n4 = (c[:-1, :-1].ravel(), c[:-1, 1:].ravel(),
                      c[1:, 1:].ravel(), c[1:, :-1].ravel())
    if typ == "Q1":
        return coord, np.column_stack([n1, n2, n3, n4])
    if typ == "P1":   # každý obdélník rozdělíme na dva trojúhelníky
        return coord, np.column_stack([n1, n2, n3, n1, n3, n4]).reshape(-1, 3)
    raise ValueError(typ)


def z_gmsh(dim=2):
    """Aktuální síť Gmsh: coord, elem (slovník podle typu, např. "Triangle 3")
    a indexy uzlů fyzikálních skupin. Tagy uzlů Gmsh přečísluje na 0..n_n-1.

    Pozor: šestiuzlové trojúhelníky Gmsh mají středy hran v pořadí 1-2, 2-3, 3-1;
    pořadí z obrázku 6.3 dává elem["Triangle 6"][:, [0, 1, 2, 4, 5, 3]].
    """
    import gmsh
    tagy, xyz, _ = gmsh.model.mesh.getNodes()
    index = np.zeros(int(tagy.max()) + 1, dtype=int)
    index[tagy.astype(int)] = np.arange(len(tagy))
    coord = xyz.reshape(-1, 3)[:, :dim]
    elem = {}
    for typ in gmsh.model.mesh.getElementTypes():
        nazev, _, _, n_p, *_ = gmsh.model.mesh.getElementProperties(typ)
        _, uzly = gmsh.model.mesh.getElementsByType(typ)
        elem[nazev] = index[uzly.astype(int)].reshape(-1, n_p)
    skupiny = {}
    for d, tag in gmsh.model.getPhysicalGroups():
        uzly, _ = gmsh.model.mesh.getNodesForPhysicalGroup(d, tag)
        skupiny[gmsh.model.getPhysicalName(d, tag)] = index[uzly.astype(int)]
    return coord, elem, skupiny


# ---------------------------------------------------------------- 6.3 a 6.4
def geometrie(ref, coord, elem):
    """Jakobiány, fyzikální gradienty (6.9) a váhy w_T,q (6.7)."""
    X = coord[elem]                                        # X[e, p, i] = N_T,p
    J = np.einsum("epi,qpj->eqij", X, ref["dPhi"])         # (6.4)
    detJ = np.linalg.det(J)
    if np.any(detJ <= 0):
        raise ValueError("det J <= 0: zkontrolujte číslování uzlů v elem.")
    dphi = np.einsum("qpj,eqji->eqpi", ref["dPhi"], np.linalg.inv(J))
    w = ref["vahy"] * detJ                                 # w[e, q]
    return dphi, w


# ---------------------------------------------------------------- 6.7
def stupne_volnosti(elem):
    """edof[e, 2p+i] = 2*elem[e, p] + i (pořadí v1(N1), v2(N1), ...)."""
    edof = np.empty((len(elem), 2 * elem.shape[1]), dtype=int)
    edof[:, 0::2] = 2 * elem
    edof[:, 1::2] = 2 * elem + 1
    return edof


def matice_tuhosti(ref, coord, elem, lam=1.0, mu=1.0):
    """K = B^T D_elast B jako v závěru sekce 6.7."""
    dphi, w = geometrie(ref, coord, elem)
    ne, nq, n_p, _ = dphi.shape
    Bv = np.zeros((ne, nq, 3, 2 * n_p))                    # bloky B_T,q
    Bv[:, :, 0, 0::2] = dphi[..., 0]                       # eps11
    Bv[:, :, 1, 1::2] = dphi[..., 1]                       # eps22
    Bv[:, :, 2, 0::2] = dphi[..., 1]                       # 2 eps12
    Bv[:, :, 2, 1::2] = dphi[..., 0]
    radky = np.arange(3 * ne * nq).reshape(ne, nq, 3, 1)
    sloupce = stupne_volnosti(elem)[:, None, None, :]
    radky, sloupce = np.broadcast_arrays(radky, sloupce)
    B = sparse.coo_array((Bv.ravel(), (radky.ravel(), sloupce.ravel())),
                         shape=(3 * ne * nq, 2 * len(coord))).tocsr()
    C = np.array([[lam + 2 * mu, lam, 0], [lam, lam + 2 * mu, 0], [0, 0, mu]])
    D = sparse.kron(sparse.diags_array(w.ravel()), C, format="csr")
    return (B.T @ D @ B).tocsr()


# ---------------------------------------------------------------- 6.8
def vektor_sil(ref, coord, elem, F):
    """b~ = sum w_T,q M_q^T f_T,q pro konstantní objemovou sílu F."""
    _, w = geometrie(ref, coord, elem)
    be = np.einsum("eq,qp,i->epi", w, ref["Phi"], np.asarray(F, float))
    edof = stupne_volnosti(elem)
    return np.bincount(edof.ravel(), be.reshape(len(elem), -1).ravel(),
                       minlength=2 * len(coord))


# ---------------------------------------------------------------- 6.6
def vyres(ref, coord, elem, F, gamma0=None, lam=1.0, mu=1.0):
    """Pevné uchycení na Gamma_0 (maska nebo indexy uzlů; výchozí: x1 = 0), řešení K u = b."""
    if gamma0 is None:
        gamma0 = np.isclose(coord[:, 0], 0.0)
    gamma0 = np.asarray(gamma0)
    if gamma0.dtype != bool:
        gamma0 = np.isin(np.arange(len(coord)), gamma0)
    K = matice_tuhosti(ref, coord, elem, lam, mu)
    b = vektor_sil(ref, coord, elem, F)
    ID = np.flatnonzero(~gamma0)                           # uzly mimo Gamma_0
    free = np.column_stack([2 * ID, 2 * ID + 1]).ravel()
    u = np.zeros(2 * len(coord))
    u[free] = spsolve(K[free][:, free], b[free])
    return u, K, b, free


def kresli(coord, elem, u, rohy, zvetseni, ax=None, cmap="viridis"):
    """Deformovaný tvar s barevnou mapou |u_h| (jen rohové uzly prvků)."""
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    U = u.reshape(-1, 2)
    xy = coord + zvetseni * U
    tri = (elem[:, [0, 1, 2]] if rohy == 3 else
           np.vstack([elem[:, [0, 1, 2]], elem[:, [0, 2, 3]]]))
    obr = ax.tripcolor(xy[:, 0], xy[:, 1], tri, np.linalg.norm(U, axis=1),
                       shading="gouraud", cmap=cmap)
    for e in elem[:, :rohy]:
        ax.fill(*xy[e].T, fill=False, lw=0.3, color="k")
    ax.set_aspect("equal")
    ax.set_title(f"|u_h|, zvětšení {zvetseni:g}x")
    return obr
