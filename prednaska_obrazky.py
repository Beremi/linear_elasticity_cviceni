"""Doprovodné obrázky a kreslicí funkce k notebooku cviceni_mkp_pruznost.ipynb.

Notebook obsahuje jen kód potřebný k výpočtu. Ilustrační obrázky (referenční
prvky, bazické funkce, transformace, struktura matic, ...) vyrábí tento skript
do složky obrazky_prednaska/ a notebook je jen vkládá do textu:

    python prednaska_obrazky.py

Funkce kresli_sit, kresli a graf_konvergence notebook importuje a kreslí jimi
sítě a výsledky, které sám spočítá.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import sympy as sp
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import mkp2d
from mkp2d import XI, kresli, xi1, xi2, xi3

OUT = Path(__file__).resolve().parent / "obrazky_prednaska"

REF_UZLY = {
    "P1 (obr. 6.2)": [(0, 0), (1, 0), (0, 1)],
    "P2 (obr. 6.3)": [(0, 0), (1, 0), (0, 1), (.5, .5), (0, .5), (.5, 0)],
    "Q1 (obr. 6.4)": [(-1, -1), (1, -1), (1, 1), (-1, 1)],
    "redukovaný Q2 (obr. 6.5)": [(-1, -1), (1, -1), (1, 1), (-1, 1),
                                 (0, -1), (1, 0), (0, 1), (-1, 0)],
}
g = 1 / np.sqrt(3)
PHI_P1 = sp.Matrix([1 - xi1 - xi2, xi1, xi2])
PHI_Q1 = sp.Matrix([(1 - xi1) * (1 - xi2) / 4, (1 + xi1) * (1 - xi2) / 4,
                    (1 + xi1) * (1 + xi2) / 4, (1 - xi1) * (1 + xi2) / 4])
KVADR_P1 = ([(1 / 3, 1 / 3)], [1 / 2])
KVADR_Q1 = ([(-g, -g), (g, -g), (g, g), (-g, g)], [1, 1, 1, 1])
F = np.array([0.0, -1e-3])


# ------------------------------------------------------------ kreslicí funkce pro notebook
def kresli_sit(ax, coord, elem, gamma0=None, cisla=True, rohy=None, barva="0.4"):
    """Síť: uzly (modře), prvky (červeně), lokální čísla prvku 0 (zeleně), uzly na Gamma_0."""
    rohy = rohy or elem.shape[1]
    for e in elem[:, :rohy]:
        ax.fill(*coord[e].T, fc="none", ec=barva, lw=0.6)
    if gamma0 is not None:
        ax.plot(*coord[gamma0].T, "s", color="tab:red", ms=12 if cisla else 5, zorder=0,
                label=r"uzly na $\Gamma_0$")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    if cisla:
        for T, e in enumerate(elem):
            ax.text(*coord[e[:rohy]].mean(axis=0), str(T), color="tab:red", ha="center", va="center")
        for j, (x, y) in enumerate(coord):
            ax.text(x, y, str(j), color="tab:blue", fontsize=8, ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="tab:blue"))
        stred = coord[elem[0, :rohy]].mean(axis=0)
        for p, j in enumerate(elem[0, :rohy]):
            ax.text(*(0.6 * coord[j] + 0.4 * stred), str(p + 1), color="tab:green", fontsize=9,
                    ha="center", va="center")
    ax.set_aspect("equal")


def graf_konvergence(vysledky, u_ref):
    """vysledky[typ] = [(počet stupňů volnosti, u2), ...]: chyba proti referenční hodnotě."""
    plt.figure(figsize=(5.5, 3.8))
    for typ, data in vysledky.items():
        n, u2 = np.array(data).T
        plt.loglog(n, abs(u2 - u_ref), "o-", label=typ)
    plt.xlabel("počet stupňů volnosti $2n_n$")
    plt.ylabel(r"$|u_2(4,0)-u_2^{\rm ref}|$")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.title("chyba průhybu na zjemňovaných sítích")
    plt.show()


# ------------------------------------------------------------ pomocné funkce pro obrázky
def vycisli(M, body):
    """Matice SymPy v proměnných xi vyčíslená v každém bodě: (body, řádky, sloupce)."""
    body = np.atleast_2d(np.asarray(body, float))
    f = sp.lambdify(XI[:body.shape[1]], M, "numpy")
    return np.array([np.array(f(*b), dtype=float) for b in body])


def body_na_prvku(trojuhelnik, n=24):
    if trojuhelnik:
        s = np.linspace(0, 1, n + 1)
        A, B = np.meshgrid(s, s)
        m = A + B <= 1 + 1e-12
        return np.column_stack([A[m], B[m]])
    s = np.linspace(-1, 1, n + 1)
    A, B = np.meshgrid(s, s)
    return np.column_stack([A.ravel(), B.ravel()])


def kresli_uzly(ax, uzly, znacka=r"\hat N", rohy=None, barva="tab:blue", predpona="", vypln="whitesmoke"):
    N = np.asarray(uzly, float)
    rohy = rohy or (3 if len(N) in (3, 6) else 4)
    ax.fill(*N[:rohy].T, fc=vypln, ec="k")
    for p, (x, y) in enumerate(N):
        ax.plot(x, y, "o", color=barva)
        ax.annotate(rf"${znacka}_{{{predpona}{p + 1}}}$", (x, y), textcoords="offset points",
                    xytext=(5, 5), color=barva)
    ax.set_aspect("equal")


def chi(phi, N, body):
    return vycisli(phi, body)[:, :, 0] @ np.asarray(N, float)


def det_J(phi, N, body):
    d = np.asarray(body).shape[1]
    J = np.einsum("pi,mpj->mij", np.asarray(N, float), vycisli(phi.jacobian(XI[:d]), body))
    return np.linalg.det(J)


def kresli_transformaci(ax, phi, N, n=8, detJ=True, referencni=False):
    N = np.asarray(N, float)
    trojuhelnik = len(N) in (3, 6)
    if detJ:
        body = body_na_prvku(trojuhelnik, 40)
        x, d = chi(phi, N, body), det_J(phi, N, body)
        mez = max(abs(d).max(), 1e-12)
        obr = ax.tripcolor(x[:, 0], x[:, 1], mtri.Triangulation(*body.T).triangles, d,
                           shading="gouraud", cmap="RdBu", norm=TwoSlopeNorm(0, -mez, mez))
        plt.colorbar(obr, ax=ax, shrink=0.8, label=r"$\det J_T$")
    s = np.linspace(0, 1, 40)
    for c in np.linspace(0, 1, n + 1) if trojuhelnik else np.linspace(-1, 1, n + 1):
        if trojuhelnik:
            cary = [np.column_stack([c + 0 * s, (1 - c) * s]), np.column_stack([(1 - c) * s, c + 0 * s])]
        else:
            t = 2 * s - 1
            cary = [np.column_stack([c + 0 * t, t]), np.column_stack([t, c + 0 * t])]
        for cara in cary:
            ax.plot(*chi(phi, N, cara).T, color="k", lw=0.5)
    if referencni:
        kresli_uzly(ax, N, vypln="none")
    else:
        kresli_uzly(ax, N, znacka="N", predpona="T,", barva="tab:red", vypln="none")


def uloz(fig, nazev):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / nazev, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(OUT / nazev)


def gmsh_obdelnik(nx, ny, ctyruhelniky=False):
    """Stejná síť se stejným číslováním uzlů jako sit_obdelnik v notebooku
    (strukturovaná, transfinite; uzly seřazené podle tagů Gmsh)."""
    import gmsh
    gmsh.clear()
    gmsh.model.occ.addRectangle(0, 0, 0, 4, 1)                 # plocha 1, hrany 1 dolní, 2 pravá, 3 horní, 4 levá
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(1, [4], tag=1, name="Gamma0")  # levá hrana = Gamma_0
    for hrana, pocet in [(1, nx), (2, ny), (3, nx), (4, ny)]:
        gmsh.model.mesh.setTransfiniteCurve(hrana, pocet + 1)
    gmsh.model.mesh.setTransfiniteSurface(1)
    if ctyruhelniky:
        gmsh.model.mesh.setRecombine(2, 1)
    gmsh.model.mesh.generate(2)
    tagy, xyz, _ = gmsh.model.mesh.getNodes()
    typ, n_p = (3, 4) if ctyruhelniky else (2, 3)
    _, uzly = gmsh.model.mesh.getElementsByType(typ)
    uzly_g0, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, 1)
    return (xyz.reshape(-1, 3)[np.argsort(tagy), :2],
            uzly.reshape(-1, n_p).astype(int) - 1,
            uzly_g0.astype(int) - 1)


# ------------------------------------------------------------ jednotlivé obrázky
def obr_uloha():
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.add_patch(plt.Rectangle((0, 0), 4, 1, fc="lightsteelblue", ec="k"))
    ax.plot([0, 0], [0, 1], color="tab:red", lw=4, label=r"$\Gamma_0$: $u=0$")
    for y in np.linspace(0, 1, 11):
        ax.plot([-0.1, 0], [y - 0.08, y], color="tab:red", lw=1)
    X, Y = np.meshgrid(np.linspace(0.4, 3.6, 9), [0.35, 0.8])
    ax.quiver(X, Y, 0 * X, -np.ones_like(Y), color="tab:green", scale=22, width=0.004)
    ax.text(3.75, 0.45, "$F$", color="tab:green", fontsize=13)
    ax.text(2, 1.08, r"$\Gamma_1$: $\sigma n=0$", ha="center")
    ax.set(xlim=(-0.35, 4.3), ylim=(-0.2, 1.3), aspect="equal",
           title=r"Modelová úloha: $\Omega=(0,4)\times(0,1)$, $\lambda=\mu=1$, $F=(0,-10^{-3})$")
    ax.legend(loc="lower right", fontsize=8)
    uloz(fig, "01_uloha.png")


def obr_coord_elem():
    """Malá triangulace se čtyřmi trojúhelníky a výpis jejích polí coord a elem."""
    coord = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], float)
    elem = np.array([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]])
    barvy = ["tab:red", "tab:purple", "tab:orange", "tab:brown"]
    fig, (ax, ax_t) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
    for T, e in enumerate(elem):
        ax.fill(*coord[e].T, fc=barvy[T], alpha=0.12, ec="0.3")
        stred = coord[e].mean(axis=0)
        ax.text(*stred, f"T={T}", color=barvy[T], ha="center", va="center", weight="bold")
        for p, j in enumerate(e):                       # lokální čísla 1, 2, 3 uvnitř prvku u vrcholu
            ax.text(*(0.7 * coord[j] + 0.3 * stred), str(p + 1), color=barvy[T], fontsize=9,
                    ha="center", va="center")
    for j, (x, y) in enumerate(coord):
        ax.text(x, y, str(j), color="tab:blue", ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec="tab:blue"))
    ax.set(xlim=(-0.2, 2.2), ylim=(-0.2, 1.2), aspect="equal")
    ax.set_title("uzly (modře, globální čísla) a prvky (lokální čísla 1, 2, 3 proti směru hod. ručiček)",
                 fontsize=9)
    ax_t.axis("off")
    radky = ["coord  (řádek j = souřadnice uzlu j)"]
    radky += [f"  {j}:  [{x:g}, {y:g}]" for j, (x, y) in enumerate(coord)]
    ax_t.text(0, 1, "\n".join(radky), family="monospace", va="top", fontsize=10, transform=ax_t.transAxes)
    ax_t.text(0, 0.42, "elem  (řádek T = uzly N_T,1, N_T,2, N_T,3)", family="monospace", va="top",
              fontsize=10, transform=ax_t.transAxes)
    for T, e in enumerate(elem):
        ax_t.text(0, 0.34 - 0.08 * T, f"  {T}:  {e.tolist()}", family="monospace", va="top", fontsize=10,
                  color=barvy[T], transform=ax_t.transAxes)
    uloz(fig, "01b_coord_elem.png")


def obr_referencni_prvky():
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.3))
    for ax, (nazev, uzly) in zip(axs, REF_UZLY.items()):
        kresli_uzly(ax, uzly)
        ax.set_title(nazev)
        ax.margins(0.25)
    fig.tight_layout()
    uloz(fig, "02_referencni_prvky.png")


def obr_baze(phi, uzly, nazev):
    uzly = np.asarray(uzly, float)
    body = body_na_prvku(len(uzly) in (3, 6))
    hodnoty = vycisli(phi, body)[:, :, 0]
    tri = mtri.Triangulation(*body.T)
    fig = plt.figure(figsize=(3.3 * len(uzly), 3.2))
    for p in range(len(uzly)):
        ax = fig.add_subplot(1, len(uzly), p + 1, projection="3d")
        ax.plot_trisurf(tri, hodnoty[:, p], cmap="viridis", lw=0.1, edgecolor="0.4", alpha=0.9)
        vyska = np.zeros(len(uzly))
        vyska[p] = 1
        ax.scatter(*uzly.T, vyska, color="tab:red", s=25, depthshade=False)
        ax.set_title(rf"$\hat\varphi_{{{p + 1}}}$, rovno 1 v $\hat N_{{{p + 1}}}$")
        ax.set_zlim(-0.15, 1)
        ax.view_init(28, -125)
    fig.tight_layout()
    uloz(fig, nazev)


def obr_transformace():
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    kresli_transformaci(axs[0], PHI_P1, REF_UZLY["P1 (obr. 6.2)"], detJ=False, referencni=True)
    axs[0].set_title(r"$\hat T$ a mřížka $\xi_1,\xi_2=$ konst.")
    kresli_transformaci(axs[1], PHI_P1, [(1.0, 0.5), (3.0, 1.0), (1.6, 2.4)], detJ=False)
    axs[1].set_title(r"$T=\chi_T(\hat T)$, P1: afinní zobrazení")
    fig.subplots_adjust(wspace=0.5)
    fig.text(0.5, 0.52, r"$\longrightarrow$" + "\n" + r"$\chi_T$", fontsize=18, ha="center", va="center")
    uloz(fig, "05_transformace_P1.png")

    pripady = {"konvexní, levotočivě": [(0, 0), (3, 0), (2.5, 2), (0.5, 1.5)],
               "nekonvexní": [(0, 0), (3, 0), (1, 1), (0, 3)],
               "pravotočivě": [(0, 0), (0.5, 1.5), (2.5, 2), (3, 0)]}
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, (nazev, N) in zip(axs, pripady.items()):
        kresli_transformaci(ax, PHI_Q1, N)
        ax.set_title(nazev)
    fig.tight_layout()
    uloz(fig, "06_transformace_Q1.png")


def obr_kvadratura():
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.3))
    for (ax_ref, ax_T), phi, uzly, (body, vahy), N in [
            (axs[:2], PHI_P1, REF_UZLY["P1 (obr. 6.2)"], KVADR_P1, [(1.0, 0.5), (3.0, 1.0), (1.6, 2.4)]),
            (axs[2:], PHI_Q1, REF_UZLY["Q1 (obr. 6.4)"], KVADR_Q1, [(0, 0), (3, 0), (2.5, 2), (0.5, 1.5)])]:
        body = np.array(body)
        kresli_uzly(ax_ref, uzly)
        ax_ref.scatter(*body.T, s=300 * np.array(vahy), color="tab:orange", zorder=3)
        ax_ref.set_title(r"$\hat A_q$, velikost $\sim\hat\omega_q$")
        w_T = np.array(vahy) * np.abs(det_J(phi, N, body))
        kresli_uzly(ax_T, N, znacka="N", predpona="T,", barva="tab:red")
        ax_T.scatter(*chi(phi, N, body).T, s=300 * w_T / max(w_T.max(), 1), color="tab:orange", zorder=3)
        ax_T.set_title(r"$A_{T,q}=\chi_T(\hat A_q)$, $\omega_{T,q}$")
    fig.tight_layout()
    uloz(fig, "07_kvadratura.png")


def obr_globalni_baze():
    fig = plt.figure(figsize=(13, 4))
    for k, typ, (phi, uzly, kv) in [(1, "P1", (PHI_P1, REF_UZLY["P1 (obr. 6.2)"], KVADR_P1)),
                                    (2, "Q1", (PHI_Q1, REF_UZLY["Q1 (obr. 6.4)"], KVADR_Q1))]:
        coord, elem = mkp2d.sit_obdelnik(6, 3, typ, L=3, H=1.5)
        ax = fig.add_subplot(1, 2, k, projection="3d")
        j = 17
        body = body_na_prvku(typ == "P1", 8)
        hodnoty = vycisli(phi, body)[:, :, 0]
        tri = mtri.Triangulation(*body.T).triangles
        for T in range(len(elem)):
            x = hodnoty @ coord[elem[T]]
            obsahuje = j in elem[T]
            z = hodnoty[:, list(elem[T]).index(j)] if obsahuje else 0 * x[:, 0]
            ax.plot_trisurf(x[:, 0], x[:, 1], tri, z, color="tab:orange" if obsahuje else "0.85",
                            lw=0.05, edgecolor="0.5", alpha=0.9)
        ax.set_title(rf"{typ}: globální bazická funkce $\varphi_j$")
        ax.view_init(35, -60)
        ax.set_zlim(0, 1.2)
    fig.tight_layout()
    uloz(fig, "08_globalni_baze.png")


def obr_matice(coord, elem):
    ref = mkp2d.referencni_prvek(PHI_P1, REF_UZLY["P1 (obr. 6.2)"], *KVADR_P1)
    dphi, w = mkp2d.geometrie(ref, coord, elem)
    T = 0
    B0 = np.zeros((3, 6))
    B0[0, 0::2], B0[1, 1::2], B0[2, 0::2], B0[2, 1::2] = dphi[T, 0, :, 0], dphi[T, 0, :, 1], dphi[T, 0, :, 1], dphi[T, 0, :, 0]
    C = np.array([[3.0, 1, 0], [1, 3, 0], [0, 0, 1]])
    K0 = w[T, 0] * B0.T @ C @ B0
    sloupce = [rf"$v_{i}(N_{{T,{p}}})$" for p in (1, 2, 3) for i in (1, 2)]

    fig, ax = plt.subplots(figsize=(7, 2.4))
    ax.imshow(B0, cmap="RdBu", vmin=-4, vmax=4)
    for (r, s), v in np.ndenumerate(B0):
        ax.text(s, r, f"{v:g}", ha="center", va="center")
    ax.set_xticks(range(6), sloupce)
    ax.set_yticks(range(3), [r"$\varepsilon_{11}$", r"$\varepsilon_{22}$", r"$2\varepsilon_{12}$"])
    ax.set_title(r"$B_{T,q}$ pro prvek $T=0$ sítě 8×2")
    uloz(fig, "09_matice_B.png")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(K0, cmap="RdBu", vmin=-abs(K0).max(), vmax=abs(K0).max())
    for (r, s), v in np.ndenumerate(K0):
        ax.text(s, r, f"{v:.2g}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(6), sloupce, rotation=60)
    ax.set_yticks(range(6), sloupce)
    ax.set_title(r"lokální matice $K_T$, $T=0$ (symetrická)")
    uloz(fig, "10_lokalni_K.png")

    Kt = mkp2d.matice_tuhosti(ref, coord, elem)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.spy(Kt, markersize=2, color="0.6")
    dofs5 = mkp2d.stupne_volnosti(elem)[5]
    I, Jm = np.meshgrid(dofs5, dofs5)
    ax.plot(Jm.ravel(), I.ravel(), "s", color="tab:red", ms=3, label=r"$R_T^\top K_T R_T$, $T=5$")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(r"struktura $\tilde K$")
    uloz(fig, "11_sestaveni.png")

    tuhe = {"posun v $x_1$": (1 + 0 * coord[:, 0], 0 * coord[:, 0]),
            "posun v $x_2$": (0 * coord[:, 0], 1 + 0 * coord[:, 0]),
            "otočení": (-(coord[:, 1] - 0.5), coord[:, 0] - 2)}
    fig, axs = plt.subplots(1, 3, figsize=(14, 2.6))
    for ax, (nazev, (r1, r2)) in zip(axs, tuhe.items()):
        r = np.column_stack([r1, r2]).ravel()
        kresli_sit(ax, coord, elem, cisla=False, barva="tab:blue")
        kresli_sit(ax, coord + 0.25 * r.reshape(-1, 2), elem, cisla=False, barva="tab:red")
        ax.set_title(rf"{nazev}: $\|\tilde K r\|$ = {np.abs(Kt @ r).max():.1e}")
    axs[0].plot([], [], color="tab:blue", label="původní síť")
    axs[0].plot([], [], color="tab:red", label=r"posunutá síť $x+0{,}25\,r(x)$")
    fig.legend(loc="upper center", ncol=2, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    uloz(fig, "13_tuhe_pohyby.png")

    bt = mkp2d.vektor_sil(ref, coord, elem, F)
    fig, ax = plt.subplots(figsize=(10, 3))
    kresli_sit(ax, coord, elem, cisla=False)
    ax.quiver(*coord.T, *bt.reshape(-1, 2).T, color="tab:green", angles="xy", scale_units="xy",
              scale=6e-4, width=0.004)
    ax.set_title(r"uzlové síly $\tilde b$: vnitřní uzly dostanou víc než okrajové (sdílejí víc prvků)")
    uloz(fig, "14_uzlove_sily.png")


def obr_dirichlet(coord, elem, gamma0):
    ref = mkp2d.referencni_prvek(PHI_P1, REF_UZLY["P1 (obr. 6.2)"], *KVADR_P1)
    I_D = np.setdiff1d(np.arange(len(coord)), gamma0)
    volne = np.column_stack([2 * I_D, 2 * I_D + 1]).ravel()
    M = mkp2d.matice_tuhosti(ref, coord, elem).toarray()
    maska = np.ones_like(M, dtype=bool)
    maska[np.ix_(volne, volne)] = False
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.3))
    axs[0].imshow(np.ma.masked_where(M == 0, M), cmap="RdBu", vmin=-2, vmax=2)
    axs[0].imshow(np.ma.masked_where(~maska, maska), cmap="Greys", alpha=0.6, vmin=0, vmax=1.5)
    axs[0].set_title(r"$\tilde K$: šedě vynechané řádky a sloupce ($\Gamma_0$)")
    K = M[np.ix_(volne, volne)]
    axs[1].imshow(np.ma.masked_where(K == 0, K), cmap="RdBu", vmin=-2, vmax=2)
    axs[1].set_title(r"$K$ (stupně volnosti uzlů z $I_D$), regulární")
    uloz(fig, "15_dirichlet.png")


def obr_poradi_P2():
    import gmsh
    gmsh.clear()
    r = gmsh.model.occ.addRectangle(0, 0, 0, 4, 1)
    d = gmsh.model.occ.addDisk(2, 0.5, 0, 0.25, 0.25)
    gmsh.model.occ.cut([(2, r)], [(2, d)])
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.3)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)
    c2, e2, _ = mkp2d.z_gmsh()
    P2_gmsh = e2["Triangle 6"]
    P2_text = P2_gmsh[:, [0, 1, 2, 4, 5, 3]]
    stred = np.array([2, 0.5])
    T = np.argmin(np.linalg.norm(c2[P2_gmsh[:, 3:]].mean(axis=1) - stred, axis=1))
    t = np.linspace(0, 1, 30)[:, None]

    def hrana(ax, A, Bp, Mp, **kw):
        ax.plot(*((1 - t) * (1 - 2 * t) * A + 4 * t * (1 - t) * Mp + t * (2 * t - 1) * Bp).T, **kw)

    hrany = [(0, 1, 5), (1, 2, 3), (2, 0, 4)]
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.6))
    for ax, pole, nazev in [(axs[0], P2_gmsh, "pořadí Gmsh"), (axs[1], P2_text, "pořadí učebního textu (obr. 6.3)")]:
        for a, b, m in hrany:
            hrana(ax, *c2[P2_text[T, [a, b, m]]], color="k")
        for p, j in enumerate(pole[T]):
            ax.plot(*c2[j], "o", color="tab:blue" if p < 3 else "tab:orange")
            ax.annotate(str(p + 1), c2[j], textcoords="offset points", xytext=(5, 4))
        ax.set_aspect("equal")
        ax.set_title(nazev)
    for e in P2_text:
        for a, b, m in hrany:
            hrana(axs[2], *c2[e[[a, b, m]]], color="k", lw=0.5)
    axs[2].add_patch(plt.Circle(stred, 0.25, fill=False, color="tab:red", ls="--"))
    axs[2].plot(*c2[np.unique(P2_text[:, 3:])].T, ".", color="tab:orange", ms=3)
    axs[2].set(xlim=(1.5, 2.5), ylim=(0, 1), aspect="equal", title="zakřivené P2 hrany u otvoru")
    fig.tight_layout()
    uloz(fig, "16_poradi_P2.png")


def kresli_plochy(ax, coord, plochy, barva):
    ax.add_collection3d(Poly3DCollection(coord[plochy], fc=barva, ec="k", lw=0.2, alpha=0.9))
    ax.set_box_aspect(np.ptp(coord, axis=0))
    for i, osa in enumerate("xyz"):
        getattr(ax, f"set_{osa}lim")(coord[:, i].min(), coord[:, i].max())


def obr_3D():
    import gmsh
    gmsh.clear()
    gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, 2.0, 0.5)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.2)
    gmsh.model.mesh.generate(3)
    coord, elem, _ = mkp2d.z_gmsh(dim=3)
    tetra = elem["Tetrahedron 4"]
    polovina = tetra[coord[tetra].mean(axis=1)[:, 1] < 0]
    steny = np.sort(polovina[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]].reshape(-1, 3), axis=1)
    steny, pocet = np.unique(steny, axis=0, return_counts=True)
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    kresli_plochy(ax, coord, elem["Triangle 3"], "lightsteelblue")
    ax.set_title("povrchová síť válce")
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    kresli_plochy(ax, coord, steny[pocet == 1], "wheat")
    ax.view_init(20, 60)
    ax.set_title("řez: čtyřstěny s těžištěm $x_2<0$")
    uloz(fig, "17_valec_ctyrsteny.png")

    gmsh.clear()
    box = gmsh.model.occ.addBox(0, 0, 0, 4, 1, 1)
    gmsh.model.occ.synchronize()
    for _, t in gmsh.model.getEntities(1):
        gmsh.model.mesh.setTransfiniteCurve(t, int(round(gmsh.model.occ.getMass(1, t) / 0.25)) + 1)
    for _, s in gmsh.model.getEntities(2):
        gmsh.model.mesh.setTransfiniteSurface(s)
        gmsh.model.mesh.setRecombine(2, s)
    gmsh.model.mesh.setTransfiniteVolume(box)
    gmsh.model.mesh.generate(3)
    coord, elem, _ = mkp2d.z_gmsh(dim=3)
    hexa = elem["Hexahedron 8"]
    fig = plt.figure(figsize=(12, 4.5))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    N8 = np.array([(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                   (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], float)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
        ax.plot(*N8[[a, b]].T, "k", lw=1)
    for p, x in enumerate(N8):
        ax.text(*(1.15 * x), str(p + 1), color="tab:blue")
    ax.set_title(r"referenční šestistěn $(-1,1)^3$")
    ax.set_box_aspect((1, 1, 1))
    steny = hexa[:, [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]].reshape(-1, 4)
    _, idx, pocet = np.unique(np.sort(steny, axis=1), axis=0, return_index=True, return_counts=True)
    ax = fig.add_subplot(1, 3, (2, 3), projection="3d")
    kresli_plochy(ax, coord, steny[idx[pocet == 1]], "lightsteelblue")
    ax.set_title("hranol: povrch šestistěnové sítě")
    fig.tight_layout()
    uloz(fig, "18_hranol_sestisteny.png")


def main():
    import gmsh
    plt.rcParams["figure.dpi"] = 100
    obr_uloha()
    obr_coord_elem()
    obr_referencni_prvky()
    obr_baze(PHI_P1, REF_UZLY["P1 (obr. 6.2)"], "03_baze_P1.png")
    obr_baze(PHI_Q1, REF_UZLY["Q1 (obr. 6.4)"], "04_baze_Q1.png")
    obr_transformace()
    obr_kvadratura()
    obr_globalni_baze()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    coord, elem, _ = gmsh_obdelnik(8, 2)
    obr_matice(coord, elem)
    obr_dirichlet(*gmsh_obdelnik(4, 1))
    obr_poradi_P2()
    obr_3D()
    gmsh.finalize()


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()
