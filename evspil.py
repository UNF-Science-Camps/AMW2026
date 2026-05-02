"""
evspil.py - Modul til Evolutionær Spilteori

Indeholder funktioner til:
  * Ørn-due-spillet og lignende 2-strategi-spil
  * Replicator-dynamik (med konstant og tidsvarierende V)
  * Spil med flere strategier (ørn-due-hævner, firben/sten-saks-papir)
  * Individ-baseret simulering med fødsel, død og mutation
  * Rovdyr-bytte-simulering
  * Gittermodel (rumlig simulering)
  * Sæsonbaseret økosystem og miljøskift (projekter)

Hurtig tilstand:
  Sæt `evspil.HURTIG_TILSTAND = False` for at bruge en langsom referenceimplementering
  af gittermodellen (god til at se hvordan den virker). Default er True.
"""

import numpy as np
import matplotlib.pyplot as plt

# Global toggle: hvis True bruges JIT-kompileret/parallelliseret kode under hætten;
# hvis False bruges en ren Python-implementering. Resultaterne er de samme.
HURTIG_TILSTAND = True

try:
    from numba import njit as _njit, prange as _prange
    _HAR_JIT = True
except ImportError:
    _HAR_JIT = False
    def _njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def decorator(f):
            return f
        return decorator
    _prange = range


# ====================================================================
# ØRN-DUE-SPILLET
# ====================================================================

def gevinst_orn_due(V, C):
    """Returnerer 2x2 gevinsttabellen for ørn-due-spillet.
    Række 0 = ørn, række 1 = due. Søjle 0 = ørn, søjle 1 = due.
    """
    return np.array([
        [V/2 - C/2, V],
        [0,         V/2]
    ])


def E_orn(p, V, C):
    """Forventet gevinst for en ørn, når andelen af ørne er p."""
    return p * (V/2 - C/2) + (1 - p) * V


def E_due(p, V, C):
    """Forventet gevinst for en due, når andelen af ørne er p."""
    return p * 0 + (1 - p) * V/2


def E_gennemsnit(p, V, C):
    """Gennemsnitlig gevinst i en population med ørne-andel p."""
    return p * E_orn(p, V, C) + (1 - p) * E_due(p, V, C)


def ess_orn_due(V, C):
    """Den evolutionært stabile ørne-andel p* = V/C (afkortet til [0,1])."""
    return min(V / C, 1.0)


# ====================================================================
# GENERELT 2-STRATEGI-SPIL (forsvarer/vandrer, jæger/lurepasser, ...)
# ====================================================================

def forventet_gevinst_2(p, M):
    """Forventet gevinst for hver af to strategier i et 2x2-spil.

    p: andel af strategi A (række 0).
    M: 2x2 gevinstmatrix, hvor M[i,j] er gevinsten for strategi i mod j.

    Returnerer (E_A, E_B).
    """
    M = np.asarray(M, dtype=float)
    E_A = p * M[0, 0] + (1 - p) * M[0, 1]
    E_B = p * M[1, 0] + (1 - p) * M[1, 1]
    return E_A, E_B


# ====================================================================
# PLOT-FUNKTIONER
# ====================================================================

def plot_gevinster(V, C, ax=None):
    """Plot E_orn(p) og E_due(p) som funktion af p, og marker krydset (ESS)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    p = np.linspace(0, 1, 200)
    ax.plot(p, E_orn(p, V, C), label="E_ørn(p)", linewidth=2)
    ax.plot(p, E_due(p, V, C), label="E_due(p)", linewidth=2)
    p_star = V / C
    if 0 < p_star < 1:
        ax.axvline(p_star, color="gray", linestyle="--",
                   label=f"p* = V/C = {p_star:.2f}")
        ax.plot([p_star], [E_orn(p_star, V, C)], "ko")
    ax.set_xlabel("Andel af ørne, p")
    ax.set_ylabel("Forventet gevinst")
    ax.set_title(f"Forventet gevinst (V={V}, C={C})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_gevinster_egne_formler(E_orn_funktion, E_due_funktion, V, C, ax=None):
    """Plot E_orn(p) og E_due(p) ud fra dine egne formler.

    E_orn_funktion og E_due_funktion skal hver tage (p, V, C) og returnere et tal.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    p = np.linspace(0, 1, 200)
    ax.plot(p, E_orn_funktion(p, V, C), label="E_ørn(p) (din formel)", linewidth=2)
    ax.plot(p, E_due_funktion(p, V, C), label="E_due(p) (din formel)", linewidth=2)
    ax.set_xlabel("Andel af ørne, p")
    ax.set_ylabel("Forventet gevinst")
    ax.set_title(f"Dine egne formler (V={V}, C={C})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_2d_favorabilitet(V=10, C_min=2, C_max=40, n=200):
    """2D-plot der viser, for hvilke (p, C/V)-kombinationer ørn er favorabel.

    Farve viser E_orn - E_due. Den grå streg er ligevægten p* = V/C.
    """
    p_vals = np.linspace(0, 1, n)
    C_vals = np.linspace(C_min, C_max, n)
    P, Cm = np.meshgrid(p_vals, C_vals)
    diff = E_orn(P, V, Cm) - E_due(P, V, Cm)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(p_vals, C_vals / V, diff, cmap="RdBu_r",
                       vmin=-abs(diff).max(), vmax=abs(diff).max(), shading="auto")
    ax.contour(p_vals, C_vals / V, diff, levels=[0], colors="black", linewidths=2)
    ax.plot(p_vals, 1 / p_vals, "k--", alpha=0.6, label="p = V/C (ligevægt)")
    ax.set_ylim(C_min / V, C_max / V)
    ax.set_xlabel("Andel af ørne, p")
    ax.set_ylabel("C / V")
    ax.set_title("E_ørn − E_due  (rød: ørn er bedst, blå: due er bedst)")
    plt.colorbar(im, ax=ax, label="E_ørn − E_due")
    ax.legend(loc="upper right")
    return ax


# ====================================================================
# REPLICATOR-DYNAMIK
# ====================================================================

def replicator_skridt(p, V, C):
    """Et enkelt skridt af replicator-dynamikken (præcist som i teksten):
    p_{t+1} = p_t * E_orn(p_t) / E_gennemsnit(p_t).

    Bemærk: Når gevinsterne er negative (f.eks. V=10, C=20, p stor) kan denne formel
    give p > 1 eller p < 0. Det er en kendt begrænsning ved den diskrete replicator.
    Brug `simuler_replicator` for stabile langtidsplots.
    """
    E_gns = E_gennemsnit(p, V, C)
    if E_gns == 0:
        return p
    return p * E_orn(p, V, C) / E_gns


def _replicator_skridt_stabil(p, V, C):
    """Stabil version af replicator-skridt: skifter gevinster så de er positive.

    Replicator-dynamikken er matematisk invariant under additiv shift, så ligevægten
    p* = V/C er den samme. Resultatet holdes i [0, 1].
    """
    M = gevinst_orn_due(V, C)
    shift = max(0.0, -float(M.min()) + 1.0)
    a = (V/2 - C/2) + shift
    b = V + shift
    c = 0 + shift
    d = V/2 + shift
    E_h = p * a + (1 - p) * b
    E_d = p * c + (1 - p) * d
    E_avg = p * E_h + (1 - p) * E_d
    if E_avg <= 0:
        return p
    p_ny = p * E_h / E_avg
    return float(np.clip(p_ny, 0.0, 1.0))


def simuler_replicator(p0, V, C, n_generationer):
    """Simulerer replicator-dynamikken med konstant V og C (numerisk stabil version).

    Returnerer et array med p_0, p_1, ..., p_n. Ligevægten er stadig p* = V/C.
    """
    p = np.zeros(n_generationer + 1)
    p[0] = p0
    for t in range(n_generationer):
        p[t + 1] = _replicator_skridt_stabil(p[t], V, C)
    return p


def simuler_replicator_V_t(p0, V_funktion, C, n_generationer):
    """Simulerer replicator-dynamikken med tidsvarierende V(t).

    V_funktion: funktion af t (heltal) der returnerer V på det tidspunkt.
    Returnerer to arrays: (p_historik, V_historik).
    """
    p = np.zeros(n_generationer + 1)
    V_hist = np.zeros(n_generationer + 1)
    p[0] = p0
    V_hist[0] = V_funktion(0)
    for t in range(n_generationer):
        V_t = V_funktion(t)
        V_hist[t] = V_t
        p[t + 1] = _replicator_skridt_stabil(p[t], V_t, C)
    V_hist[-1] = V_funktion(n_generationer)
    return p, V_hist


# ====================================================================
# SPIL MED FLERE STRATEGIER (n x n)
# ====================================================================

def gevinst_orn_due_haevner(V, C):
    """3x3-gevinstmatrix for ørn (0), due (1), hævner (2)."""
    return np.array([
        [V/2 - C/2, V,        V/2 - C/2],
        [0,         V/2,      V/2      ],
        [V/2 - C/2, V/2,      V/2      ]
    ])


def gevinst_firben(W=1.0):
    """3x3-gevinstmatrix for orange (0), blå (1), gul (2) - sten-saks-papir-cyklus.

    Orange slår blå, blå slår gul, gul slår orange.
    """
    return np.array([
        [0,  W, -W],
        [-W, 0,  W],
        [W, -W,  0]
    ])


def forventet_gevinst_n(p, M):
    """Forventet gevinst for hver strategi i et n-strategi-spil.

    p: array af andele (skal summere til 1).
    M: n x n gevinstmatrix.

    Returnerer et array med E_i for hver strategi i.
    """
    return np.asarray(M, dtype=float) @ np.asarray(p, dtype=float)


def replicator_skridt_n(p, M):
    """Et skridt af replicator-dynamikken med n strategier.

    Bemærk: Med negative gevinster (eller nul-sum-spil som firben) kan den naive
    formel p * E / E_gns give E_gns = 0 eller negative værdier. Brug
    `_replicator_skridt_n_stabil` til simuleringer.
    """
    p = np.asarray(p, dtype=float)
    E = forventet_gevinst_n(p, M)
    E_gns = float(p @ E)
    if E_gns == 0:
        return p
    return p * E / E_gns


def _replicator_skridt_n_stabil(p, M):
    """Stabil version af n-strategi replicator-skridt.

    Skifter gevinstmatricen så alle entries er positive. Replicator-dynamikken har
    samme fikspunkter under additiv shift, så ESS er den samme. Resultatet
    normaliseres til en gyldig sandsynlighedsfordeling.
    """
    M = np.asarray(M, dtype=float)
    p = np.asarray(p, dtype=float)
    shift = max(0.0, -float(M.min()) + 1.0)
    M_shifted = M + shift
    E = M_shifted @ p
    E_gns = float(p @ E)
    if E_gns <= 0:
        return p
    p_ny = p * E / E_gns
    p_ny = np.clip(p_ny, 0.0, None)
    s = p_ny.sum()
    if s > 0:
        p_ny = p_ny / s
    return p_ny


def simuler_replicator_n(p0, M, n_generationer):
    """Simulerer replicator-dynamikken med n strategier (numerisk stabil version).

    Returnerer et (n_generationer+1) x n array med andelene over tid.
    """
    p0 = np.asarray(p0, dtype=float)
    n = len(p0)
    historik = np.zeros((n_generationer + 1, n))
    historik[0] = p0
    for t in range(n_generationer):
        historik[t + 1] = _replicator_skridt_n_stabil(historik[t], M)
    return historik


# ====================================================================
# INDIVID-BASERET SIMULERING
# ====================================================================

# Strategi-koder: 0 = ørn, 1 = due

def simuler_individer(N=200, andel_orn=0.5, V=10, C=20, n_skridt=200,
                      E_start=10.0, E_doed=0.0, E_foedsel=15.0,
                      doedsrate=0.0, mutationsrate=0.0,
                      kapacitet=None, seed=None):
    """Individ-baseret simulering af ørn-due-spillet.

    Hvert tidsskridt:
      1. Hvert individ parres tilfældigt med et andet og spiller spillet.
      2. Individer med energi > E_foedsel reproducerer (afkommet får E_start).
      3. Individer med energi <= E_doed dør.
      4. Hvert individ har sandsynlighed `doedsrate` for at dø af baggrundsårsager.
      5. Ved reproduktion er der `mutationsrate` chance for at afkommet skifter strategi.
      6. Hvis `kapacitet` er angivet, er reproduktion tæthedsafhængig: chancen for at en
         berettiget reproduktion lykkes er max(0, 1 − N/kapacitet). Det forhindrer
         eksponentiel vækst og giver en naturlig populationsstørrelse omkring kapaciteten.

    Returnerer en dict med:
      'andel_orn': array, andel ørne ved hvert skridt
      'befolkning': array, total population ved hvert skridt
      'antal_orn': array, antal ørne ved hvert skridt
      'antal_due': array, antal duer ved hvert skridt
    """
    if kapacitet is None:
        kapacitet = N
    rng = np.random.default_rng(seed)
    G = gevinst_orn_due(V, C)

    n_orn = int(round(N * andel_orn))
    n_due = N - n_orn
    strategier = np.array([0] * n_orn + [1] * n_due, dtype=np.int8)
    energi = np.full(len(strategier), E_start, dtype=float)

    andel_hist = [n_orn / N if N > 0 else 0.0]
    pop_hist = [N]
    antal_orn_hist = [n_orn]
    antal_due_hist = [n_due]

    for _ in range(n_skridt):
        n = len(strategier)
        if n < 2:
            andel_hist.append(0.0 if n == 0 else float(strategier[0] == 0))
            pop_hist.append(n)
            antal_orn_hist.append(int(np.sum(strategier == 0)))
            antal_due_hist.append(int(np.sum(strategier == 1)))
            continue

        # 1) Parring og spil
        order = rng.permutation(n)
        a = order[0:n - 1:2]
        b = order[1:n:2]
        sa = strategier[a]
        sb = strategier[b]
        energi[a] += G[sa, sb]
        energi[b] += G[sb, sa]

        # 2) Reproduktion
        kan_reproducere = energi > E_foedsel
        n_nye = int(np.sum(kan_reproducere))
        if n_nye > 0:
            nye_strategier = strategier[kan_reproducere].copy()
            if mutationsrate > 0:
                muter = rng.random(n_nye) < mutationsrate
                nye_strategier[muter] = 1 - nye_strategier[muter]
            energi[kan_reproducere] -= E_start
            strategier = np.concatenate([strategier, nye_strategier])
            energi = np.concatenate([energi,
                                     np.full(n_nye, E_start, dtype=float)])

        # 3) Død (energi for lav)
        levende = energi > E_doed
        # 4) Baggrundsdødelighed
        if doedsrate > 0:
            overlever_baggrund = rng.random(len(strategier)) >= doedsrate
            levende = levende & overlever_baggrund
        strategier = strategier[levende]
        energi = energi[levende]

        # 5) Bærekapacitet: tilfældig udtynding hvis populationen overskrider kapacitet
        if kapacitet is not None and len(strategier) > kapacitet:
            keep = rng.choice(len(strategier), size=kapacitet, replace=False)
            strategier = strategier[keep]
            energi = energi[keep]

        n = len(strategier)
        n_orn = int(np.sum(strategier == 0))
        n_due = n - n_orn
        andel_hist.append(n_orn / n if n > 0 else 0.0)
        pop_hist.append(n)
        antal_orn_hist.append(n_orn)
        antal_due_hist.append(n_due)

    return {
        "andel_orn": np.array(andel_hist),
        "befolkning": np.array(pop_hist),
        "antal_orn": np.array(antal_orn_hist),
        "antal_due": np.array(antal_due_hist),
    }


def plot_individ_simulering(resultat, V=None, C=None, vis_ess=True, ax=None):
    """Plot resultat fra simuler_individer: andel ørne over tid.

    Hvis V og C angives, og vis_ess=True, tegnes p* = V/C som vandret linje.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    t = np.arange(len(resultat["andel_orn"]))
    ax.plot(t, resultat["andel_orn"], label="Andel ørne")
    if vis_ess and V is not None and C is not None and C > 0:
        p_star = min(V / C, 1.0)
        ax.axhline(p_star, color="red", linestyle="--",
                   label=f"p* = V/C = {p_star:.2f}")
    ax.set_xlabel("Tidsskridt")
    ax.set_ylabel("Andel ørne")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Individ-baseret simulering")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


# ====================================================================
# ROVDYR-BYTTE-SIMULERING
# ====================================================================

# Rovdyr-strategier: 0 = jæger, 1 = lurepasser/lejlighedsjæger

def simuler_rovdyr_bytte(N_rovdyr=20, andel_jaeger=0.5, N_bytte=500,
                         n_skridt=300,
                         jaeger_p_fang=0.8, jaeger_omkost=3.0,
                         lurer_p_fang=0.3, lurer_omkost=1.0,
                         E_per_bytte=6.0, bytte_vaekst=0.3,
                         bytte_kapacitet=2000,
                         halv_maetning=500,
                         E_start=10.0, E_doed=0.0, E_foedsel=18.0,
                         rovdyr_kapacitet=200,
                         doedsrate=0.05, mutationsrate=0.0, seed=None):
    """Simulerer rovdyr-bytte-systemet med jæger- og lurepasser-strategier.

    Hvert tidsskridt:
      1. Hvert rovdyr forsøger at fange et bytte. Faktiske fangstchance er
         p_fang * n_bytte / (n_bytte + halv_maetning) (Holling Type II respons).
      2. Hvert rovdyr betaler sin energi-omkostning. Hvis det fanger bytte,
         får det E_per_bytte energi, og byttet fjernes fra bytte-populationen.
      3. Bytte-populationen vokser logistisk: n_bytte += vaekst * n_bytte * (1 - n_bytte/kapacitet).
      4. Rovdyr med energi > E_foedsel reproducerer (mutation muligt).
      5. Rovdyr med energi <= E_doed eller ramt af baggrundsdødelighed dør.
      6. Hvis rovdyr-population overskrider rovdyr_kapacitet, sker tilfældig udtynding.

    Returnerer dict med 'antal_rovdyr', 'antal_bytte', 'andel_jaeger'.
    """
    rng = np.random.default_rng(seed)

    n_jaeger = int(round(N_rovdyr * andel_jaeger))
    n_lurer = N_rovdyr - n_jaeger
    strategier = np.array([0] * n_jaeger + [1] * n_lurer, dtype=np.int8)
    energi = np.full(len(strategier), E_start, dtype=float)
    n_bytte = N_bytte

    rovdyr_hist = [len(strategier)]
    bytte_hist = [n_bytte]
    jaeger_hist = [n_jaeger / len(strategier) if len(strategier) > 0 else 0.0]

    for _ in range(n_skridt):
        n = len(strategier)
        if n > 0:
            # 1) Hvert rovdyr forsøger at fange bytte
            er_jaeger = strategier == 0
            # Holling Type II: når bytte er knapt, falder fangstchancen
            metning = n_bytte / (n_bytte + halv_maetning) if n_bytte > 0 else 0.0
            p_fang = np.where(er_jaeger, jaeger_p_fang, lurer_p_fang) * metning
            omkost = np.where(er_jaeger, jaeger_omkost, lurer_omkost)
            energi -= omkost
            forsoeg = rng.random(n) < p_fang
            order = rng.permutation(n)
            for idx in order:
                if forsoeg[idx] and n_bytte > 0:
                    energi[idx] += E_per_bytte
                    n_bytte -= 1

        # 2) Bytte vokser logistisk
        if bytte_kapacitet is not None and bytte_kapacitet > 0:
            tilvaekst = bytte_vaekst * n_bytte * (1 - n_bytte / bytte_kapacitet)
            n_bytte = max(0, int(round(n_bytte + tilvaekst)))
        elif n_bytte > 0:
            n_bytte = int(round(n_bytte * (1 + bytte_vaekst)))

        # 3) Reproduktion
        if n > 0:
            kan_reproducere = energi > E_foedsel
            n_nye = int(np.sum(kan_reproducere))
            if n_nye > 0:
                nye_strategier = strategier[kan_reproducere].copy()
                if mutationsrate > 0:
                    muter = rng.random(n_nye) < mutationsrate
                    nye_strategier[muter] = 1 - nye_strategier[muter]
                energi[kan_reproducere] -= E_start
                strategier = np.concatenate([strategier, nye_strategier])
                energi = np.concatenate([energi,
                                         np.full(n_nye, E_start, dtype=float)])

            # 4) Død (energi for lav + baggrundsdødelighed)
            levende = energi > E_doed
            if doedsrate > 0:
                overlever = rng.random(len(strategier)) >= doedsrate
                levende = levende & overlever
            strategier = strategier[levende]
            energi = energi[levende]

            # 5) Bærekapacitet for rovdyr (tilfældig udtynding)
            if rovdyr_kapacitet is not None and len(strategier) > rovdyr_kapacitet:
                keep = rng.choice(len(strategier), size=rovdyr_kapacitet, replace=False)
                strategier = strategier[keep]
                energi = energi[keep]

        n = len(strategier)
        n_jaeger = int(np.sum(strategier == 0))
        rovdyr_hist.append(n)
        bytte_hist.append(n_bytte)
        jaeger_hist.append(n_jaeger / n if n > 0 else 0.0)

    return {
        "antal_rovdyr": np.array(rovdyr_hist),
        "antal_bytte": np.array(bytte_hist),
        "andel_jaeger": np.array(jaeger_hist),
    }


def plot_rovdyr_bytte(resultat):
    """Plot tre grafer for resultatet fra simuler_rovdyr_bytte."""
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    t = np.arange(len(resultat["antal_rovdyr"]))
    axes[0].plot(t, resultat["antal_rovdyr"], color="darkred")
    axes[0].set_ylabel("Antal rovdyr")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, resultat["antal_bytte"], color="seagreen")
    axes[1].set_ylabel("Antal bytte")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(t, resultat["andel_jaeger"], color="navy")
    axes[2].set_ylabel("Andel jægere")
    axes[2].set_xlabel("Tidsskridt")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(True, alpha=0.3)
    fig.suptitle("Rovdyr-bytte-system")
    fig.tight_layout()
    return axes


# ====================================================================
# PROJEKT 1: GITTERMODEL (rumlig simulering)
# ====================================================================
#
# I gittermodellen bor hvert dyr på et felt i et L x L gitter. Det spiller
# kun spillet med sine 4 (eller 8) nærmeste naboer. Hvert tidsskridt:
#   1) Hvert felt akkumulerer en samlet gevinst fra spil mod alle sine naboer.
#   2) Hvert felt kigger på sin bedste nabo. Hvis nabopen har højere gevinst,
#      kopierer feltet naboens strategi med sandsynlighed (G_nabo - G_selv) / D_max,
#      hvor D_max er den størst mulige gevinstforskel.
# Opdateringen er synkron: vi bruger de gamle strategier til at beregne alle
# nye strategier, så rækkefølgen ikke har betydning.

# Naboskaber (kan bruges af både den langsomme og den hurtige version)
NABOER_4 = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)], dtype=np.int64)
NABOER_8 = np.array([(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)
                     if not (di == 0 and dj == 0)], dtype=np.int64)


def _init_gitter(L, andel_orn, rng):
    """Lav et L x L gitter, hvor 0 = ørn og 1 = due."""
    return np.where(rng.random((L, L)) < andel_orn, 0, 1).astype(np.int8)


def _gitter_skridt_python(grid, G, naboer_arr, D_max, random_vals):
    """Ét skridt af gittersimuleringen — ren Python, langsom men let at læse."""
    L = grid.shape[0]
    n_nb = naboer_arr.shape[0]
    payoffs = np.zeros((L, L), dtype=float)
    new_grid = grid.copy()

    for i in range(L):
        for j in range(L):
            s = grid[i, j]
            tot = 0.0
            for k in range(n_nb):
                ni = (i + naboer_arr[k, 0]) % L
                nj = (j + naboer_arr[k, 1]) % L
                tot += G[s, grid[ni, nj]]
            payoffs[i, j] = tot

    for i in range(L):
        for j in range(L):
            best_p = payoffs[i, j]
            best_s = grid[i, j]
            for k in range(n_nb):
                ni = (i + naboer_arr[k, 0]) % L
                nj = (j + naboer_arr[k, 1]) % L
                if payoffs[ni, nj] > best_p:
                    best_p = payoffs[ni, nj]
                    best_s = grid[ni, nj]
            if best_s != grid[i, j]:
                p_copy = (best_p - payoffs[i, j]) / D_max
                if random_vals[i, j] < p_copy:
                    new_grid[i, j] = best_s
    return new_grid


@_njit(cache=True, parallel=True)
def _gitter_skridt_jit(grid, payoff_matrix, naboer_arr, D_max, random_vals):
    """Ét skridt af gittersimuleringen — JIT-kompileret og parallelliseret."""
    L = grid.shape[0]
    n_nb = naboer_arr.shape[0]
    payoffs = np.zeros((L, L), dtype=np.float64)
    new_grid = grid.copy()

    for i in _prange(L):
        for j in range(L):
            s = grid[i, j]
            tot = 0.0
            for k in range(n_nb):
                ni = (i + naboer_arr[k, 0]) % L
                nj = (j + naboer_arr[k, 1]) % L
                tot += payoff_matrix[s, grid[ni, nj]]
            payoffs[i, j] = tot

    for i in _prange(L):
        for j in range(L):
            best_p = payoffs[i, j]
            best_s = grid[i, j]
            for k in range(n_nb):
                ni = (i + naboer_arr[k, 0]) % L
                nj = (j + naboer_arr[k, 1]) % L
                if payoffs[ni, nj] > best_p:
                    best_p = payoffs[ni, nj]
                    best_s = grid[ni, nj]
            if best_s != grid[i, j]:
                p_copy = (best_p - payoffs[i, j]) / D_max
                if random_vals[i, j] < p_copy:
                    new_grid[i, j] = best_s
    return new_grid


def simuler_gitter(L=80, andel_orn=0.5, V=10, C=20, n_skridt=200,
                   naboer=4, n_snapshots=6, seed=None):
    """Simulér gittermodellen i et L x L gitter.

    Hvert dyr spiller mod sine 4 (eller 8) nærmeste naboer og kan kopiere den
    bedst stillede nabos strategi med sandsynlighed proportional med gevinstforskellen.

    Hastighed afhænger af den globale toggle `evspil.HURTIG_TILSTAND`:
      True  (default) — bruger en optimeret implementering, god til store gitre
      False           — bruger en langsom referenceimplementering, god til at
                        kontrollere at de to giver samme resultat

    Returnerer dict med:
      'andel_orn'        : array, andel ørne ved hvert skridt
      'snapshots'        : liste af gitter-billeder jævnt fordelt i tiden
      'snapshot_skridt'  : tidsskridt for hvert snapshot
      'final_grid'       : det endelige gitter
      'tilstand'         : hvilken tilstand der blev brugt ('hurtig' eller 'langsom')
    """
    rng = np.random.default_rng(seed)
    G = gevinst_orn_due(V, C).astype(np.float64)
    grid = _init_gitter(L, andel_orn, rng)
    naboer_arr = (NABOER_4 if naboer == 4 else NABOER_8).astype(np.int64)
    n_nb = len(naboer_arr)
    D_max = float(G.max() - G.min()) * n_nb

    bruges_hurtig = HURTIG_TILSTAND and _HAR_JIT
    skridt_funktion = _gitter_skridt_jit if bruges_hurtig else _gitter_skridt_python

    history = [float(np.mean(grid == 0))]
    if n_snapshots < 2:
        n_snapshots = 2
    snapshot_idx = set(np.linspace(0, n_skridt, n_snapshots, dtype=int).tolist())
    snapshots = [grid.copy()]
    snapshot_skridt = [0]

    for t in range(1, n_skridt + 1):
        random_vals = rng.random((L, L))
        grid = skridt_funktion(grid, G, naboer_arr, D_max, random_vals)
        history.append(float(np.mean(grid == 0)))
        if t in snapshot_idx and t != 0:
            snapshots.append(grid.copy())
            snapshot_skridt.append(t)

    return {
        "andel_orn": np.array(history),
        "snapshots": snapshots,
        "snapshot_skridt": snapshot_skridt,
        "final_grid": grid,
        "tilstand": "hurtig" if bruges_hurtig else "langsom",
    }


# --------------------------------------------------------------------
# Plot-hjælpere til gittermodellen
# --------------------------------------------------------------------

_GITTER_CMAP = plt.matplotlib.colors.ListedColormap(["#c1272d", "#0072b2"])


def plot_gitter_snapshots(resultat, titel=None, figsize=None):
    """Plot snapshots af gitteret over tid (ørne i rødt, duer i blåt)."""
    snapshots = resultat["snapshots"]
    skridt = resultat["snapshot_skridt"]
    n = len(snapshots)
    if figsize is None:
        figsize = (2.6 * n, 3.0)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, snap, t in zip(axes, snapshots, skridt):
        ax.imshow(snap, cmap=_GITTER_CMAP, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"t = {t}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
    if titel is None:
        titel = "Gittermodel: rød = ørn, blå = due"
    fig.suptitle(titel, fontsize=12)
    fig.tight_layout()
    return fig, axes


def plot_gitter_andel(resultat, V=None, C=None, ax=None, label=None,
                      farve="#c1272d"):
    """Plot andelen af ørne over tid for gittersimulering."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(resultat["andel_orn"], color=farve, linewidth=2, label=label)
    if V is not None and C is not None and C > 0:
        p_star = min(V / C, 1.0)
        ax.axhline(p_star, color="black", linestyle="--", alpha=0.7,
                   label=f"velblandet ESS p* = V/C = {p_star:.2f}")
    ax.set_xlabel("Tidsskridt")
    ax.set_ylabel("Andel ørne")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    if label is not None or (V is not None and C is not None):
        ax.legend()
    return ax


def plot_gitter_oversigt(resultat, V, C, titel=None):
    """Kombineret plot: snapshots over tid + andel ørne over tid."""
    snapshots = resultat["snapshots"]
    skridt = resultat["snapshot_skridt"]
    n = len(snapshots)
    fig = plt.figure(figsize=(max(10, 2.0 * n), 6.5))
    gs = fig.add_gridspec(2, n, height_ratios=[1.6, 1])
    # Snapshots i øverste række
    snap_axes = []
    for i, (snap, t) in enumerate(zip(snapshots, skridt)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(snap, cmap=_GITTER_CMAP, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"t = {t}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        snap_axes.append(ax)
    # Tidsserie nederst
    ax_ts = fig.add_subplot(gs[1, :])
    plot_gitter_andel(resultat, V=V, C=C, ax=ax_ts)
    if titel:
        fig.suptitle(titel, fontsize=13)
    fig.tight_layout()
    return fig, snap_axes, ax_ts


def plot_endelig_gitter(resultat, V=None, C=None, ax=None, titel=None):
    """Plot kun det endelige gitter."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(resultat["final_grid"], cmap=_GITTER_CMAP, vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if titel is None and V is not None and C is not None:
        andel = float(resultat["andel_orn"][-1])
        titel = f"V={V}, C={C}, slut-andel ørne = {andel:.2f}"
    if titel:
        ax.set_title(titel, fontsize=11)
    return ax


# ====================================================================
# PROJEKT 2: SÆSONØKOSYSTEM (individ-simulering med tidsvarierende ressourcer)
# ====================================================================

def simuler_individer_dynamisk(N=500, andel_orn=0.5,
                               V_funktion=None, C_funktion=None,
                               V=None, C=None,
                               n_skridt=400,
                               E_start=10.0, E_doed=0.0, E_foedsel=15.0,
                               doedsrate=0.05, mutationsrate=0.0,
                               kapacitet=None, seed=None):
    """Individ-simulering hvor V og/eller C kan være funktioner af tiden.

    Brug enten V_funktion(t) og C_funktion(t) (funktioner af tidsskridt) eller
    konstanterne V og C. Hvis kun ét er funktion, bruges den anden konstant.

    Parametre er ellers som i `simuler_individer`. Returnerer derudover
    'V_historik' og 'C_historik'.
    """
    rng = np.random.default_rng(seed)
    if kapacitet is None:
        kapacitet = N
    if V_funktion is None:
        if V is None:
            raise ValueError("Angiv enten V eller V_funktion")
        V_funktion = lambda t: V
    if C_funktion is None:
        if C is None:
            raise ValueError("Angiv enten C eller C_funktion")
        C_funktion = lambda t: C

    n_orn = int(round(N * andel_orn))
    n_due = N - n_orn
    strategier = np.array([0] * n_orn + [1] * n_due, dtype=np.int8)
    energi = np.full(len(strategier), E_start, dtype=float)

    andel_hist = [n_orn / N if N > 0 else 0.0]
    pop_hist = [N]
    V_hist = [float(V_funktion(0))]
    C_hist = [float(C_funktion(0))]

    for t in range(n_skridt):
        V_t = float(V_funktion(t))
        C_t = float(C_funktion(t))
        G = gevinst_orn_due(V_t, C_t)

        n = len(strategier)
        if n >= 2:
            order = rng.permutation(n)
            a = order[0:n - 1:2]
            b = order[1:n:2]
            sa = strategier[a]
            sb = strategier[b]
            energi[a] += G[sa, sb]
            energi[b] += G[sb, sa]

        # Reproduktion
        kan_reproducere = energi > E_foedsel
        n_nye = int(np.sum(kan_reproducere))
        if n_nye > 0:
            nye_strategier = strategier[kan_reproducere].copy()
            if mutationsrate > 0:
                muter = rng.random(n_nye) < mutationsrate
                nye_strategier[muter] = 1 - nye_strategier[muter]
            energi[kan_reproducere] -= E_start
            strategier = np.concatenate([strategier, nye_strategier])
            energi = np.concatenate([energi,
                                     np.full(n_nye, E_start, dtype=float)])

        # Død
        levende = energi > E_doed
        if doedsrate > 0:
            overlever = rng.random(len(strategier)) >= doedsrate
            levende = levende & overlever
        strategier = strategier[levende]
        energi = energi[levende]

        # Bærekapacitet
        if kapacitet is not None and len(strategier) > kapacitet:
            keep = rng.choice(len(strategier), size=kapacitet, replace=False)
            strategier = strategier[keep]
            energi = energi[keep]

        n = len(strategier)
        n_orn = int(np.sum(strategier == 0))
        andel_hist.append(n_orn / n if n > 0 else 0.0)
        pop_hist.append(n)
        V_hist.append(V_t)
        C_hist.append(C_t)

    return {
        "andel_orn": np.array(andel_hist),
        "befolkning": np.array(pop_hist),
        "V_historik": np.array(V_hist),
        "C_historik": np.array(C_hist),
    }


def plot_dynamisk_simulering(resultat, vis_V=True, vis_C=False, vis_ess=True,
                              titel=None):
    """Plot resultat fra simuler_individer_dynamisk: ressource(r), population og andel."""
    n_paneler = 1 + (1 if vis_V else 0) + (1 if vis_C else 0) + 1
    fig, axes = plt.subplots(n_paneler, 1, figsize=(10, 2.3 * n_paneler), sharex=True)
    if n_paneler == 1:
        axes = [axes]
    idx = 0
    if vis_V:
        axes[idx].plot(resultat["V_historik"], color="#e69f00", linewidth=2)
        axes[idx].fill_between(np.arange(len(resultat["V_historik"])),
                               0, resultat["V_historik"],
                               color="#e69f00", alpha=0.15)
        axes[idx].set_ylabel("V(t)\n(ressourcer)")
        axes[idx].grid(True, alpha=0.3)
        idx += 1
    if vis_C:
        axes[idx].plot(resultat["C_historik"], color="#7b3294", linewidth=2)
        axes[idx].set_ylabel("C(t)\n(kampomkost.)")
        axes[idx].grid(True, alpha=0.3)
        idx += 1
    axes[idx].plot(resultat["befolkning"], color="#1b7837", linewidth=2)
    axes[idx].fill_between(np.arange(len(resultat["befolkning"])),
                           0, resultat["befolkning"],
                           color="#1b7837", alpha=0.15)
    axes[idx].set_ylabel("Population")
    axes[idx].grid(True, alpha=0.3)
    idx += 1
    axes[idx].plot(resultat["andel_orn"], color="#c1272d", linewidth=2,
                   label="andel ørne")
    if vis_ess:
        V = resultat["V_historik"]
        C = resultat["C_historik"]
        p_star = np.minimum(V / np.maximum(C, 1e-9), 1.0)
        axes[idx].plot(p_star, color="black", linestyle="--", alpha=0.6,
                       label="p*(t) = V(t)/C(t)")
    axes[idx].legend(loc="upper right")
    axes[idx].set_ylabel("Andel ørne")
    axes[idx].set_xlabel("Tidsskridt")
    axes[idx].set_ylim(-0.05, 1.05)
    axes[idx].grid(True, alpha=0.3)
    if titel:
        fig.suptitle(titel, fontsize=13)
    fig.tight_layout()
    return axes
