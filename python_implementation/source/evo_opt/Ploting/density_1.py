import numpy as np
import matplotlib.pyplot as plt


def _kde_log(exponents, grid, bandwidth):
    x = np.log10(np.asarray(exponents, dtype=float))
    d = np.zeros_like(grid)
    for xi in x:
        d += np.exp(-0.5 * ((grid - xi) / bandwidth) ** 2)
    return d / (len(x) * bandwidth * np.sqrt(2.0 * np.pi))


def plot_exponent_density(
    exponent_sets,
    labels=None,
    bandwidth=0.35,
    n_grid=400,
    pad=0.6,
    fill=True,
    colors=None,
    show_mean_in_label=True,
    ax=None,
    title=None,
):
    sets = [np.asarray(s, dtype=float) for s in exponent_sets]
    if any(np.any(s <= 0) for s in sets):
        raise ValueError("All exponents must be positive (log10 is taken).")

    if labels is None:
        labels = [f"set {i + 1}" for i in range(len(sets))]
    if colors is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        colors = [cycle[i % len(cycle)] for i in range(len(sets))] if cycle else \
                 [None] * len(sets)

    all_log = np.concatenate([np.log10(s) for s in sets])
    lo, hi = all_log.min() - pad, all_log.max() + pad
    grid = np.linspace(lo, hi, n_grid)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
    else:
        fig = ax.figure

    for s, lab, col in zip(sets, labels, colors):
        dens = _kde_log(s, grid, bandwidth)
        if show_mean_in_label:
            lab = f"{lab}  (mean log10 zeta = {np.log10(s).mean():.2f})"
        (line,) = ax.plot(grid, dens, lw=2.2, color=col, label=lab)
        if fill:
            ax.fill_between(grid, dens, color=line.get_color(), alpha=0.10)
        ax.scatter(np.log10(s), np.zeros(len(s)), color=line.get_color(),
                   s=40, zorder=5, clip_on=False)

    ax.set_xlabel("log10(exponent)   diffuse <-  -> tight")
    ax.set_ylabel("density")
    ax.set_ylim(bottom=0)
    ax.margins(x=0)
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, ax


# ground_s = [1.2235865418626042e+06, 1.7579759352039013e+05, 3.5013435633395697e+04, 8.9342627809358564e+03, 2.7391920731807936e+03, 9.5877732912713407e+02, 3.6818461825471417e+02, 1.5051917563597291e+02, 6.4076947057106153e+01, 2.7961217111613475e+01, 1.2369305500164561e+01, 5.5025712589527220e+00, 2.4450847578243811e+00, 1.0776126011776241e+00, 4.6680508643216723e-01, 1.9623777862689337e-01, 7.8624012949741612e-02]
# anion_s =  [1.0496933687597523e+06, 1.4885353639065070e+05, 2.9717207313931041e+04, 7.6469693546896642e+03, 2.3648054382907903e+03, 8.3253412877746860e+02, 3.2036171968402124e+02, 1.3081871486850486e+02, 5.5519790426103121e+01, 2.4138123860959105e+01, 1.0639200218720998e+01, 4.7124966754897528e+00, 2.0777975230907582e+00, 9.0052855735117832e-01, 3.7678727983048743e-01, 1.4830782451880603e-01, 5.2971184325058922e-02]
# four_p_s = [6.0885344128230801e+06, 8.4530578321333742e+05, 1.6381609937342259e+05, 4.0738135640366017e+04, 1.2169782390081758e+04, 4.1534003900454500e+03, 1.5610800354168703e+03, 6.2988124277568795e+02, 2.6832344836686752e+02, 1.1945759081769113e+02, 5.5250862823081114e+01, 2.6440382051793062e+01, 1.3032955417546257e+01, 6.5691573263921903e+00, 3.3443118953287820e+00, 1.6860041193970847e+00, 8.1727289004969772e-01]
# plot_exponent_density([ground_s, anion_s, four_p_s],)
# plt.show()

# ground_p = [1.5916090620282484e+03, 3.6361237106427711e+02, 1.1294579810222763e+02, 4.2647943082405597e+01, 1.8048315226364377e+01, 8.0929047773653107e+00, 3.7066594613343038e+00, 1.6950148136135301e+00, 7.6270697671210153e-01, 3.3369758871259136e-01, 1.3985312572771708e-01, 5.4837878595882260e-02]
# anion_p  = [1.2278722869933479e+03, 2.7522634648530453e+02, 8.6842684173419372e+01, 3.3111095635872999e+01, 1.3874658557105739e+01, 6.0625082931160632e+00, 2.6917752777094974e+00, 1.1967401229333512e+00, 5.2268198310478708e-01, 2.1566441644358045e-01, 7.8021594101952113e-02, 2.1824523929949304e-02]
# four_p_p = [5.7131719889892929e+03, 1.1925440397680156e+03, 3.6879255342236343e+02, 1.4386878331388087e+02, 6.3967395152708541e+01, 3.0646472381931254e+01, 1.5433750863987486e+01, 8.1098856573762284e+00, 4.4285670673739279e+00, 2.4766586482348711e+00, 1.3639396293865258e+00, 6.8433786668023266e-01]
# plot_exponent_density([ground_p, anion_p, four_p_p],)
# plt.show()

# ground_d = [1.6807275391749585e+01, 5.6126006883145170e+00, 1.8111467952009486e+00, 5.1556449512192248e-01, 1.6565602209086366e-01]
# anion_d  = [1.6629536267069241e+01, 5.4949251331890121e+00, 1.8795610628595512e+00, 4.0251810066325089e-01, 1.1570739499871803e-01]
# four_p_d = [4.0159326870233244e+01, 1.6956667970407413e+01, 9.1803866599317399e+00, 4.3979194148372702e+00, 1.8733174030498427e+00]
# plot_exponent_density([ground_d, anion_d, four_p_d],)
# plt.show()

ground_f = [1.5412033312464761e+01, 5.3897465946799326e+00, 1.3305382809833508e+00, 3.3906705616576144e-01]
anion_f  = [1.3779008136206885e+01, 4.5424166702583548e+00, 6.3641025029092491e-01, 2.1513851912996207e-01]
four_p_f = [3.2438640360733856e+01, 1.4529175268860461e+01, 6.8055323348102394e+00, 2.9260072682354652e+00]
plot_exponent_density([ground_f, anion_f, four_p_f],)
plt.show()