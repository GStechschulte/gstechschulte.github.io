import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.stats as stats

    return gridspec, mo, np, plt, stats


@app.cell
def _(np):
    rng = np.random.default_rng(42)
    return (rng,)


@app.cell
def _(rng):
    N = 60
    mu_true = 5.0
    sigma_true = 1.0
    historical = rng.normal(mu_true, sigma_true, size=N)
    return N, historical, mu_true, sigma_true


@app.cell
def _(historical, np):
    empirical_sorted = np.sort(historical)
    return (empirical_sorted,)


@app.cell
def _(np):
    def w1_sorted(x_sorted, y_sorted):
        """W_1 between two equal-weight N-point empirical distributions."""
        return np.mean(np.abs(x_sorted - y_sorted))

    return (w1_sorted,)


@app.cell
def _(np, stats, w1_sorted):
    x_test = np.array([1.0, 2.0, 3.0, 4.0])
    y_test = np.array([1.5, 2.0, 3.5, 4.5])

    assert np.isclose(w1_sorted(x_test, y_test), stats.wasserstein_distance(x_test, y_test)), (
        "Formula mismatch — check sorting!"
    )
    return


@app.cell
def _(np, rng, stats):
    def sample_wasserstein_ball_simple(eng, empirical, epsilon, n_samples):
        """
        Sample from 
        the Wasserstein ball using normalize-then-scale.

        The logic in plain english:
          1. Draw a random direction in R^N (any symmetric distribution works).
          2. Normalize it so its L1 radius equals exactly epsilon — this puts
             it on the surface of the L1 ball (the "sphere").
          3. Scale by Uniform(0, 1) to land somewhere inside the ball,
             not just on its boundary.

        This guarantees W_1(P_hat, Q) <= epsilon for every sample by construction.
        The tradeoff vs. Barthe et al. is that sampling is not perfectly uniform
        over the ball — points cluster slightly toward the center — but the
        distributions we generate are just as diverse and valid for intuition.
        """

        N = len(empirical)

        raw_deltas = rng.normal(0, 1, size=(n_samples, N))
        l1_radii = np.mean(np.abs(raw_deltas), axis=1, keepdims=True)
        sphere_deltas = raw_deltas * (epsilon / l1_radii)

        scale = rng.uniform(0, 1, size=(n_samples, 1))
        scaled_deltas = sphere_deltas * scale

        distributions, w1_distances = [], []
        for delta in scaled_deltas:
            perturbed = np.sort(empirical + delta)
            w1 = stats.wasserstein_distance(empirical, perturbed)
            distributions.append(perturbed)
            w1_distances.append(w1)

        return distributions, np.array(w1_distances)

    return (sample_wasserstein_ball_simple,)


@app.cell
def _(mo):
    slider = mo.ui.slider(
        start=0.01,
        stop=1.5,
        step=0.05,
        label="Ball radius ε",
        value=0.3,
        show_value=True,
    )
    return (slider,)


@app.cell
def _(empirical_sorted, rng, sample_wasserstein_ball_simple, slider):
    ball_eps = slider.value
    ball_dists, ball_w1s = sample_wasserstein_ball_simple(rng, empirical_sorted, ball_eps, 300)
    return ball_dists, ball_eps, ball_w1s


@app.cell
def _(
    N,
    ball_dists,
    ball_eps,
    ball_w1s,
    empirical_sorted,
    gridspec,
    mo,
    mu_true,
    np,
    plt,
    sigma_true,
    slider,
    stats,
):
    color = "#2196F3"
    x_grid = np.linspace(mu_true - 4 * sigma_true, mu_true + 4 * sigma_true, 500)

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(
        "Wasserstein Ball: Sampling from the Ambiguity Set  |  "
        r"Center $\hat{\mathbb{P}}$: empirical distribution, $N=60$, $\mathcal{N}(5,1)$",
        fontsize=11,
    )

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    ax_cdf = fig.add_subplot(gs[0, 0])

    for _q in ball_dists[:80]:
        _cdf_y = np.arange(1, N + 1) / N
        ax_cdf.step(_q, _cdf_y, color=color, alpha=0.08, linewidth=0.8, where="post")

    _cdf_y = np.arange(1, N + 1) / N
    ax_cdf.step(
        empirical_sorted,
        _cdf_y,
        color="black",
        linewidth=2.0,
        where="post",
        label=r"$\hat{\mathbb{P}}$ (empirical)",
        zorder=5,
    )
    ax_cdf.plot(
        x_grid,
        stats.norm.cdf(x_grid, mu_true, sigma_true),
        color="gray",
        linewidth=1.5,
        linestyle="--",
        label=r"True $\mathcal{N}(5,1)$",
    )
    ax_cdf.set_title(f"CDF ribbon — ε = {ball_eps:.2f}", fontsize=12, color=color, fontweight="bold")
    ax_cdf.set_xlabel("Value", fontsize=10)
    ax_cdf.set_ylabel("CDF", fontsize=10)
    ax_cdf.legend(fontsize=8, loc="upper left")
    ax_cdf.set_xlim(x_grid[0], x_grid[-1])
    ax_cdf.set_ylim(-0.02, 1.05)
    ax_cdf.grid(True, alpha=0.3)
    ax_cdf.text(
        0.98, 0.5,
        f"Ribbon width\n∝ ε = {ball_eps:.2f}",
        transform=ax_cdf.transAxes,
        fontsize=8, ha="right", va="center", color=color,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax_hist = fig.add_subplot(gs[0, 1])
    ax_hist.hist(ball_w1s, bins=30, color=color, alpha=0.75, edgecolor="white", linewidth=0.5)
    ax_hist.axvline(ball_eps, color="black", linewidth=2, linestyle="--", label=f"ε = {ball_eps:.2f}")
    ax_hist.axvline(
        ball_w1s.mean(), color=color, linewidth=1.5, linestyle="-",
        label=f"mean = {ball_w1s.mean():.3f}",
    )
    ax_hist.set_xlabel("Actual W₁ distance to P̂", fontsize=10)
    ax_hist.set_ylabel("Count", fontsize=10)
    ax_hist.set_title(f"W₁ distances (ε = {ball_eps:.2f})", fontsize=11)
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, alpha=0.3)
    ax_hist.text(
        0.02, 0.95,
        f"Max = {ball_w1s.max():.3f} ≤ ε ✓",
        transform=ax_hist.transAxes,
        fontsize=9, va="top", color="green", fontweight="bold",
    )

    mo.vstack([slider, mo.as_html(fig)])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
