import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        r"""
        # Bayesian Updating: Beta-Binomial Model

        Bayesian inference updates a **prior** belief about an unknown parameter
        $\theta$ using observed data to produce a **posterior** distribution.

        For a coin with unknown bias $\theta \in [0, 1]$, the conjugate model is:

        $$\theta \sim \text{Beta}(\alpha, \beta) \qquad \text{(prior)}$$
        $$k \mid \theta, n \sim \text{Binomial}(n, \theta) \qquad \text{(likelihood)}$$

        Because the Beta distribution is conjugate to the Binomial likelihood, the
        posterior is closed-form:

        $$\theta \mid k, n \sim \text{Beta}(\alpha + k,\; \beta + n - k)$$

        Use the controls below to explore how the prior and the observed data shape
        the posterior.
        """
    )
    return


@app.cell
def __(mo):
    mo.md("## Prior parameters")
    return


@app.cell
def __(mo):
    alpha_slider = mo.ui.slider(
        start=0.5,
        stop=20.0,
        step=0.5,
        value=1.0,
        label="Prior α (pseudo-heads)",
        show_value=True,
    )
    beta_slider = mo.ui.slider(
        start=0.5,
        stop=20.0,
        step=0.5,
        value=1.0,
        label="Prior β (pseudo-tails)",
        show_value=True,
    )
    mo.vstack([alpha_slider, beta_slider])
    return alpha_slider, beta_slider


@app.cell
def __(mo):
    mo.md("## Observed data")
    return


@app.cell
def __(mo):
    heads_slider = mo.ui.slider(
        start=0,
        stop=100,
        step=1,
        value=6,
        label="Observed heads (k)",
        show_value=True,
    )
    tails_slider = mo.ui.slider(
        start=0,
        stop=100,
        step=1,
        value=4,
        label="Observed tails (n − k)",
        show_value=True,
    )
    mo.vstack([heads_slider, tails_slider])
    return heads_slider, tails_slider


@app.cell
def __(alpha_slider, beta_slider, heads_slider, tails_slider):
    alpha_prior = alpha_slider.value
    beta_prior = beta_slider.value
    k = heads_slider.value
    n_minus_k = tails_slider.value
    n = k + n_minus_k

    alpha_post = alpha_prior + k
    beta_post = beta_prior + n_minus_k
    return alpha_post, alpha_prior, beta_post, beta_prior, k, n, n_minus_k


@app.cell
def __(alpha_post, alpha_prior, beta_post, beta_prior, k, mo, n):
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    theta = np.linspace(0, 1, 500)

    prior_pdf = stats.beta.pdf(theta, alpha_prior, beta_prior)
    likelihood = stats.binom.pmf(k, n, theta)
    post_pdf = stats.beta.pdf(theta, alpha_post, beta_post)

    # Normalise likelihood for visual comparison
    lik_norm = likelihood / (likelihood.max() + 1e-12) * post_pdf.max()

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.fill_between(theta, prior_pdf, alpha=0.25, color="#4C72B0")
    ax.plot(theta, prior_pdf, color="#4C72B0", linewidth=2)

    ax.fill_between(theta, lik_norm, alpha=0.20, color="#DD8452")
    ax.plot(theta, lik_norm, color="#DD8452", linewidth=2, linestyle="--")

    ax.fill_between(theta, post_pdf, alpha=0.30, color="#55A868")
    ax.plot(theta, post_pdf, color="#55A868", linewidth=2)

    prior_patch = mpatches.Patch(color="#4C72B0", alpha=0.6, label=f"Prior  Beta({alpha_prior:.1f}, {beta_prior:.1f})")
    lik_patch = mpatches.Patch(color="#DD8452", alpha=0.5, label=f"Likelihood  Bin(n={n}, k={k})  [scaled]")
    post_patch = mpatches.Patch(color="#55A868", alpha=0.7, label=f"Posterior  Beta({alpha_post:.1f}, {beta_post:.1f})")

    ax.legend(handles=[prior_patch, lik_patch, post_patch], fontsize=10)
    ax.set_xlabel("θ  (coin bias)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Prior → Posterior update", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    mo.mpl.interactive(fig)
    return (
        ax,
        fig,
        lik_norm,
        lik_patch,
        likelihood,
        mpatches,
        np,
        plt,
        post_patch,
        post_pdf,
        prior_patch,
        prior_pdf,
        stats,
        theta,
    )


@app.cell
def __(alpha_post, alpha_prior, beta_post, beta_prior, k, mo, n, stats):
    prior_mean = alpha_prior / (alpha_prior + beta_prior)
    post_mean = alpha_post / (alpha_post + beta_post)
    post_mode = (alpha_post - 1) / (alpha_post + beta_post - 2) if (alpha_post > 1 and beta_post > 1) else None
    post_std = stats.beta.std(alpha_post, beta_post)
    mle = k / n if n > 0 else 0.0

    mode_str = f"{post_mode:.3f}" if post_mode is not None else "—"

    mo.md(
        f"""
        ## Summary statistics

        | Quantity | Value |
        |---|---|
        | Prior mean | {prior_mean:.3f} |
        | MLE (k/n) | {mle:.3f} |
        | **Posterior mean** | **{post_mean:.3f}** |
        | Posterior mode | {mode_str} |
        | Posterior std | {post_std:.3f} |

        The posterior mean is a **weighted average** of the prior mean and the MLE,
        shrinking towards the prior when $n$ is small and towards the data as $n$ grows.
        """
    )
    return mle, mode_str, post_mean, post_mode, post_std, prior_mean


if __name__ == "__main__":
    app.run()
