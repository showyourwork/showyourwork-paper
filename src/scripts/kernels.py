from sympy import *
import paths

# Free parameters
alpha, beta, gamma, Gamma, lam, tau = symbols(
    r"\alpha \beta \gamma \Gamma \lambda \tau"
)

# Define the kernels. Changing anything here will automatically
# update the corresponding table in the article.
kernels = {
    "Constant": alpha**2,
    "Squared Exponential": exp(-((tau / lam) ** 2) / 2),
    "Exponential": exp(-(tau / lam)),
    r"Mat\'ern 3/2": (1 + sqrt(3) * tau / lam) * exp(-sqrt(3) * tau / lam),
    r"Mat\'ern 5/2": (1 + sqrt(5) * tau / lam + 5 * (tau / lam) ** 2 / 3)
    * exp(-sqrt(5) * tau / lam),
    "Rational Quadratic": (1 + tau**2 / (2 * gamma * lam**2)) ** (-gamma),
    "Cosine": cos(2 * pi * tau / lam),
    "Sine Squared Exponential": exp(-Gamma * sin(pi * tau / lam) ** 2),
    "Stochastic Harmonic Oscillator": cos(sqrt(1 - beta**2) * tau / lam)
    + beta / sqrt(1 - beta**2) * sin(sqrt(1 - beta**2) * tau / lam),
}

# Write all of them to the LaTeX table
with open(paths.output / "kernels.tex", "w") as f:
    f.write(
        (r"\\" + "\n").join(
            name + " & " + "$" + latex(expr) + "$" for (name, expr) in kernels.items()
        )
    )

params = dict(snakemake.params)
kernel = params.pop("kernel")
params = {eval(key): value for key, value in params.items()}
with open(paths.output / "kernel_value.tex", "w") as f:
    value = kernels[kernel].subs(params).evalf()
    params = ",\,".join([f"{key} = {value:.2f}" for key, value in params.items()])
    f.write(rf"the value of the {kernel} kernel for ${params}$ is ${value:.5f}$.")
