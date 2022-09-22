rule train_two_moons_flow:
    output:
        "src/data/two_moons_flow.pzflow.pkl"
    cache:
        True
    script:
        "src/scripts/train_two_moons_flow.py"

rule kernels:
    output:
        "src/tex/output/kernels.tex",
        "src/tex/output/kernel_value.tex"
    params:
        kernel="Stochastic Harmonic Oscillator",
        beta=0.75,
        lam=0.5,
        tau=0.25
    script:
        "src/scripts/kernels.py"