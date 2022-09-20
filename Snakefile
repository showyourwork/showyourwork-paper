rule train_two_moons_flow:
    output:
        "src/data/two_moons_flow.pzflow.pkl"
    cache:
        True
    script:
        "src/scripts/train_two_moons_flow.py"

rule kernels:
    output:
        "src/tex/output/kernels.tex"
    script:
        "src/scripts/kernels.py"