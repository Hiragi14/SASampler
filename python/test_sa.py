import sasampler

Q = [
    [1.0, 4.0, 0.0],
    [0.0, -2.0, -1.2],
    [0.0, 0.0, 0.5],
]

samples = sasampler.sample_qubo_matrix(
    Q,
    num_reads=20,
    beta_start=0.01,
    beta_end=10.0,
    num_betas=100,
    sweeps_per_beta=10,
    seed=42,
)

for s in samples[:5]:
    print(s.energy, s.state, s.binary)