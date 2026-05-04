mod ising;
mod sa;
use ising::{IsingProblem};
use sa::{IsingSA, SAParams};


fn main() -> Result<(), String> {
    // Small example:
    // E = 0.2 s0 - 0.1 s1 + 0.3 s2 + 0.5 s0 s1 - 0.4 s1 s2
    let h = vec![0.2, -0.1, 0.3];
    let starts = vec![0, 1];
    let ends = vec![1, 2];
    let values = vec![0.5, -0.4];

    let problem = IsingProblem::from_edges(h, &starts, &ends, &values)?;

    let params = SAParams::geometric_beta_schedule(
        0.01, // beta_start: high temperature
        10.0, // beta_end: low temperature
        100,  // number of beta values
        10,   // sweeps per beta
        42,   // seed
    )?;

    let mut sa = IsingSA::new(params);
    let samples = sa.sample(&problem, 20);

    for (rank, sample) in samples.iter().take(5).enumerate() {
        println!(
            "rank={} energy={:.6} state={:?}",
            rank, sample.energy, sample.state
        );
    }

    Ok(())
}
