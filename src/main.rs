mod ising;
mod qubo;
mod sa;

use ising::IsingProblem;
use qubo::QuboParams;
use sa::{IsingSA, SAParams};

fn main() -> Result<(), String> {
    test_ising_direct()?;
    test_qubo_matrix_conversion()?;

    Ok(())
}

fn test_ising_direct() -> Result<(), String> {
    println!("=== Ising direct test ===");

    // E = 0.2 s0 - 0.1 s1 + 0.3 s2 + 0.5 s0 s1 - 0.4 s1 s2
    let h = vec![0.2, -0.1, 0.3];
    let starts = vec![0, 1];
    let ends = vec![1, 2];
    let values = vec![0.5, -0.4];

    let problem = IsingProblem::from_edges(h, &starts, &ends, &values)?;

    let params = SAParams::geometric_beta_schedule(
        0.01,
        10.0,
        100,
        10,
        42,
    )?;

    let mut sa = IsingSA::new(params);
    let samples = sa.sample(&problem, 20);

    for (rank, sample) in samples.iter().take(5).enumerate() {
        println!(
            "rank={} energy={:.6} state={:?}",
            rank, sample.energy, sample.state
        );
    }

    println!();

    Ok(())
}

fn test_qubo_matrix_conversion() -> Result<(), String> {
    println!("=== QUBO matrix -> Ising test ===");

    // QUBO:
    // E_Q(x) = 1.0 x0 - 2.0 x1 + 0.5 x2
    //        + 4.0 x0 x1 - 1.2 x1 x2
    //
    // Matrix form:
    // E_Q(x) = x^T Q x
    //
    // Here we store the quadratic terms in the upper triangular part.
    let q = vec![
        vec![1.0, 4.0, 3.0],
        vec![0.0, -2.0, -1.2],
        vec![0.0, 0.0, 0.5],
    ];

    let qubo = QuboParams::from_matrix(&q)?;
    println!("QUBO linear = {:?}", qubo.linear);
    println!("QUBO starts = {:?}", qubo.starts);
    println!("QUBO ends    = {:?}", qubo.ends);
    println!("QUBO values  = {:?}", qubo.values);
    println!("QUBO offset  = {:.6}", qubo.offset);

    let ising = qubo.to_ising()?;
    println!("Ising h      = {:?}", ising.h);
    println!("Ising starts = {:?}", ising.starts);
    println!("Ising ends   = {:?}", ising.ends);
    println!("Ising values = {:?}", ising.values);
    println!("Ising offset = {:.6}", ising.offset);

    let problem = IsingProblem::from_edges(
        ising.h.clone(),
        &ising.starts,
        &ising.ends,
        &ising.values,
    )?;

    let params = SAParams::geometric_beta_schedule(
        0.01,
        10.0,
        100,
        10,
        123,
    )?;

    let mut sa = IsingSA::new(params);
    let samples = sa.sample(&problem, 20);

    for (rank, sample) in samples.iter().take(5).enumerate() {
        let binary: Vec<u8> = sample
            .state
            .iter()
            .map(|&s| if s == 1 { 1 } else { 0 })
            .collect();

        let qubo_energy = sample.energy + ising.offset;

        println!(
            "rank={} ising_energy={:.6} qubo_energy={:.6} spin={:?} binary={:?}",
            rank,
            sample.energy,
            qubo_energy,
            sample.state,
            binary
        );
    }

    println!();

    Ok(())
}
