use crate::ising::IsingProblem;

use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;


#[derive(Debug, Clone)]
pub struct SAParams {
    pub sweeps_per_beta: usize,
    pub beta_schedule: Vec<f64>,
    pub seed: u64,
}

impl SAParams {
    pub fn geometric_beta_schedule(
        beta_start: f64,
        beta_end: f64,
        num_betas: usize,
        sweeps_per_beta: usize,
        seed: u64,
    ) -> Result<Self, String> {
        if beta_start <= 0.0 || beta_end <= 0.0 {
            return Err("beta_start and beta_end must be positive".to_string());
        }
        if num_betas < 2 {
            return Err("num_betas must be at least 2".to_string());
        }
        if sweeps_per_beta == 0 {
            return Err("sweeps_per_beta must be positive".to_string());
        }

        let ratio = (beta_end / beta_start).powf(1.0 / (num_betas as f64 - 1.0));
        let beta_schedule = (0..num_betas)
            .map(|k| beta_start * ratio.powf(k as f64))
            .collect();

        Ok(Self {
            sweeps_per_beta,
            beta_schedule,
            seed,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Sample {
    pub state: Vec<i8>,
    pub energy: f64,
}

pub struct IsingSA {
    params: SAParams,
    rng: Xoshiro256PlusPlus,
}

impl IsingSA {
    pub fn new(params: SAParams) -> Self {
        let rng = Xoshiro256PlusPlus::seed_from_u64(params.seed);
        Self { params, rng }
    }

    pub fn random_spin_state(&mut self, num_vars: usize) -> Vec<i8> {
        (0..num_vars)
            .map(|_| if self.rng.gen_bool(0.5) { 1 } else { -1 })
            .collect()
    }

    pub fn run_once(&mut self, problem: &IsingProblem, state: &mut [i8]) -> f64 {
        assert_eq!(state.len(), problem.num_vars());

        let num_vars = problem.num_vars();
        let mut delta_energy = problem.initial_delta_energy(state);

        for &beta in &self.params.beta_schedule {
            // Same idea as D-Wave/neal:
            // if exp(-beta * dE) is below RNG resolution, the move is effectively impossible.
            // This avoids unnecessary exp() calls for very bad uphill moves.
            let threshold = 44.361_419_555_836_5 / beta; // ln(2^64)

            for _ in 0..self.params.sweeps_per_beta {
                for var in 0..num_vars {
                    let d_e = delta_energy[var];

                    if d_e >= threshold {
                        continue;
                    }

                    let accept = if d_e <= 0.0 {
                        true
                    } else {
                        let p = (-beta * d_e).exp();
                        self.rng.gen_range(0.0..1.0) < p
                    };

                    if accept {
                        // Update neighbors' delta energies before flipping state[var].
                        let old_spin = state[var] as f64;

                        for (&neighbor, &coupling) in problem.neighbors[var]
                            .iter()
                            .zip(problem.couplings[var].iter())
                        {
                            delta_energy[neighbor] +=
                                4.0 * old_spin * coupling * state[neighbor] as f64;
                        }

                        state[var] *= -1;
                        delta_energy[var] *= -1.0;
                    }
                }
            }
        }

        problem.energy(state)
    }

    pub fn sample(&mut self, problem: &IsingProblem, num_reads: usize) -> Vec<Sample> {
        let mut samples = Vec::with_capacity(num_reads);

        for _ in 0..num_reads {
            let mut state = self.random_spin_state(problem.num_vars());
            let energy = self.run_once(problem, &mut state);
            samples.push(Sample { state, energy });
        }

        samples.sort_by(|a, b| a.energy.total_cmp(&b.energy));
        samples
    }
}