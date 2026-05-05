mod ising;
mod qubo;
mod sa;

use crate::ising::IsingProblem;
use crate::qubo::QuboParams;
use crate::sa::{IsingSA, SAParams};
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone)]
pub struct PySample {
    #[pyo3(get)]
    pub state: Vec<i8>,

    #[pyo3(get)]
    pub binary: Vec<i8>,

    #[pyo3(get)]
    pub energy: f64,
}

#[pyfunction]
#[pyo3(signature = (
    h,
    starts,
    ends,
    values,
    num_reads = 100,
    beta_start = 0.01,
    beta_end = 10.0,
    num_betas = 100,
    sweeps_per_beta = 10,
    seed = 42
))]
fn sample_ising_edges(
    h: Vec<f64>,
    starts: Vec<usize>,
    ends: Vec<usize>,
    values: Vec<f64>,
    num_reads: usize,
    beta_start: f64,
    beta_end: f64,
    num_betas: usize,
    sweeps_per_beta: usize,
    seed: u64,
) -> PyResult<Vec<PySample>> {
    let problem = IsingProblem::from_edges(h, &starts, &ends, &values)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    let params = SAParams::geometric_beta_schedule(
        beta_start,
        beta_end,
        num_betas,
        sweeps_per_beta,
        seed,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    let mut sa = IsingSA::new(params);
    let samples = sa.sample(&problem, num_reads);

    Ok(samples
        .into_iter()
        .map(|s| {
            let binary = s
                .state
                .iter()
                .map(|&spin| if spin == 1 { 1 } else { 0 })
                .collect();

            PySample {
                state: s.state,
                binary,
                energy: s.energy,
            }
        })
        .collect())
}

#[pyfunction]
#[pyo3(signature = (
    q,
    num_reads = 100,
    beta_start = 0.01,
    beta_end = 10.0,
    num_betas = 100,
    sweeps_per_beta = 10,
    seed = 42
))]
fn sample_qubo_matrix(
    q: Vec<Vec<f64>>,
    num_reads: usize,
    beta_start: f64,
    beta_end: f64,
    num_betas: usize,
    sweeps_per_beta: usize,
    seed: u64,
) -> PyResult<Vec<PySample>> {
    let qubo = QuboParams::from_matrix(&q)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    let ising = qubo
        .to_ising()
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    let problem = IsingProblem::from_edges(
        ising.h.clone(),
        &ising.starts,
        &ising.ends,
        &ising.values,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    let params = SAParams::geometric_beta_schedule(
        beta_start,
        beta_end,
        num_betas,
        sweeps_per_beta,
        seed,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    let mut sa = IsingSA::new(params);
    let samples = sa.sample(&problem, num_reads);

    Ok(samples
        .into_iter()
        .map(|s| {
            let binary = s
                .state
                .iter()
                .map(|&spin| if spin == 1 { 1 } else { 0 })
                .collect();

            PySample {
                state: s.state,
                binary,
                // 元のQUBOエネルギーに戻すため offset を足す
                energy: s.energy + ising.offset,
            }
        })
        .collect())
}

#[pyfunction]
fn qubo_matrix_to_ising_params(
    q: Vec<Vec<f64>>,
) -> PyResult<(Vec<f64>, Vec<usize>, Vec<usize>, Vec<f64>, f64)> {
    let qubo = QuboParams::from_matrix(&q)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    let ising = qubo
        .to_ising()
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    Ok((
        ising.h,
        ising.starts,
        ising.ends,
        ising.values,
        ising.offset,
    ))
}

#[pymodule]
fn sasampler(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySample>()?;
    m.add_function(wrap_pyfunction!(sample_ising_edges, m)?)?;
    m.add_function(wrap_pyfunction!(sample_qubo_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(qubo_matrix_to_ising_params, m)?)?;
    Ok(())
}