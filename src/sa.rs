use crate::ising::IsingProblem;

use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

/// SAの実行条件、特に温度スケジュールを管理するための構造体です。
/// SA本体では、各ステップで逆温度`beta`を使って、エネルギーが上がるflipを受理する確率を決める。
/// ```rust
/// p = exp(-beta * ΔE)
/// ```
/// で、`ΔE`はエネルギーの変化量。この`beta`の値をどのように増やしていくかを`SAParams`が管理する。
#[cfg_attr(doc, katexit::katexit)]
#[derive(Debug, Clone)]
pub struct SAParams {
    /// 各beta値ごとに何回のsweepを行うか。1 sweepは全ての変数を一度ずつ更新すること。
    pub sweeps_per_beta: usize,
    /// 逆温度`beta`のスケジュール。SAの各ステップでこのベータ値を使って、エネルギーが上がるflipを受理する確率を決める。
    pub beta_schedule: Vec<f64>,
    /// SAの乱数シード。これを固定することで、同じ条件でSAを再現できる。
    pub seed: u64,
}

impl SAParams {
    /// 幾何的なbetaスケジュールを生成するためのコンストラクタ。
    /// `beta_start`から`beta_end`までを、`num_betas`個のbeta値で幾何的に増加させる。
    /// # Arguments
    /// - `beta_start` - スケジュールの開始点となる逆温度。通常は小さい値（高温）を指定する。
    /// - `beta_end` - スケジュールの終了点となる逆温度。通常は大きい値（低温）を指定する。
    /// - `num_betas` - スケジュールに含まれるbeta値の数。少なくとも2以上でなければならない。
    /// - `sweeps_per_beta` - 各beta値ごとに何回のsweepを行うか。1 sweepは全ての変数を一度ずつ更新すること。
    /// - `seed` - SAの乱数シード。これを固定することで、同じ条件でSAを再現できる。
    /// # Returns
    /// - `SAParams`のインスタンスを返す。スケジュールは幾何的に増加するbeta値で構成される。
    /// # Errors
    /// - `beta_start`や`beta_end`が0以下の場合、エラーを返す。
    /// - `num_betas`が2未満の場合、エラーを返す。
    /// - `sweeps_per_beta`が0の場合、エラーを返す。
    /// 
    /// # Note
    /// #### 幾何スケジュールの計算
    /// ```rust
    /// let ratio = (beta_end / beta_start).powf(1.0 / (num_betas as f64 - 1.0));
    /// ```
    /// ここでは、`beta`を等比数列で増やすための倍率`ratio`を計算している。
    /// 作りたいスケジュールは、
    /// ```text
    /// beta_0 = beta_start
    /// beta_1 = beta_start * ratio
    /// beta_2 = beta_start * ratio^2
    /// ...
    /// beta_{N-1} = beta_start * ratio^{N-1} = beta_end
    /// ```
    /// ここで、`N`は`num_betas`の値。最後の式から、`ratio`を求めると上記のようになる。
    /// ```text
    /// beta_end = beta_start * ratio^(N-1)
    /// ratio = (beta_end / beta_start)^(1/(N-1))
    /// ```
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

/// 1回のSA実行結果を保存する構造体。
#[cfg_attr(doc, katexit::katexit)]
#[derive(Debug, Clone)]
pub struct Sample {
    /// 最終的に得られたスピン状態。`state[i]` はスピン `s_i` の値を表す。`-1` または `+1` のいずれかでなければならない。
    pub state: Vec<i8>,
    /// `state` に対応するエネルギー値。
    pub energy: f64,
}

/// Ising問題に対してSAを実行するための構造体。
#[cfg_attr(doc, katexit::katexit)]
pub struct IsingSA {
    /// SAの実行条件を管理する構造体。特に温度スケジュールを管理する。
    params: SAParams,
    /// SAの乱数生成器。SAの各ステップで、エネルギーが上がるflipを受理するかどうかを決めるために乱数を使用する。
    rng: Xoshiro256PlusPlus,
}


impl IsingSA {
    /// `IsingSA`の新しいインスタンスを作成するためのコンストラクタ。
    /// # Arguments
    /// - `params` - SAの実行条件を管理する構造体。特に温度スケジュールを管理する。
    /// # Returns
    /// - `IsingSA`のインスタンスを返す。内部で乱数生成器が`params.seed`を使って初期化される。
    pub fn new(params: SAParams) -> Self {
        let rng = Xoshiro256PlusPlus::seed_from_u64(params.seed);
        Self { params, rng }
    }

    /// ランダムなスピン状態を生成する。各スピンは`-1`または`+1`のいずれかで、等しい確率で選ばれる。
    /// # Arguments
    /// - `num_vars` - 生成するスピン状態の変数の数。`num_vars`個のスピンを持つ状態が生成される。
    /// # Returns
    /// - ランダムに生成されたスピン状態を表すベクトル。`state[i]` はスピン `s_i` の値を表す。`-1` または `+1` のいずれかでなければならない。
    pub fn random_spin_state(&mut self, num_vars: usize) -> Vec<i8> {
        (0..num_vars)
            .map(|_| if self.rng.gen_bool(0.5) { 1 } else { -1 })
            .collect()
    }

    /// SAを1回実行する。与えられた初期状態からスタートして、SAの温度スケジュールに従って状態を更新していく。
    /// # Arguments
    /// - `problem` - SAを実行する対象のIsing問題。エネルギー計算やエネルギー変化量の計算に使用される。
    /// - `state` - SAの初期状態を表すベクトル。`state[i]` はスピン `s_i` の値を表す。`-1` または `+1` のいずれかでなければならない。
    /// # Returns
    /// - SAの実行が完了した後の状態に対応するエネルギー値を返す。`state`はSAの実行中に更新される。
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

    /// SAを複数回実行して、得られたサンプルをエネルギーの昇順でソートして返す。
    /// # Arguments
    /// - `problem` - SAを実行する対象のIsing問題。エネルギー計算やエネルギー変化量の計算に使用される。
    /// - `num_reads` - SAを何回実行するか。各実行でランダムな初期状態からスタートする。
    /// # Returns
    /// - SAを複数回実行して得られたサンプルのベクトル。各サンプルは、最終的に得られたスピン状態とそのエネルギー値を含む。サンプルはエネルギーの昇順でソートされている。
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