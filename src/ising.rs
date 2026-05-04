/// イジングモデルの問題インスタンスを表す構造体。
///
/// この構造体は、スピン変数 $s_i \in \{-1, +1\}$ に対する
/// 次のエネルギー関数を表す。
///
/// $$
/// E(s) = \sum_i h_i s_i + \sum_{(i,j)} J_{ij} s_i s_j
/// $$
///
/// ここで、$h_i$ は各スピンにかかる外部磁場、$J_{ij}$ はスピン $i$ と $j$
/// の間の相互作用係数を表す。
///
/// `neighbors[i][k]` と `couplings[i][k]` は対応しており、
/// スピン `i` が隣接するスピン番号 `neighbors[i][k]` との間に
/// 結合係数 `couplings[i][k]` を持つことを意味する。
///
/// # Notes
///
/// - `h.len()` はスピン数を表す。
/// - `neighbors.len()` と `couplings.len()` は `h.len()` と一致している必要がある。
/// - 各 `i` について、`neighbors[i].len()` と `couplings[i].len()` は一致している必要がある。
/// - 無向グラフとして扱う場合、結合 `(i, j)` を `i -> j` と `j -> i` の両方に
///   登録すると、エネルギー計算時に二重カウントされる可能性がある。
#[cfg_attr(doc, katexit::katexit)]
#[derive(Debug, Clone)]
pub struct IsingProblem {
    /// 各スピンに対する外部磁場 $h_i$。
    ///
    /// `h[i]` は、スピン $s_i$ にかかる線形項の係数を表す。
    ///
    /// エネルギー関数における寄与は次の通り。
    ///
    /// ```text
    /// h[i] * s[i]
    /// ```
    pub h: Vec<f64>,

    /// 各スピンに隣接するスピンのインデックス。
    ///
    /// `neighbors[i]` は、スピン `i` と相互作用を持つスピンの一覧を表す。
    /// 例えば `neighbors[i][k] = j` のとき、スピン `i` はスピン `j` と
    /// 相互作用を持つ。
    ///
    /// 対応する結合係数は `couplings[i][k]` に格納される。
    pub neighbors: Vec<Vec<usize>>,

    /// 各隣接スピンとの結合係数 $J_{ij}$。
    ///
    /// `couplings[i][k]` は、`neighbors[i][k]` が示すスピンとの
    /// 相互作用係数を表す。
    ///
    /// つまり、`neighbors[i][k] = j` のとき、
    /// `couplings[i][k]` は $J_{ij}$ に対応する。
    ///
    /// エネルギー関数における寄与は次の通り。
    ///
    /// ```text
    /// couplings[i][k] * s[i] * s[neighbors[i][k]]
    /// ```
    /// ```rust
    /// neighbors[1][1] = 2
    /// couplings[1][1] = -0.4
    /// ```
    /// - 変数1の1番目の隣接先は変数2で、重みは-0.4
    pub couplings: Vec<Vec<f64>>,
}

impl IsingProblem {
    /// Isingモデルのcouplerをedge-list形式から受け取り、[`IsingProblem`] を構築する。
    ///
    /// 対象とするIsingエネルギーは次の形である。
    ///
    /// $$
    /// E(s) = \sum_i h_i s_i + \sum_{(i,j)} J_{ij} s_i s_j
    /// $$
    ///
    /// ここで、各spinは `s_i ∈ {-1, +1}` を取る。
    ///
    /// # Arguments
    ///
    /// * `h` - Isingモデルの線形係数。
    ///   `h[i]` はspin `s_i` に対応する局所場係数を表す。
    ///   変数数は `h.len()` によって決まる。
    ///
    /// * `coupler_starts` - 二次結合の始点index。
    ///   `coupler_starts[k]` は `k` 番目のcouplerの片側の変数 `u` を表す。
    ///
    /// * `coupler_ends` - 二次結合の終点index。
    ///   `coupler_ends[k]` は `k` 番目のcouplerのもう片側の変数 `v` を表す。
    ///
    /// * `coupler_values` - 二次結合の重み。
    ///   `coupler_values[k]` は
    ///   `coupler_starts[k]` と `coupler_ends[k]` の間の結合係数 `J_uv` を表す。
    ///
    /// # Returns
    ///
    /// 入力が妥当な場合は `Ok(IsingProblem)` を返す。
    /// coupler配列の長さが一致しない場合、変数indexが範囲外の場合、
    /// またはself-loopを含む場合は `Err(String)` を返す。
    ///
    /// # Example
    ///
    /// ```rust
    /// let h = vec![0.2, -0.1, 0.3];
    /// let starts = vec![0, 1];
    /// let ends = vec![1, 2];
    /// let values = vec![0.5, -0.4];
    ///
    /// let problem = IsingProblem::from_edges(h, &starts, &ends, &values)?;
    /// ```
    pub fn from_edges(
        h: Vec<f64>,
        coupler_starts: &[usize],
        coupler_ends: &[usize],
        coupler_values: &[f64],
    ) -> Result<Self, String> {
        // 配列の長さのチェック
        if coupler_starts.len() != coupler_ends.len()
            || coupler_starts.len() != coupler_values.len()
        {
            return Err("coupler vectors have mismatched lengths".to_string());
        }
        
        // 変数の定義
        let num_vars = h.len();
        // 隣接リストの初期化
        let mut neighbors = vec![Vec::new(); num_vars];
        let mut couplings = vec![Vec::new(); num_vars];

        for ((&u, &v), &j) in coupler_starts
            .iter()
            .zip(coupler_ends.iter())
            .zip(coupler_values.iter())
        {
            if u >= num_vars || v >= num_vars {
                return Err(format!(
                    "invalid coupler index: ({}, {}) for num_vars={}",
                    u, v, num_vars
                ));
            }
            if u == v {
                return Err("self-loop couplers are not supported in this minimal version".to_string());
            }

            neighbors[u].push(v);
            couplings[u].push(j);
            neighbors[v].push(u);
            couplings[v].push(j);
        }

        Ok(Self {
            h,
            neighbors,
            couplings,
        })
    }

    /// 変数の数を返す。
    pub fn num_vars(&self) -> usize {
        self.h.len()
    }

    /// 与えられたスピン状態のエネルギーを計算する。
    pub fn energy(&self, state: &[i8]) -> f64 {
        assert_eq!(state.len(), self.num_vars());

        let mut e = 0.0;

        for i in 0..self.num_vars() {
            e += self.h[i] * state[i] as f64;
        }

        // Count each undirected edge only once by requiring i < j.
        for i in 0..self.num_vars() {
            for (&j, &coupling) in self.neighbors[i].iter().zip(self.couplings[i].iter()) {
                if i < j {
                    e += coupling * state[i] as f64 * state[j] as f64;
                }
            }
        }

        e
    }

    pub fn flip_delta_energy(&self, state: &[i8], var: usize) -> f64 {
        let mut local_field = self.h[var];

        for (&neighbor, &coupling) in self.neighbors[var]
            .iter()
            .zip(self.couplings[var].iter())
        {
            local_field += coupling * state[neighbor] as f64;
        }

        -2.0 * state[var] as f64 * local_field
    }

    pub fn initial_delta_energy(&self, state: &[i8]) -> Vec<f64> {
        (0..self.num_vars())
            .map(|i| self.flip_delta_energy(state, i))
            .collect()
    }
}