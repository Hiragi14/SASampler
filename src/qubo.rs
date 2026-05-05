/// QUBO形式の問題を表す構造体。
/// 内部には、QUBOの線形項と二次項をエッジリスト形式で保持する。
#[cfg_attr(doc, katexit::katexit)]
#[derive(Debug, Clone)]
pub struct QuboParams {
    pub linear: Vec<f64>,
    pub starts: Vec<usize>,
    pub ends: Vec<usize>,
    pub values: Vec<f64>,
    pub offset: f64,
}

/// イジングモデルのパラメータを管理する構造体。
/// 内部には、イジングモデルの線形項と二次項をエッジリスト形式で保持する。
#[cfg_attr(doc, katexit::katexit)]
#[derive(Debug, Clone)]
pub struct IsingParams {
    pub h: Vec<f64>,
    pub starts: Vec<usize>,
    pub ends: Vec<usize>,
    pub values: Vec<f64>,
    pub offset: f64,
}

impl QuboParams {
    /// 行列形式のQUBOをエッジリスト形式に変換する関数。
    /// # Arguments
    /// - `q` - QUBOの行列形式。`q[i][j]` は、変数 `i` と `j` の二次項の係数を表す。対角成分 `q[i][i]` は、変数 `i` の線形項の係数を表す。
    /// # Returns
    /// - `QuboParams`のインスタンスを返す。行列形式のQUBOがエッジリスト形式に変換されたもの。
    /// # Errors
    /// - `q` が空の場合、エラーを返す。
    /// - `q` が正方行列でない場合、エラーを返す。
    /// - `q` の要素が数値でない場合、エラーを返す。
    pub fn from_matrix(q: &[Vec<f64>]) -> Result<Self, String> {
        let n = q.len();

        if n == 0 {
            return Err("QUBO matrix must not be empty".to_string());
        }

        for (row_idx, row) in q.iter().enumerate() {
            if row.len() != n {
                return Err(format!(
                    "QUBO matrix must be square: row {} has length {}, expected {}",
                    row_idx,
                    row.len(),
                    n
                ));
            }
        }

        let mut linear = vec![0.0; n];
        let mut starts = Vec::new();
        let mut ends = Vec::new();
        let mut values = Vec::new();

        for i in 0..n {
            linear[i] = q[i][i];
        }

        for i in 0..n {
            for j in (i + 1)..n {
                let q_ij = q[i][j] + q[j][i];

                if q_ij == 0.0 {
                    continue;
                }

                starts.push(i);
                ends.push(j);
                values.push(q_ij);
            }
        }

        Ok(Self {
            linear,
            starts,
            ends,
            values,
            offset: 0.0,
        })
    }

    /// QUBO形式の問題をイジング形式に変換する関数。
    /// QUBO:
    /// $$
    ///   E_Q(x) = offset + sum_i linear[i] x_i + sum_k quadratic_values[k] x_u x_v
    /// $$
    /// イジング:
    /// $$
    ///   E_I(s) = sum_i h[i] s_i + sum_(i,j) J_ij s_i s_j
    /// $$
    /// ここで、`x_i = (s_i + 1) / 2` と変換することで、QUBOのバイナリ変数 `x_i` をイジングのスピン変数 `s_i` に対応させる。
    /// この関数は、QUBOの線形項と二次項をイジングの線形項と二次項に変換し、定数オフセットも計算する。
    /// QUBOの二次項 `q_uv x_u x_v` は、イジングの二次項 `J_uv s_u s_v` と線形項 `h_u s_u + h_v s_v` に分解される。
    /// # Arguments
    /// - `self` - QUBO形式の問題を表す構造体。
    /// # Returns
    /// - `IsingParams`のインスタンスを返す。QUBO形式の問題がイジング形式に変換されたもの。
    /// # Errors
    /// - QUBOのエッジリストの長さが不一致の場合、エラーを返す。
    /// - エッジのインデックスが変数の数を超えている場合、エラーを返す。
    pub fn to_ising(&self) -> Result<IsingParams, String> {
        let n = self.linear.len();

        if self.starts.len() != self.ends.len() || self.starts.len() != self.values.len() {
            return Err("QUBO edge vectors have mismatched lengths".to_string());
        }

        let mut h = vec![0.0; n];
        let mut starts = Vec::new();
        let mut ends = Vec::new();
        let mut values = Vec::new();
        let mut offset = self.offset;

        for i in 0..n {
            h[i] += self.linear[i] / 2.0;
            offset += self.linear[i] / 2.0;
        }

        for ((&u, &v), &q_uv) in self
            .starts
            .iter()
            .zip(self.ends.iter())
            .zip(self.values.iter())
        {
            if u >= n || v >= n {
                return Err(format!(
                    "invalid QUBO edge index: ({}, {}) for num_vars={}",
                    u, v, n
                ));
            }

            if u == v {
                // x_i^2 = x_i, so diagonal quadratic terms are linear terms.
                h[u] += q_uv / 2.0;
                offset += q_uv / 2.0;
                continue;
            }

            h[u] += q_uv / 4.0;
            h[v] += q_uv / 4.0;
            offset += q_uv / 4.0;

            starts.push(u);
            ends.push(v);
            values.push(q_uv / 4.0);
        }

        Ok(IsingParams {
            h,
            starts,
            ends,
            values,
            offset,
        })
    }
}


impl IsingParams {
    /// イジング形式の問題をQUBO形式に変換する関数。
    /// イジング:
    /// $$
    ///   E_I(s) = sum_i h[i] s_i + sum_(i,j) J_ij s_i s_j
    /// $$
    /// QUBO:
    /// $$
    ///   E_Q(x) = offset + sum_i linear[i] x_i + sum_k quadratic_values[k] x_u x_v
    /// $$
    /// ここで、`x_i = (s_i + 1) / 2` と変換することで、イジングのスピン変数 `s_i` をQUBOのバイナリ変数 `x_i` に対応させる。
    /// この関数は、イジングの線形項と二次項をQUBOの線形項と二次項に変換し、定数オフセットも計算する。
    /// イジングの二次項 `J_ij s_i s_j` は、QUBOの二次項 `4 J_ij x_i x_j` と線形項 `-2 J_ij x_i - 2 J_ij x_j` に分解される。
    /// # Arguments
    /// - `self` - イジング形式の問題を表す構造体。
    /// # Returns
    /// - `QuboParams`のインスタンスを返す。イジング形式の問題がQUBO形式に変換されたもの。
    /// # Errors
    /// - イジングのエッジリストの長さが不一致の場合、エラーを返す。
    /// - エッジのインデックスが変数の数を超えている場合、エラーを返す。
    pub fn to_qubo(&self) -> Result<QuboParams, String> {
        let n = self.h.len();

        if self.starts.len() != self.ends.len() || self.starts.len() != self.values.len() {
            return Err("Ising edge vectors have mismatched lengths".to_string());
        }

        let mut linear = vec![0.0; n];
        let mut starts = Vec::new();
        let mut ends = Vec::new();
        let mut values = Vec::new();
        let mut offset = self.offset;

        // h_i s_i = h_i (2 x_i - 1)
        //          = 2 h_i x_i - h_i
        for i in 0..n {
            linear[i] += 2.0 * self.h[i];
            offset -= self.h[i];
        }

        // J_ij s_i s_j
        // = J_ij (2x_i - 1)(2x_j - 1)
        // = 4J_ij x_i x_j - 2J_ij x_i - 2J_ij x_j + J_ij
        for ((&u, &v), &j_uv) in self
            .starts
            .iter()
            .zip(self.ends.iter())
            .zip(self.values.iter())
        {
            if u >= n || v >= n {
                return Err(format!(
                    "invalid Ising edge index: ({}, {}) for num_vars={}",
                    u, v, n
                ));
            }

            if u == v {
                // For Ising, s_i^2 = 1, so self-loop is a constant offset.
                offset += j_uv;
                continue;
            }

            linear[u] -= 2.0 * j_uv;
            linear[v] -= 2.0 * j_uv;
            offset += j_uv;

            starts.push(u);
            ends.push(v);
            values.push(4.0 * j_uv);
        }

        Ok(QuboParams {
            linear,
            starts,
            ends,
            values,
            offset,
        })
    }

    /// SAで使用できるイジング形式に変換する関数。
    /// # Arguments
    /// - `self` - イジング形式の問題を表す構造体。
    /// # Returns
    /// - `IsingProblem`のインスタンスを返す。イジング形式の問題がSAで使用できる形式に変換されたもの。
    /// # Errors
    /// - イジングのエッジリストの長さが不一致の場合、エラーを返す。
    /// - エッジのインデックスが変数の数を超えている場合、エラーを返す。
    pub fn to_problem(&self) -> Result<crate::ising::IsingProblem, String> {
        crate::ising::IsingProblem::from_edges(
            self.h.clone(),
            &self.starts,
            &self.ends,
            &self.values,
        )
    }
}
