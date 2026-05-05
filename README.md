# SASampler

`SASampler` は、IsingモデルおよびQUBO問題を解くためのRust製Simulated Annealing（SA）ソルバです。

現在は、Rustからの実行に加えて、PyO3 / maturin を用いたPythonバインディングにも対応しています。

## 特徴

- Isingモデルの最適化
- QUBO行列入力への対応
- QUBOからIsingへの変換
- single spin flip によるSA
- Metropolis基準による受理判定
- `delta_energy` による差分更新
- Pythonからの呼び出し対応

---

## 対応する問題形式

### Isingモデル

```text
E(s) = sum_i h_i s_i + sum_(i,j) J_ij s_i s_j
s_i ∈ {-1, +1}
```

Rustでは、線形項 `h` と二次結合のedge-listで指定します。

```rust
let h = vec![0.2, -0.1, 0.3];
let starts = vec![0, 1];
let ends = vec![1, 2];
let values = vec![0.5, -0.4];
```

これは次のエネルギーを表します。

```text
E(s) = 0.2s0 - 0.1s1 + 0.3s2 + 0.5s0s1 - 0.4s1s2
```

---

### QUBO

```text
E_Q(x) = x^T Q x
x_i ∈ {0, 1}
```

QUBOは内部でIsing形式へ変換してからSAを実行します。

```text
x_i = (s_i + 1) / 2
s_i = 2x_i - 1
```

---

## Rustでの使用例

```rust
mod ising;
mod qubo;
mod sa;

use ising::IsingProblem;
use sa::{IsingSA, SAParams};

fn main() -> Result<(), String> {
    let h = vec![0.2, -0.1, 0.3];
    let starts = vec![0, 1];
    let ends = vec![1, 2];
    let values = vec![0.5, -0.4];

    let problem = IsingProblem::from_edges(h, &starts, &ends, &values)?;

    let params = SAParams::geometric_beta_schedule(
        0.01, // beta_start
        10.0, // beta_end
        100,  // num_betas
        10,   // sweeps_per_beta
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
```

---

## Pythonからの使用

### インストール

```bash
uv sync
source .venv/bin/activate
maturin develop
```

### QUBO行列を解く例

```python
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
```

出力例:

```text
-2.6999999999999997 [-1, 1, 1] [0, 1, 1]
```

ここで、`state` はIsing spin、`binary` はQUBO変数です。

---

## Python API

### `sample_qubo_matrix`

```python
sasampler.sample_qubo_matrix(
    q,
    num_reads=100,
    beta_start=0.01,
    beta_end=10.0,
    num_betas=100,
    sweeps_per_beta=10,
    seed=42,
)
```

QUBO行列を受け取り、SAで最適化します。

戻り値は `Sample` のリストです。

- `sample.energy`: QUBOエネルギー
- `sample.state`: Ising spin状態
- `sample.binary`: QUBOバイナリ状態

### `sample_ising_edges`

```python
sasampler.sample_ising_edges(
    h,
    starts,
    ends,
    values,
    num_reads=100,
    beta_start=0.01,
    beta_end=10.0,
    num_betas=100,
    sweeps_per_beta=10,
    seed=42,
)
```

edge-list形式のIsingモデルを直接解きます。

### `qubo_matrix_to_ising_params`

```python
h, starts, ends, values, offset = sasampler.qubo_matrix_to_ising_params(Q)
```

QUBO行列をIsingパラメータへ変換します。

---

## QUBO行列の規約

本実装では、QUBO行列を次のように解釈します。

```text
E_Q(x) = x^T Q x
```

そのため、非対角項は以下のようにまとめられます。

```text
x_i x_j の係数 = Q_ij + Q_ji,  i < j
```

上三角だけに係数を入れる形式はそのまま使えます。

```python
Q = [
    [1.0, 4.0, 0.0],
    [0.0, -2.0, -1.2],
    [0.0, 0.0, 0.5],
]
```

この場合、`x0 x1` の係数は `4.0`、`x1 x2` の係数は `-1.2` です。

対称行列として両側に同じ係数を入れると、二次項が2倍になるため注意してください。

---

## ビルド

Rustバイナリとしてビルド:

```bash
cargo build
```

リリースビルド:

```bash
cargo build --release
```

Python拡張として開発インストール:

```bash
maturin develop
```

wheel作成:

```bash
maturin build --release
```

---

## 現在の制限

- 近傍はsingle spin flipのみ
- 制約付きSAは未実装
- cardinality制約、layer-wise制約は未対応
- `sample_qubo_matrix` は密行列入力のため、大規模QUBOでは非効率
- `num_reads` の並列化は今後の拡張対象

---

## 今後の拡張候補

- QUBO edge-list入力API
- Rayonによる `num_reads` 並列化
- QUBOを直接扱うSA
- cardinality制約付きSA
- swap近傍
- NumPy配列を直接受け取るPython API

---

## 目的

このプロジェクトは、既存SAライブラリの単純な置き換えではなく、SAの内部処理を理解しつつ、研究用途に合わせて拡張できるソルバを作ることを目的としています。
