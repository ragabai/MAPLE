# MAPLE — Mathematical Framework and Theoretical Analysis

Status labels: **Theorem / Proposition / Corollary** have complete proofs below. **Remark** is a comment, not a
claim. **Conjecture** is a claim with no proof, which the experiments examine. Every proved statement has a
numerical sanity check in `tests/unit/test_theory_numerics.py`.

## 1. Setting and notation

* **Datasets.** A regression dataset is `D = {(x_i, y_i)}_{i=1}^n` with `x ∈ R^d`, `y ∈ R`. Real datasets come from
  an unknown meta-distribution `𝒫_real` over data-generating processes.
* **Mathematical task distribution.** A finite set of *cells* `𝒞 = {1, …, M}`, each a pair (function family, complexity
  level). Cell `c` has a fixed *task generator* `G_c`, a distribution over tasks
  `T = (D_ctx, X_qry, y_qry, f_qry)`. Here `f_qry = f*(X_qry)` is the noise-free target, and
  `y = f*(x) + ε` with `E[ε | x] = 0` and `Var(ε | x) = σ²_T(x)`. The *task prior* induced by a cell distribution
  `q ∈ Δ_M` (the probability simplex) is the mixture `G_q = Σ_c q_c G_c`. `p⁰` is the base distribution (uniform
  in all experiments).
* **Prediction system.** A PFN `θ` maps (context, query point) to a prediction `θ(x; D_ctx)`. Its task loss is the
  query mean squared error `ℓ(θ; T) = (1/|Q|) Σ_{x∈Q} (θ(x; D_ctx) − y(x))²`. Its cell risk is
  `R_c(θ) = E_{T∼G_c} ℓ(θ; T)`.
* **Experts.** A committee `E = {f_1, …, f_K}` of classical regressors. Each is fitted on `D_ctx` and predicts
  `f_k(x)`. The committee mean is `f̄ = (1/K) Σ_k f_k`.
* **Disagreement functions** (pointwise, for query `x`). The prediction matrix `P ∈ R^{K×|Q|}` has entries `P_kx = f_k(x)`.
  * committee variance (ambiguity) `A(x) = (1/K) Σ_k (f_k(x) − f̄(x))²`;
  * pairwise disagreement `D_pw(x) = 2/(K(K−1)) Σ_{i<j} |f_i(x) − f_j(x)|`;
  * normalised versions divide by `σ_y² + ε` (variance-type) or `σ_y + ε` (absolute-type), with `σ_y` the context
    target scale. Tasks are standardised (`σ_y = 1` on the context), so normalisation is implicit;
  * **PFN–committee gap** `Δ(x) = (θ(x) − f̄(x))²`. This is the signal MAPLE uses by default.
* **Cell scores.** For a pointwise disagreement `g`, the per-task score is `Y = min(B, (1/|Q|) Σ_x g(x))`, with a clip
  level `B > 0`. The cell score is `s_c = E_{T∼G_c} Y`. The label-based alternative (variant B) uses `Y = min(B, ℓ(θ; T))`.
* **Adaptive sampling distribution.** With temperature `τ > 0` and floor `γ ∈ [0, 1)`:
  `q*(s) = Gibbs(p⁰, s/τ)`, `q*_c ∝ p⁰_c exp(s_c/τ)`, and `q(s) = (1 − γ) q*(s) + γ u`, with `u` uniform.
* **Meta-training objective.** The PFN is trained by stochastic gradient descent on
  `min_θ E_{c∼q_t} E_{T∼G_c} ℓ(θ; T)`. The sampling distribution `q_t` is refreshed every `U` steps from fresh
  probe tasks (§5). The sampler's inner problem is `max_q ⟨q, s(θ)⟩ − τ KL(q‖p⁰)` (Prop. 1). With the loss score
  `s = R(θ)`, this is the KL-penalised form of the distributionally robust problem
  `min_θ max_{q: KL(q‖p⁰) ≤ ρ} Σ_c q_c R_c(θ)` (Cor. 1).
* **Expert routing.** At inference on a dataset, predictor `k ∈ {PFN, experts}` gets a weight `w_k(x) ≥ 0` with
  `Σ_k w_k(x) = 1`. Global routing uses `w_k ∝ exp(−β L̂_k / min_j L̂_j)`, where `L̂_k` is the out-of-fold MSE on
  the training data. Local routing uses `w_k(x) ∝ exp(−β ê_k(x)/ min_j ê_j(x))`, where `ê_k(x)` is the mean
  out-of-fold squared error of `k` over the `κ` nearest training points of `x`. `β, κ` are chosen by minimising mean
  NRMSE on synthetic validation tasks drawn from the final learned prior `G_q`. No real data is used.
* **Generalisation objective.** Generalisation is measured on (i) ID tasks `G_{p⁰}` with fresh seeds, (ii) OOD task
  distributions built from structures absent from every `G_c` (held-out families, nested compositions, three-way
  interactions, Voronoi partitions, heavy-tailed noise, soft regime mixtures), and (iii) `𝒫_real`. The
  generalisation gap is `NRMSE_OOD − NRMSE_ID`.

Norms: for functions of `x`, `‖g‖ := (E_{x∼P_x} g(x)²)^{1/2}` (the `L²(P_x)` norm over the query distribution), and
`osc(v) := max_c v_c − min_c v_c` for vectors.

## 2. The sampler as a regularised maximisation

**Proposition 1 (Gibbs variational principle).** Let `p⁰ ∈ Δ_M` have full support, `s ∈ R^M`, `τ > 0`. Then
`J(q) = ⟨q, s⟩ − τ KL(q‖p⁰)` has the unique maximiser over `Δ_M`
`q*_c = p⁰_c e^{s_c/τ} / Z`, with `Z = Σ_c p⁰_c e^{s_c/τ}`, and `max_q J(q) = τ log Z`.

*Proof.* For any `q ∈ Δ_M` (with `0 log 0 = 0`),
`KL(q‖q*) = Σ_c q_c log(q_c/q*_c) = Σ_c q_c log(q_c/p⁰_c) − Σ_c q_c s_c/τ + log Z = KL(q‖p⁰) − ⟨q,s⟩/τ + log Z`.
Hence `J(q) = τ log Z − τ KL(q‖q*)`. By Gibbs' inequality, `KL(q‖q*) ≥ 0` with equality iff `q = q*`, because `q*`
has full support. So `J(q) ≤ τ log Z`, with equality iff `q = q*`. ∎ (This is the finite form of the
Donsker–Varadhan variational formula [@donsker1975asymptotic].)

**Corollary 1 (KL-ball DRO).** Let `ρ > 0`, and let `s` be non-constant. The problem `max {⟨q,s⟩ : q ∈ Δ_M, KL(q‖p⁰) ≤ ρ}`
attains its maximum. If `ρ < log(1/p⁰(S_max))`, where `S_max = argmax_c s_c`, every maximiser is a Gibbs
distribution `q*` with some temperature `τ(ρ) ∈ (0, ∞)`.

*Proof.* The feasible set is compact and convex, and the objective is linear, so a maximiser exists. Slater's
condition holds (`q = p⁰` gives `KL = 0 < ρ`), so strong duality holds and there is a multiplier `τ* ≥ 0` such that
every primal maximiser `q̃` maximises `L(q) = ⟨q,s⟩ − τ*(KL(q‖p⁰) − ρ)` over `Δ_M`, with complementary slackness.
Suppose `τ* = 0`. Then `q̃` maximises `⟨q,s⟩` over `Δ_M`, so it is supported on `S_max`. Every such distribution
has `KL(q̃‖p⁰) ≥ log(1/p⁰(S_max)) > ρ` (the minimum of `KL(q‖p⁰)` over distributions supported on a set `S` is
`−log p⁰(S)`), which is infeasible. So `τ* > 0`, and by Prop. 1 the maximiser of `L` is uniquely the Gibbs
distribution with `τ = τ*`. ∎ [@bental2013robust]

*Remark 1.* With the loss score `s_c = R_c(θ)`, the alternating scheme "update `q` from Prop. 1, take gradient steps
on `E_q ℓ`" is a smoothed analogue of Group DRO's exponentiated-gradient weights [@sagawa2020distributionally].
MAPLE's default replaces `R_c` by the label-free disagreement score `s^Δ`. Prop. 4 states when that substitution
preserves the information that matters.

## 3. Properties of the adaptive sampling distribution

**Proposition 2 (no collapse; bounded tilt).** Let `p⁰ = u` (uniform) and `q = (1−γ) q*(s) + γ u`. Then:
(a) `q_c ≥ γ/M` for all `c`;
(b) `q*_a / q*_b = exp((s_a − s_b)/τ) ≤ exp(osc(s)/τ)` for all `a, b`;
(c) `KL(q*‖u) ≤ osc(s)/τ`, and `KL(q‖u) ≤ (1−γ) osc(s)/τ`.

*Proof.* (a) `q*_c ≥ 0`. (b) Take the ratio of the closed form. (c) `KL(q*‖u) = Σ_c q*_c (s_c/τ − log Z)`, and
`log Z = log((1/M) Σ_c e^{s_c/τ}) ≥ min_c s_c/τ`. So `KL(q*‖u) ≤ (⟨q*, s⟩ − min s)/τ ≤ osc(s)/τ`. KL is convex in
its first argument, so `KL(q‖u) ≤ (1−γ) KL(q*‖u) + γ KL(u‖u) = (1−γ) KL(q*‖u)`. ∎

**Proposition 3 (Lipschitz stability of the sampler).** For all `s, s′ ∈ R^M`,
`‖q*(s) − q*(s′)‖₁ ≤ osc(s − s′)/(2τ) ≤ ‖s − s′‖_∞/τ`, and the mixed sampler satisfies the same bound times `(1−γ)`.

*Proof.* Let `v = s′ − s`, `s(t) = s + t v`, `q(t) = q*(s(t))` for `t ∈ [0,1]`. Differentiating the closed form gives
`dq_c/dt = (1/τ) q_c (v_c − ⟨q, v⟩)`. Hence `‖dq/dt‖₁ = (1/τ) E_{c∼q(t)} |v_c − E_{q(t)} v|`. For a random variable
`V` with values in `[a, b]`, Cauchy–Schwarz gives `E|V − EV| ≤ (Var V)^{1/2}`, and Popoviciu's inequality gives
`Var V ≤ (b − a)²/4`. Hence `‖dq/dt‖₁ ≤ osc(v)/(2τ)` for every `t`. Then
`‖q(1) − q(0)‖₁ ≤ ∫₀¹ ‖dq/dt‖₁ dt ≤ osc(v)/(2τ)`. Finally `osc(v) ≤ 2‖v‖_∞`. The mixture is affine in `q*`, with
coefficient `1−γ`. ∎

*Remark 2.* This bound is used in Thm. 1. It also shows that the temperature `τ` directly controls how strongly
score noise moves the sampling distribution (sensitivity study, Table 8).

## 4. Disagreement as a surrogate for error

**Proposition 4 (disagreement versus excess error).** Fix a task and its context. Let `θ`, `f_1..f_K` be
predictors that depend only on the context, and let `y = f*(x) + ε` with `E[ε|x] = 0`, `Var(ε|x) = σ²(x)`, and the
query noise independent of the context. Then:
(a) `E[(θ(x) − y)²] = ‖θ − f*‖² + E σ²(x)`: the label-based loss contains the irreducible noise term additively;
(b) `|√Δ̄ − ‖θ − f*‖| ≤ ‖f̄ − f*‖`, where `Δ̄ = E_x Δ(x) = ‖θ − f̄‖²`. In particular, if `‖f̄ − f*‖ ≤ η ‖θ − f*‖`
for some `η ∈ [0, 1)`, then `(1−η)² ‖θ − f*‖² ≤ Δ̄ ≤ (1+η)² ‖θ − f*‖²`;
(c) (ambiguity decomposition [@krogh1995neural]) pointwise,
`(f̄(x) − y)² = (1/K) Σ_k (f_k(x) − y)² − A(x)`;
(d) `E[(θ − y)²] − E[(f̄ − y)²] = ‖θ − f*‖² − ‖f̄ − f*‖²`: the label-based *regret* against the committee
cancels the noise but needs labels, while `Δ̄` needs none.

*Proof.* (a) Expand `(θ − f* − ε)²`. The cross term vanishes because `E[ε | x] = 0` and `θ(x)` is independent of the
query noise. (b) Triangle inequality in `L²(P_x)`: `‖θ − f̄‖ ≤ ‖θ − f*‖ + ‖f* − f̄‖` and
`‖θ − f*‖ ≤ ‖θ − f̄‖ + ‖f̄ − f*‖`. Under the condition, `(1−η)‖θ − f*‖ ≤ ‖θ − f̄‖ ≤ (1+η)‖θ − f*‖`; square. (c) For
real numbers `a_k` with mean `ā` and any `y`: `(1/K) Σ (a_k − y)² = (ā − y)² + (1/K) Σ (a_k − ā)²`. The cross term
`2(ā − y)(1/K)Σ(a_k − ā) = 0`. (d) Apply (a) to `θ` and to `f̄`, and subtract. ∎

**Corollary 2 (ranking consistency).** Let `e_c := E_{T∼G_c} ‖θ − f*‖²_T` be the PFN's expected excess error and
`Δ_c := E Δ̄_T` the expected gap in cell `c`. Suppose that for every task, `‖f̄ − f*‖_T ≤ η ‖θ − f*‖_T`. Then
`(1−η)² e_c ≤ Δ_c ≤ (1+η)² e_c`, and for any two cells, `(1+η)² e_a < (1−η)² e_b` implies `Δ_a < Δ_b`. The
disagreement score orders cells by excess error whenever those errors differ by more than a factor of
`((1+η)/(1−η))²`. *Proof.* Integrate (b) over tasks. ∎

*Remark 3 (failure mode).* The condition fails when the committee is worse than the PFN (`η ≥ 1`). This is likely
late in training, or in cells where all classical experts are biased (e.g. extrapolation). There, `Δ` measures the
*committee's* error as much as the PFN's [@burbidge2007active]. This is why Prop. 4 is checked empirically
(Fig. 13, RQ3) rather than assumed.

## 5. Finite-sample accuracy of the estimated sampler

In each adaptation round, the sampler draws `m` independent probe tasks per cell, computes clipped per-task scores
`Y_{c,i} ∈ [0, B]`, and sets `ŝ_c = (1/m) Σ_i Y_{c,i}`.

**Theorem 1.** Assume `θ` is held fixed during the round and the probe tasks are independent. Then, with
probability at least `1 − δ`,
`max_c |ŝ_c − s_c| ≤ ε_m := B √(log(2M/δ)/(2m))` and
`‖q(ŝ) − q(s)‖₁ ≤ (1−γ) ε_m / τ = (1−γ) (B/τ) √(log(2M/δ)/(2m))`.

*Proof.* For fixed `c`, Hoeffding's inequality [@hoeffding1963probability] gives
`P(|ŝ_c − s_c| ≥ ε) ≤ 2 exp(−2mε²/B²)`. A union bound over `M` cells and setting the right-hand side to `δ` gives
`ε_m`. On that event, `osc(ŝ − s) ≤ 2ε_m`, and Prop. 3 gives
`‖q(ŝ) − q(s)‖₁ ≤ (1−γ) · 2ε_m/(2τ)`. ∎

**Corollary 3 (exponential moving average).** Suppose the per-round estimates are combined as
`s̃ = Σ_{r≥0} (1−λ) λ^r ŝ^{(r)}` with EMA factor `λ ∈ [0,1)`. This is the infinite-horizon form, where `r` counts
rounds back from the present. Suppose also that the cell scores are stationary (the same `s_c` in every round) and
all probes are independent. Then Theorem 1 holds with `m` replaced by the effective sample size
`m_eff = m (1+λ)/(1−λ)`. *Proof.* The EMA is a weighted mean `Σ_j w_j Y_j` of independent `[0,B]`
variables with `Σ_j w_j = 1`. Hoeffding's inequality for weighted sums gives
`P(|Σ w_j Y_j − s| ≥ ε) ≤ 2 exp(−2ε²/(B² Σ_j w_j²))`. With per-probe weight `(1−λ)λ^r/m` in round `r` (infinite
horizon), `Σ w² = (1−λ)²/(m(1−λ²)) = (1−λ)/(m(1+λ))`. Union bound and Prop. 3 as before. ∎

*Remark 4.* During training, `θ` (and hence `s`) drifts between rounds. The EMA then trades variance for a bias
that is not bounded here. MAPLE's default `λ = 0.5` keeps the window short (sensitivity in Table 8). The implementation initialises each cell's EMA with its first observation (finite-horizon weights), so in early rounds the effective sample size is smaller than the infinite-horizon `m(1+λ)/(1−λ)` of Corollary 3; Theorem 1 (plain average of one round) is the bound that applies there.

## 6. Bounded distortion of the base prior

**Proposition 5.** Let task losses satisfy `0 ≤ ℓ(θ; T) ≤ L`. For any `θ` and any `q ∈ Δ_M`,
`|E_{G_q} ℓ(θ) − E_{G_u} ℓ(θ)| ≤ L · TV(q, u) ≤ L √(KL(q‖u)/2) ≤ L √((1−γ) osc(s)/(2τ))`.

*Proof.* `E_{G_q} ℓ − E_{G_u} ℓ = Σ_c (q_c − u_c) R_c` with `R_c ∈ [0, L]`. Subtracting `L/2` from each `R_c`
(allowed because `Σ_c (q_c − u_c) = 0`) gives `|Σ_c (q_c − u_c)(R_c − L/2)| ≤ (L/2)‖q − u‖₁ = L·TV(q,u)`. Pinsker's
inequality gives `TV ≤ √(KL/2)`, and Prop. 2(c) gives the last step. ∎

*Remark 5.* So a model trained under the adaptive prior is evaluated on the base prior at a cost of at most
`L·TV(q, u)` relative to its own training distribution. The adaptive prior cannot *silently* move far from the
base prior. With `τ` large relative to `osc(s)`, MAPLE stays close to uniform. Prop. 5 says nothing about real
data, whose distance to any `G_q` is unknown.

## 7. Computational complexity

Let `n` be the context size, `n_q` the number of query points, `d_m` the model width, `L` the depth, and `K` the
committee size. One PFN forward pass costs `O(L[(n² + n n_q) d_m + (n + n_q) d_m²])`. Queries attend to the context
only, so there is no `n_q²` term. One adaptation round costs `M · m · (C_PFN + Σ_k C_fit,k(n, d) + Σ_k C_pred,k)`,
plus `O(M)` for the sampler update. Over `S` training steps with update period `U`, the relative overhead is
`(S/U) · round cost / (S · step cost)`. It is measured, not assumed (Table 6). The defaults use `M = 54` cells and
`K = 5` experts; `m` and `U` are set in `configs/experiments.yaml`. Routing at inference costs `F`-fold
out-of-fold fits of the `K` experts plus `F` PFN passes, and the local variant adds a kNN query
(`O(n log n)` with a tree index).

## 8. Conjectures (not proved; examined empirically)

* **C1 (two-timescale tracking).** If the sampler is updated slowly relative to the PFN's learning (`U` large, `τ`
  large), `q_t` tracks the KL-regularised DRO solution for the current `θ_t`. No convergence result for the
  coupled dynamics is claimed.
* **C2 (noise robustness).** Disagreement-based sampling (C/D) allocates less probability than loss-based sampling (B)
  to high-noise tasks, and matches or beats B at equal compute. Prop. 4(a) motivates this. Tested via RQ2.
* **C3 (transfer).** Improvements on the structural OOD suites carry over to real data only to the extent that
  `𝒫_real` resembles the mathematical prior. The real-data experiments test this; no bound is available.
