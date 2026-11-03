
# Response to Reviewer NR51

Thank you for your comments on the practical scope of the results. Below we correct one point
where our writing invited a misreading, then describe in full the simulations we ran to
relax Assumption 2.1 in the directions you raise.

## 1. We do not assume isotropic covariates

Assumption 2.1(i) allows $\\Sigma$ (the training cluster covariance) to be any positive-definite matrix with bounded spectrum. Theorems 2.3, 2.4, and 2.5, which give the risk of the ensemble and of the pooled learner under both random and
fixed effects, all use that more general definition.

Two things are stated at less than that generality. The first is Assumption 2.1(iii), which
sets $E(x^\\star) = 0$ and $\\mathrm{Var}(x^\\star) = I$ on the test point only, not the training data. 
The isotropic test point simplifies many of the expressions needed in our theorems, which is why this choice was made.
The second is Corollary 2.6, which sets $\\Sigma = I$; that one is about comparing the pooled and ensemble risks directly rather than expressing the risks of each learner separately (which we do for more general positive-definite $\\Sigma$). 

However, you are correct to point out that $\\Sigma$ is the same in every cluster. All the
heterogeneity sits in the means $\\mu_\\ell$. Remark 2.2 claims the argument extends to joint
mean and covariance shift without a formal proof - we have now provided simulation evidence for this setting (below). 

## 2. Supplemental simulations

[See tables.txt]

## 3. Where $K = 2$ is used

Each learner is analyzed for general $K$. Theorem 2.3 gives the ensemble risk, Theorem 2.4
the optimal tuning and weights, Theorem 2.5 the pooled risk, and none of them restricts $K$
or $\\Sigma$.

$K = 2$ is only used when the two learners are directly compared in a ratio, and even there only in the last two parts of
Corollary 2.6 - we will clarify this in the revision.
Part (1) is the general-$K$ part showing the necessary and sufficient condition for
the optimally tuned pooled learner to be asymptotically no worse than the optimally tuned,
optimally weighted ensemble. Part (2) adds $K = 2$ to turn that condition into
an explicit risk ratio, and part (3) adds $K = 2$ with equal cluster sizes $n_1 = n_2$ to
reach the ridgeless form, $(1-\\gamma)/(1-2\\gamma)$ for $\\gamma \\lt 1/2$ in (3a) which we verify in simulations.

The reason for specializing is for tractability and interpretability. Dividing one risk expression by the other at
general $K$ and general $\\Sigma$ leaves a ratio depending on $K$ aspect ratios and on the
spectrum of $\\Sigma$, which is difficult to interpret. At $\\Sigma = I$ and $K = 2$ it collapses
to something that shows directly how the gap moves with the aspect ratio.


## 4. XGBoost comparison and transfer learning

What we claim, for ridge and ridgeless regression, is that splitting by cluster and ensembling
does not beat pooling. We are not claiming ridge beats XGBoost; that is a question about
which function class fits a dataset, and its answer would neither support nor contradict
ours.

There is no source/target asymmetry here and no concept shift.
Estimators such as Li, Cai and Li (2022) correct a source-to-target coefficient difference
that is exactly zero under Assumption 2.1, so their debiasing and aggregation steps collapse
onto pooled estimation. They generalize pooling rather than compete with it, and we will say
so in Related Work. [PLACEHOLDER] numerical confirmation.

In terms of how someone would recognize a mean-shift-only setting - our results suggest this wouldn't be necessary,
since mean shifts (or numerically, covariance shifts) do not change the conclusion that pooling is superior to ensembling. 
The only case in which ensembling helps is for ridgless learners where the pooled is at its interpolation threshold. 


## 5. Figures

Our idea in plotting all of the different parameterizations of covariate structure together in each panel was to display that none of them have meaningful differences with one another, and thus the performance is highly overlapping. However, 
we can add a view in which we isolate each setup with a single panel so they can all be seen.

We are happy to run further configurations during the discussion period.

---

# Response to Reviewer 4ovb

Thank you for the concrete suggestions on the proof of Theorem 2.5; we agree with all of
them. Since your third point asks for experiments outside Assumption 2.1, we give those in
full and keep the proof items brief.

## 1. Experiments outside Assumption 2.1

[See tables.txt]

## 2. Streamlining the proof of Theorem 2.5

Both suggestions make sense, and we will make changes accordingly. First, the off-diagonal analysis will become its own lemma, since it is used in both Theorem 2.3 and Theorem 2.5 and not clearly organized as currently written. 

Second, we will add a roadmap in the beginning of the proof outlining each of the blocks, and organize the blocks into sections with brief statements about what will be shown in each. 


## 3. Getting the ridgeless results as a limit of the ridge results

[Rajarshi]

## 4. The parameters in Theorem 2.5 ($\\kappa$ and its derivative)

[Rajarshi]


## 5. Narrowness of the setting

We agree the framing is more general than the settings in the theorems, and will adjust paper text accordingly. The abstract and introduction will state at the outset that the outcome model is well specified and linear, that
coefficients and noise variance are shared across clusters, that cluster labels are known,
and that covariate heterogeneity enters through mean shifts with a covariance common to all
clusters.
We will also make it more clear that the training cluster covariance $\\Sigma$
itself is not assumed isotropic, since all reviewers read it the other way. 
2.1(i) takes any positive-definite $\\Sigma$ with bounded
spectrum. The isotropy in 2.1(iii) is on the test point, done for analytical tractability of risk formulas.
We will also add a proven-versus-conjecturedlist:

- Proven, general $K$ and general $\\Sigma$: the ensemble and pooled risk formulas (Theorems
  2.3, 2.4, 2.5), random and fixed effects, arbitrary mean shifts.
- Proven, general $K$, $\\Sigma = I$, random effects: the dominance condition (Corollary
  2.6(1)).
- Proven, $K = 2$, $\\Sigma = I$: the explicit ratio (2.6(2)) and the ridgeless
  characterization (2.6(3)), the latter also assuming equal cluster sizes $n_1 = n_2$.
- Numerical only: pooling beating ensembling under heterogeneous or non-identity $\\Sigma$, unequal noise,
  non-shared coefficients and non-Gaussian covariates; the interpolation thresholds;
  fixed-effects optimal ensembling.

We are happy to run further configurations during the discussion period.

---

# Response to Reviewer 2nhs

Thank you for an accurate summary, and for pointing out the mismatch between our
framing and our actual scope. We agree, and take each question in turn.

## Q1. Which statements hold for general $K$ and general $\\Sigma$?

- Proven for general $K$ and general $\\Sigma$, arbitrary mean shifts, random and fixed
  effects: Theorems 2.3 (ensemble risk), 2.4 (optimally tuned, optimally weighted ensemble)
  and 2.5 (pooled risk). We wrote these through $F_{\\Sigma,p}$ and
  $Q_\\gamma(\\rho) = [\\rho\\{I + m_\\gamma(\\rho)\\Sigma\\}]^{-1}$ so that any
  positive-definite $\\Sigma$ with bounded spectrum is carried through. $\\Sigma$ is not
  assumed isotropic; it is assumed common across clusters, with heterogeneity entering
  through $\\mu_\\ell$.
- Proven for general $K$, $\\Sigma = I$, random effects: Corollary 2.6(1), the necessary and
  sufficient condition for pooling beating ensembling
- Proven for $K = 2$, $\\Sigma = I$, random effects: Corollary 2.6(2), the explicit risk ratio in the optimally tuned case, and 2.6(3), the ridgeless ratios, which additionally assume equal cluster sizes $n_1 = n_2$.
- Conjectured with numerical support: dominance for non-identity or cluster-heterogeneous
  $\\Sigma$, unequal noise and non-Gaussian covariates (Q3); behavior at the interpolation
  thresholds; optimal ensembling under fixed effects (Q4).

We will make more explicit that each learner analyzed on its own is general in $K$ and in $\\Sigma$;
$K = 2$ allows for calculation of the explicit ratio between approaches, not the more general finding that pooling is superior to ensembling.


## Q2. Make the abstract and introduction explicit about the assumptions

Agreed, and this is a framing error on our part which we will fix in the revision. The
abstract and introduction will state specifically that we study a well-specified linear outcome model;
coefficients and noise variance shared across clusters; known cluster labels; covariates following a subgaussian
distribution and covariate heterogeneity entering through mean shifts, 
with $\\Sigma$ any positive-definite matrix but common to all clusters; and a centered, isotropic test distribution.

We will also fix the phrases you flagged, such as "arbitrary cluster-specific covariate
distributions" in Section 1.1 and replace with details above.
 We have provided more details as to wording in our AC response.

## Q3. One simulation with covariance shifts or unequal noise levels

[See tables.txt]

## Q4. A practical diagnostic under fixed effects

[Rajarshi]

## Q5. What if clusters were estimated rather than observed?

Our hypothesis is that very little would change, supported by one of the experiments in Section 4,
in which study an ablation in which the data has two genuine clusters but the
ensemble base learners are trained on a uniformly random split bearing no relation to them.
Results were identical to the true-partition results for both ridge and ridgeless, and a
further experiment with no cluster structure gave the same raw MSEs and ratio. Any estimated cluster would lie
between the random split and correct split case, which we show to be equivalent.


___ 

# Response to the Area Chair

Thank you for setting out clearly what would improve the paper. We take the three conditions in
order; supporting detail is in our replies to R1, R2 and R3.

## 1. Narrowed framing: the specific edits

Below is some proposed wording to correct the scope claims. 

Abstract, first change. Currently: "...under a linear outcome model and consider all ridge
penalty values $\\lambda \\ge 0$, including the special case of the ridgeless predictor at
$\\lambda = 0$." Proposed: "We address this case for ridge-regularized least-squares
regression under a well-specified linear outcome model in which the regression coefficients
and the noise variance are shared across clusters, the cluster labels are known, and
covariate heterogeneity enters through cluster-specific mean vectors with a covariance matrix
common to all clusters. We consider all ridge penalty values $\\lambda \\ge 0$, including the
ridgeless predictor at $\\lambda = 0$."

Abstract, second change. Currently: "we argue that under random effects, an optimally tuned
pooled ridge predictor always outperforms ensembles of individually optimally tuned
predictors." Proposed: "we give a necessary and sufficient condition, for any number of
clusters, under which an optimally tuned pooled ridge predictor is asymptotically no worse
than the optimally tuned, optimally weighted ensemble. We verify that condition in closed
form for two clusters and numerically for up to five." 

Section 1.1, first change. Currently: "...under a well-specified linear outcome model and
arbitrary cluster-specific covariate distributions." Proposed: "...under a well-specified
linear outcome model in which each cluster has its own mean vector and all clusters share a
single covariance matrix $\\Sigma$, which may be any positive-definite matrix with bounded
spectrum." This is also the phrase R3 flagged.

Section 1.1, second change. Currently: "...uniformly smaller (asymptotic) prediction risk
than the ensemble built on the true cluster partitions. This holds independently of the
angles, norms, and covariance structures of the cluster means." Proposed: "we prove that the
pooled predictor has smaller asymptotic prediction risk than the ensemble built on the true
cluster partition, under random effects with $\\Sigma = I$. The gap does not depend on the
angles or norms of the cluster means."

Section 2.4. The following is is the sentence that caused the confusion about the scope of the $K = 2$ vs 
more general results, so we propose replacing it. Currently: "Setting $K = 2$, $\\Sigma = I$, and random effects on
$\\beta$ admits a clean comparison and lets us quantify the precise benefit of the pooled
learner over the optimally tuned, optimally weighted ensemble." Proposed: "Under $\\Sigma = I$
and random effects on $\\beta$, part (1) of the corollary below gives a necessary and
sufficient condition for the pooled learner to dominate at any number of clusters $K$. Restricting to
$K = 2$ in part (2), and equal cluster sizes in the ridgeless case of part (3), admits a clean
comparison and lets us quantify the precise benefit of the pooled learner over the optimally
tuned, optimally weighted ensemble."

We will also move the limitations paragraph up in the paper to more clearly define scope from the beginning.

## 2. Two clarifications, since all three reviews read Assumption 2.1 as tighter than it is

Assumption 2.1 does not require isotropic covariates: 2.1(i) allows any positive-definite
$\\Sigma$ with bounded spectrum, and Theorems 2.3, 2.4 and 2.5 hold at that generality. The
isotropy in 2.1(iii) is on the test point - our primary restriction is that $\\Sigma$ is common to every cluster.
We have responded to this comment for all reviewers, and will clarify it in the revision.

Each learner alone is analyzed for general $K$: Theorems 2.3 and 2.4 for the ensemble, 2.5 for the pooled learner. Corollary 2.6 then divides by $K$ (as we have stated above). 

## 3. Numerical experiments beyond Assumption 2.1

[See tables.txt]


## 4. Conjectured versus proven

To be added to the contributions section:

- Proven, general $K$ and general $\\Sigma$, random and fixed effects: the ensemble and pooled
  risk formulas (Theorems 2.3, 2.4, 2.5).
- Proven, general $K$, $\\Sigma = I$, random effects: the dominance condition (Corollary
  2.6(1)).
- Proven, $K = 2$, $\\Sigma = I$: the explicit ratio (2.6(2)) and the ridgeless
  characterization (2.6(3)), the latter also assuming equal cluster sizes $n_1 = n_2$.
- Conjectured, numerically supported: dominance under non-identity or cluster-heterogeneous
  $\\Sigma$, unequal noise, a simple case of non-shared coefficients,  non-Gaussian covariates.


Another change, which we detailed in our replies to the second reviewer: we will make the off-diagonal leave-two-out argument a standalone lemma and organize the proof with a roadmap and clearer sections.
