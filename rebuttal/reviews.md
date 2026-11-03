Meta Review of Submission31134 by Area Chair vA95
Meta Reviewby Area Chair vA9522 Jul 2026, 10:37 (modified: 23 Jul 2026, 11:31)Senior Area Chairs, Area Chairs, Authors, Reviewers Submitted, Program Chairs, Area Chair vA95Revisions
Metareview:
(a) Summary.

The paper compares pooled ridge/ridgeless regression against ensembles of cluster-specific predictors when data partition into clusters that differ only in covariate distribution (covariate shift, no concept shift) — a case prior work on heterogeneity-motivated ensembling leaves unexplored. Under a well-specified linear model with shared coefficients, all penalties λ ≥ 0 including the ridgeless limit λ = 0, and both random- and fixed-effects coefficient models, the main results are: under random effects, an optimally tuned pooled ridge predictor always outperforms ensembles of individually optimally tuned predictors; under fixed effects, a general risk formula characterizes the roles of the coefficients and covariate-distribution shifts. For ridgeless regression, ensembling helps mainly near interpolation thresholds, where splitting acts as implicit regularization (R3). This generalizes prior i.i.d. risk analyses of bagging and random-partition estimation to covariate shift and cluster-aware partitioning.

(b) Main strengths.

The question — when does cluster-wise ensembling beat pooling if only covariate distributions differ — is clean, relevant, and rarely investigated (R1, R2, R3). All three reviewers and I find the theory solid: derivations are detailed (R2), the pooling–ensembling comparison is mathematically explicit, simulations align well with the theory, and the interpolation-threshold explanation of ridgeless behavior is illuminating (R3). The paper is well written with clearly defined notation (R1; R3 concurs on clarity).

(c) Main weaknesses.

The shared concern is scope versus framing, not correctness. The theory assumes a well-specified linear model, shared coefficients and noise variance, known clusters, and mostly mean-shift heterogeneity with identity/common covariance (Assumption 2.1); R1 questions the practicality of 2.1(iii) and how results deviate under correlation structure, R2 calls the setting narrow, and R3 notes broad language in parts of the paper mismatched to the much more specific assumptions. Simulations are synthetic Gaussian experiments within Assumption 2.1; there is no evaluation outside it, no real-data illustration (R1), and K = 2 in Corollary 2.6 is a special case (R1). Secondary issues: complicated parameters (κ and its derivative) in Theorem 2.5 and a two-page off-diagonal analysis that could be modularized (R2); stacked, hard-to-distinguish figures (R1); optimal ensembling under fixed effects left open (R3).

(d) What would adjust the odds.

I read this as a technically solid contribution answering a well-posed question, and I am positively inclined. For acceptance I would need: (1) narrowed framing — abstract and introduction revised to state Assumption 2.1's scope plainly, per R3's mismatch point; (2) numerical experiments probing robustness beyond Assumption 2.1 — correlated covariates (R1), different noise levels or non-shared coefficients (R2), and K > 2 — to indicate whether the qualitative conclusions persist; (3) a clear statement of what is conjectured versus proven outside the analyzed regime. Item (2) asks for simulations only; real-data benchmarking against generic predictors (e.g., XGBoost, R1) would strengthen the paper but is not required for a theory contribution and is explicitly de-prioritized. Convincing responses on (1)–(2) would make me comfortable recommending acceptance.

Add:
Official Review of Submission31134 by Reviewer NR51
Official Reviewby Reviewer NR5126 Jun 2026, 06:37 (modified: 23 Jul 2026, 09:50)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer NR51Revisions
Summary:
This paper studies asymptotic comparison between two ridge regression: pooled vs ensemble. Authors work on the asymptotic risk formulas for both under the random effect and fixed effects.

Contribution Type: Theory: The main contribution is via theoretical analyses and proofs.
Strengths And Weaknesses:
Strengths: 1. Paper is very easy and clear to understand. Presentation is good with all necessary notation clearly defined. Well-written paper. 2. The research topic is interesting. As much as I know, pooled vs ensemble ridge regression is rarely investigated in the literature.

Weaknesses: please see the question section

Quality: 3: good
Clarity: 3: good
Significance: 2: not good
Originality: 2: not good
Questions:
All the theoretical results are based on the assumption 2.1. However, 2.1 (iii) is too strong in the sense that it is hard to have variance of x to be identity matrix in practice. How do we guarantee this? What if there is a correlation structure? How the theory would deviate if there is a correlation structure? Does the theoretical result still hold? If not hold, how to avoid/improve it? To be a solid paper, all these questions need to be addressed, not just working on the simple case.
Authors have assumption 2.1(i), which limits the covariance structure in the source data. In transfer learning, we typically have some variation in the covariance structure. Again, in practice, how to identify such cases when the is only mean shift? A real data analysis is strongly recommended to demonstrate your usefulness. There are so many different methods that can be used for prediction. It is likely that the simple target-only method will beat your proposed method, for example XGBoost. Then what is the beifit of the proposed method? Also, there are many transfer learning in (high-dimensional) linear regression. Authors should also compare.
K=2 in Corollary 2.6 is not a general case. Considering more complicated case, e.g., K=3-5 can make this paper much stronger.
For all figures, authors should present in another way such that the difference between different scenarios is clearly identified. All lines are stacked together, it is hard to see any benefits of the proposed method.
Limitations:
Overall, the paper is interesting. However, conditions used in the theoretical results are too strong and it is hard to see the usefulness in practice. Also, no real data comparison is given, which makes it hard to see the potential broad influence of the paper.

Rating: 3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
No

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes
Add:
Official Review of Submission31134 by Reviewer 4ovb
Official Reviewby Reviewer 4ovb25 Jun 2026, 10:06 (modified: 23 Jul 2026, 09:50)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer 4ovbRevisions
Summary:
This work presents a detailed analysis comparing pooled ridge regression with the ensemble procedure.

Contribution Type: Theory: The main contribution is via theoretical analyses and proofs.
Strengths And Weaknesses:
strengths: -- The high-level problem of comparing pooling vs ensembling is an interesting and important one. -- The theory part is quite solid, with detailed derivations.

weaknesses: -- The setting under Assumption 2.1 is somehow narrow, as also pointed out by the authors. -- The presentation of the work can be greatly improved. -- The results for the pooled setting rely on complicated parameters, for instance, Kappa and its derivative in Theorem 2.5, even though the authors have taken the effort to evaluate a simple setting in Coro. 2.6.

Quality: 3: good
Clarity: 3: good
Significance: 3: good
Originality: 3: good
Questions:
-- The presentation of proof of Theorem 2.5 can be more streamlined. (1) For instance, in the fixed effects setting, it might be better to separate out the analysis of the off-diagonal terms into a stand-alone technical lemma, as it currently takes more than 2 pages. (2) Also, instead of using multiple "therefore, thus" repeatedly, you might want to first present the intermediate goal along with a roadmap and then present the proof steps toward the goal.

-- I think having some numerical experiments outside of Assumption 2.1 can help provide a better sense of the pooled vs ensembled settings. For instance, how would the two compare under different noise levels or without shared coefficients?

-- Is there a more direct way to show the ridgeless results as the limit of the ridge results?

Limitations:
Yes

Rating: 4: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
Confidence: 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
NA

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes
Add:
Official Review of Submission31134 by Reviewer 2nhs
Official Reviewby Reviewer 2nhs20 Jun 2026, 20:45 (modified: 23 Jul 2026, 09:50)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer 2nhsRevisions
Summary:
This paper compares pooled ridge/ridgeless regression with ensembles of cluster-specific regressors under covariate shift but no concept shift. The setting is a well-specified linear model where clusters share the same regression coefficient but differ in their covariate distributions. In a proportional high-dimensional regime, the authors derive asymptotic risk formulas for the ensemble and pooled predictors under random-effects and fixed-effects assumptions on the coefficient. The main conclusion is that, for optimally tuned ridge regression under random effects, pooling is better than an optimally weighted ensemble. For ridgeless regression, ensembling can help mainly near interpolation thresholds because splitting acts like implicit regularization. Simulations with two Gaussian mean-shift clusters support this aspect-ratio explanation.

Contribution Type: Theory: The main contribution is via theoretical analyses and proofs.
Strengths And Weaknesses:
The paper asks a clean and relevant question. In heterogeneous data analysis, it is common to train separate models by source and ensemble them, but it is not always clear whether this helps when only the covariate distribution changes. The paper isolates this question and gives a precise answer for ridge and ridgeless linear regression. The comparison between pooling and ensembling is mathematically explicit, and the simulations are well aligned with the theory. I also appreciate that the paper explains the ridgeless behavior through interpolation thresholds and implicit regularization.

The main limitation is that the result is narrower than the broad motivation. The theory assumes a well-specified linear outcome model, shared coefficients, shared noise variance, known clusters, and mostly mean-shift covariate heterogeneity with common covariance. The simulations are also synthetic Gaussian experiments with identity covariance. Thus the paper supports a useful theoretical message, but it does not show that the same conclusion holds in broader real heterogeneous-data settings.

There is also some mismatch between broad language and the actual assumptions. For example, parts of the paper sound like they cover arbitrary cluster-specific covariate distributions, but the main assumption is much more specific. The strongest clean comparison is in the random-effects setting, and some fixed-effects questions, especially optimal ensembling, are left open. I think the paper is solid and clear, but its significance is moderate because the scope is quite specialized.

Quality: 3: good
Clarity: 3: good
Significance: 3: good
Originality: 3: good
Questions:
Which pooling-dominance statements hold for general K and general covariance Sigma, and which rely on the simplified 
 comparison?
Please make the abstract and introduction more explicit about the shared conditional model, known clusters, and mean-shift/common-covariance assumptions.
Can the authors add one simulation with covariance shifts or unequal noise levels, even if only to show where the conclusion changes?
In the fixed-effects setting, can the authors give a practical diagnostic for when ensembling may be preferred?
How much would the conclusion change if clusters are estimated from data rather than observed?
Limitations:
Yes. The paper has a useful limitations paragraph. Still, the main limitations should be made more visible in the abstract or introduction because they are important for interpreting the main message.

Rating: 4: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
Confidence: 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
none

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes