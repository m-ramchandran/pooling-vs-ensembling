<!-- Copied verbatim from OpenReview, submission 31134. Math rendered on the site was lost in copying. -->

Paper Decision
Decisionby Program Chairs24 Sept 2026, 04:27 (modified: 24 Sept 2026, 09:29)Program Chairs, AuthorsRevisions
Decision: Accept (poster)
Metareview:
(a) Summary.

The paper compares pooled ridge/ridgeless regression against ensembles of cluster-specific predictors when data partition into clusters that differ only in covariate distribution (covariate shift, no concept shift) — a case prior work on heterogeneity-motivated ensembling leaves unexplored. Under a well-specified linear model with shared coefficients, all penalties λ ≥ 0 including the ridgeless limit λ = 0, and both random- and fixed-effects coefficient models, the main results are: under random effects, an optimally tuned pooled ridge predictor always outperforms ensembles of individually optimally tuned predictors; under fixed effects, a general risk formula characterizes the roles of the coefficients and covariate-distribution shifts. For ridgeless regression, ensembling helps mainly near interpolation thresholds, where splitting acts as implicit regularization (R3). This generalizes prior i.i.d. risk analyses of bagging and random-partition estimation to covariate shift and cluster-aware partitioning.

(b) Main strengths.

The question — when does cluster-wise ensembling beat pooling if only covariate distributions differ — is clean, relevant, and rarely investigated (R1, R2, R3). All three reviewers and I find the theory solid: derivations are detailed (R2), the pooling–ensembling comparison is mathematically explicit, simulations align well with the theory, and the interpolation-threshold explanation of ridgeless behavior is illuminating (R3). The paper is well written with clearly defined notation (R1; R3 concurs on clarity).

(c) Main weaknesses.

The shared concern is scope versus framing, not correctness. The theory assumes a well-specified linear model, shared coefficients and noise variance, known clusters, and mostly mean-shift heterogeneity with identity/common covariance (Assumption 2.1); R1 questions the practicality of 2.1(iii) and how results deviate under correlation structure, R2 calls the setting narrow, and R3 notes broad language in parts of the paper mismatched to the much more specific assumptions. Simulations are synthetic Gaussian experiments within Assumption 2.1; there is no evaluation outside it, no real-data illustration (R1), and K = 2 in Corollary 2.6 is a special case (R1). Secondary issues: complicated parameters (κ and its derivative) in Theorem 2.5 and a two-page off-diagonal analysis that could be modularized (R2); stacked, hard-to-distinguish figures (R1); optimal ensembling under fixed effects left open (R3).

During the rebuttal, the author have addressed the following questions: (1) narrowed framing — abstract and introduction revised to state Assumption 2.1's scope plainly, per R3's mismatch point; (2) numerical experiments probing robustness beyond Assumption 2.1 — correlated covariates (R1), different noise levels or non-shared coefficients (R2), and K > 2 — to indicate whether the qualitative conclusions persist; (3) a clear statement of what is conjectured versus proven outside the analyzed regime. Item (2) asks for simulations only; real-data benchmarking against generic predictors (e.g., XGBoost, R1) would strengthen the paper but is not required for a theory contribution and is explicitly de-prioritized.

Final Justification:
I recommend acceptance.

On the other hand, overall enthusiasm for this paper is low, mainly because it appears to address a specialized problem whose impact is unclear. If we have too many acceptances and we need to revert some, this could be a candidate.

Citation Checker by Program Chairs
Citation Checkerby Program Chairs04 Aug 2026, 11:19 (modified: 04 Sept 2026, 22:20)Program Chairs, Senior Area Chairs, Area Chairs, AuthorsRevisions
Comment:
Automatically flagged references
Here we list the references that were automatically flagged by our checker. Note that our checker may miss many hallucinated references, and some of the listed references may have been flagged erroneously. It is therefore crucial that you check them manually, following the guidance above. We rely on the Area Chairs to decide whether these constitute violations of responsible authorship.

For papers falling into either of the following categories, we expect your default recommendation to be a rejection on the grounds of irresponsible authorship. This is strong presumption rather than a strict rule; as the Area Chair, you retain discretion, and may recommend otherwise where the specific circumstances of a paper warrant it:

The paper contains at least two hallucinated references according to the criteria described below the report.
The paper contains one hallucinated reference and there is some additional evidence of AI slop.
If a paper has only one problematic reference, but no additional concerns about the quality of the paper or its presentation, please simply alert the authors in your meta-review.

Please complete the form “Hallucinated References Investigation” to communicate the results of your investigation prior to completing your meta reviews.

Citation Checker Report
Reference Check Summary
0/25 problematic references.

Hallucinated/problematic reference criteria
These are adopted from ICLR and ICML 2026 hallucinated reference guidance:

If you are unable to find any document or publication with a title anywhere near the one provided in the reference, it should be considered hallucinated. Minor differences in title should not be considered evidence of a hallucinated reference.
If the reference's authors are only slightly incorrect, e.g., a few subtle misspellings, missing a few authors, or wrong authorship order, we do not consider this to imply a hallucinated reference, but please let the submission's authors know in your meta-review that they should fix this issue.
If the author list has additional issues, e.g., added authors that are not actually authors of that publication or missing a large fraction of authors, then the reference should be considered hallucinated.
If the reference's venue is a real venue but is incorrect for that reference, we do not consider this to imply a hallucinated reference, but please tell the submission's authors in your meta-review, these include wrong journal name, arXiv ID, etc.
If there is no evidence that the reference's venue exists, the reference should be considered hallucinated.
Finally, we note that there are many generic references that refer to existing artifacts (e.g. "GPT-4o, OpenAI et al. 2024") but do not refer to any real document. These are fine.
Even with the above guidelines in mind, there will likely be some borderline cases. Please bring those up with your SAC if you need additional guidance.
Official Review of Submission31134 by Reviewer NR51
Official Reviewby Reviewer NR5126 Jun 2026, 06:37 (modified: 24 Sept 2026, 10:16)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer NR51Revisions
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
Final Justification:
I still have concerns about the assumptions. From the simulation results, authors confirm that target is still isotropic Gaussian, which makes the results weak. I will keep my score unchanged.

Rebuttal by Authors
Rebuttalby Authors (Maya Ramchandran, Rajarshi Mukherjee)27 Jul 2026, 20:40 (modified: 28 Jul 2026, 07:28)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:
Thank you for your comments on the practical scope of the results. Below, we correct one point where our writing invited a misreading, then describe in full the simulations we ran to relax Assumption 2.1 in the directions you raise.

1. We do not assume isotropic covariates
Assumption 2.1(i) allows 
 (the training cluster covariance) to be any positive-definite matrix with spectrum bounded away from zero and infinity. Theorems 2.3, 2.4, and 2.5, which give the risk of the ensemble and of the pooled learner under both random and fixed effects, all use that more general definition.

Two things are stated at less than that generality. The first is Assumption 2.1(iii), which sets 
 and 
 on the test point only, not the training data. The isotropic test point simplifies many of the expressions needed in our theorems, which is why this choice was made. The second is Corollary 2.6, which sets 
; that one compares the pooled and ensemble risks directly rather than expressing the risks of each learner separately (which we do for more general positive-definite 
). Here too, the isotropic design allows us to simplify the general formulae presented in Theorems 2.3, 2.4, and 2.5 to illustrate the relative performance of pooled versus merged analysis as a function of comparatively easier-to-understand parameters.

However, you are correct to point out that 
 is the same in every cluster. All the heterogeneity sits in the means 
. Remark 2.2 claims the argument extends to joint mean and covariance shift without a formal proof - we have now provided simulation evidence for this setting (below).

2. Supplemental simulations
We have done some additional simulations that go beyond the mean shift assumption, including heterogeneous covariances, different noise levels, opposing true signal between clusters, and a higher number of clusters. Below are examples for ridge regression learners when varying the covariance structure and noise level - a full description of the experiments is in the comment for Reviewer 4ovb. For the experiments below, as well as other variations for both ridge and ridgeless, we find that the conclusions presented in the paper continue to hold - we will provide full results in the revision.

Varying covariance structure:

k=2	k=3	k=4	k=5
identity	1.12 ± 0.02	1.13 ± 0.02	1.12 ± 0.02	1.10 ± 0.01
diagonal, per-cluster	1.11 ± 0.02	1.11 ± 0.01	1.09 ± 0.01	1.06 ± 0.005
equicorr rho=0.2	1.15 ± 0.03	1.16 ± 0.02	1.15 ± 0.02	1.14 ± 0.01
equicorr rho=0.5	1.15 ± 0.03	1.19 ± 0.02	1.18 ± 0.02	1.18 ± 0.02
equicorr rho=0.8	1.25 ± 0.07	1.25 ± 0.04	1.25 ± 0.04	1.28 ± 0.05
equicorr, per-cluster	1.12 ± 0.02	1.16 ± 0.02	1.18 ± 0.02	1.19 ± 0.02
factor (shared W)	1.01 ± 0.002	1.00 ± 0.001	1.00 ± 0.001	1.00 ± 0.001
factor (per-cluster W)	1.01 ± 0.002	1.01 ± 0.001	1.00 ± 0.001	1.00 ± 0.001
nonlinear	1.19 ± 0.03	1.28 ± 0.04	1.29 ± 0.03	1.30 ± 0.03
Varying noise levels between clusters:

k=2	k=3	k=4	k=5
identity	1.00 ± 0.01	1.02 ± 0.01	1.04 ± 0.01	1.04 ± 0.01
diagonal, per-cluster	1.02 ± 0.01	1.05 ± 0.01	1.05 ± 0.01	1.04 ± 0.005
equicorr rho=0.2	1.01 ± 0.01	1.04 ± 0.01	1.04 ± 0.01	1.04 ± 0.01
equicorr rho=0.5	1.01 ± 0.01	1.05 ± 0.01	1.07 ± 0.01	1.06 ± 0.01
equicorr rho=0.8	1.07 ± 0.03	1.14 ± 0.03	1.11 ± 0.02	1.12 ± 0.02
equicorr, per-cluster	1.00 ± 0.01	1.03 ± 0.01	1.04 ± 0.01	1.05 ± 0.01
factor (shared W)	1.01 ± 0.002	1.00 ± 0.001	1.00 ± 0.001	1.00 ± 0.001
factor (per-cluster W)	1.01 ± 0.002	1.00 ± 0.001	1.00 ± 0.001	1.00 ± 0.001
nonlinear	1.04 ± 0.01	1.09 ± 0.02	1.13 ± 0.02	1.13 ± 0.01
3. Where 
 is used
Each learner is analyzed for general 
. Theorem 2.3 gives the ensemble risk, Theorem 2.4 the optimal tuning and weights, Theorem 2.5 the pooled risk, and none of them restricts 
 or 
.

 is only used when the two learners are directly compared in a ratio, and even there only in the last two parts of Corollary 2.6 - we will clarify this in the revision. Part (1) is the general-
 part showing the necessary and sufficient condition for the optimally tuned pooled learner to be asymptotically no worse than the optimally tuned, optimally weighted ensemble. Part (2) adds 
 to turn that condition into an explicit risk ratio, and part (3) adds 
 with equal cluster sizes 
 to reach the ridgeless form, 
 for 
 in (3a), which we verify in simulations.

The reason for specializing is for tractability and interpretability. Dividing one risk expression by the other at general 
 and general 
 leaves a ratio depending on 
 aspect ratios and on the spectrum of 
, which is difficult to interpret. At 
 and 
 it collapses to something that shows directly how the gap moves with the aspect ratio.

4. XGBoost comparison and transfer learning
What we claim, for ridge and ridgeless regression, is that splitting by cluster and ensembling does not beat pooling. We are not claiming ridge beats XGBoost; that is a question about which function class fits a dataset, and its answer would neither support nor contradict ours.

There is no source/target asymmetry here and no concept shift. Estimators such as Li, Cai and Li (2022) correct a source-to-target coefficient difference that is exactly zero under Assumption 2.1, so their debiasing and aggregation steps collapse onto pooled estimation. They generalize pooling rather than compete with it, and we will say so in Related Work. [PLACEHOLDER] numerical confirmation.

In terms of how someone would recognize a mean-shift-only setting - our results suggest this wouldn't be necessary, since mean shifts (or numerically, covariance shifts) do not change the conclusion that pooling is superior to ensembling. The only case in which ensembling helps is for ridgeless learners where the pooled is at its interpolation threshold.

5. Figures
Our idea in plotting all of the different parameterizations of covariate structure together in each panel was to display that none of them have meaningful differences with one another, and thus the performance is highly overlapping. However, we can also add a view in which we isolate each setup with a single panel so they can all be seen.

We are happy to run further configurations during the discussion period.

Official Comment by Reviewer NR51
Official Commentby Reviewer NR5103 Aug 2026, 20:05Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Comment:
Thanks for the response. For your response 2 about the supplemental simulations, what is the varying covariance structure for? The source or the target?

For your assumption 2.1, by saying that the simpler assumption can be extended to more general forms is not mathematically rigorous. Give this paper's contribution is built on exact theorems, I'd like either an actual derivation for the general forms or an explicit statement that this is conjectural, not established.

 Replying to Official Comment by Reviewer NR51
Official Comment by Authors
Official Commentby Authors (Maya Ramchandran, Rajarshi Mukherjee)04 Aug 2026, 04:45Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Comment:
We sincerely thank the reviewer for helping us think through response 2 in more detail. For our response 2, the varying covariance structure is for the source, where covariance structures vary across source clusters. The target remains an isotropic Gaussian across the various simulations. For assumption 2.1 and the general framing, we agree that in the revision we will clarify the language to make it explicitly clear that the general forms are conjectured, with some numerical support from these additional simulations.

Official Review of Submission31134 by Reviewer 4ovb
Official Reviewby Reviewer 4ovb25 Jun 2026, 10:06 (modified: 24 Sept 2026, 10:16)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer 4ovbRevisions
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
Rebuttal by Authors
Rebuttalby Authors (Maya Ramchandran, Rajarshi Mukherjee)27 Jul 2026, 20:41 (modified: 28 Jul 2026, 07:28)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:
Thank you for the concrete suggestions on the proof of Theorem 2.5; we agree with all of them. Since your third point asks for experiments outside Assumption 2.1, we give those in full and keep the proof items brief.

1. Experiments outside Assumption 2.1
We have done several additional simulations for both ridge and ridgeless learners, including varying the covariance structure between clusters, noise level between clusters, and coefficient vector across clusters. For brevity, we show a few examples for ridge regression learners below - throughout, we find that the conclusions from the paper continue to hold for both ridge and ridgeless regression. We will provide full simulation results in the revision.

Design:

 clusters, 
 training (pooled sample size is 
) and 200 test points each, every scenario run over 100 replicates. Means are drawn from 
, so clusters are well separated.

A point from cluster 
 is 
, where 
 has mean zero and covariance 
. The seven different simulation scenarios below differ only in the distribution of 
, and the first four give every cluster the same covariance:

identity: 
.
equicorrelated: 
 with 
, at 
 and 
.
spiked: 
, with 
 a 
 standard Gaussian drawn once per replicate. For this structure and the cluster-specific version below, 
 is drawn in 
, following Hastie et al. (2022).
non-gaussian: 
 with 
 and 
 a 
 matrix of 
 entries, the constant restoring unit variance per coordinate. There is no 
 here, so this case leaves Assumption 2.1 altogether. The outcome is still exactly linear in the observed 
.
Three let the covariance differ across clusters, which is the case Remark 2.2 leaves open:

diagonal, per-cluster scale: 
, with 
 the first 
 of 
, so the spread across clusters is 
 at 
 and 
 at 
.
equicorrelated, cluster-specific 
: 
, with 
 the first 
 of 
.
low-rank perturbation, cluster-specific 
: 
 with 
 and 
 standard Gaussian, so the perturbed directions themselves move between clusters.
Outcome: 
 with 
, and 
 either shared or 
. Here 
 is scaled so that 
 averages 1 over clusters in every structure. Noise is either 
 throughout, or varies across clusters spanning 
 with 
, holding both the average level and the max/min ratio fixed as 
 varies.

We report ensemble/pooled test-MSE ratio ± 95% t C (ratio > 1 means that the pooled is better than the ensemble), and show results below for a few of the scenarios listed above that we ran.

Table 1: Covariance structure × k — ridge, equal noise variance, shared beta

k=2	k=3	k=4	k=5
identity	1.12 ± 0.02	1.13 ± 0.02	1.12 ± 0.02	1.10 ± 0.01
diagonal, per-cluster	1.11 ± 0.02	1.11 ± 0.01	1.09 ± 0.01	1.06 ± 0.005
equicorr rho=0.2	1.15 ± 0.03	1.16 ± 0.02	1.15 ± 0.02	1.14 ± 0.01
equicorr rho=0.5	1.15 ± 0.03	1.19 ± 0.02	1.18 ± 0.02	1.18 ± 0.02
equicorr rho=0.8	1.25 ± 0.07	1.25 ± 0.04	1.25 ± 0.04	1.28 ± 0.05
equicorr, per-cluster	1.12 ± 0.02	1.16 ± 0.02	1.18 ± 0.02	1.19 ± 0.02
factor (shared W)	1.01 ± 0.002	1.00 ± 0.001	1.00 ± 0.001	1.00 ± 0.001
factor (per-cluster W)	1.01 ± 0.002	1.01 ± 0.001	1.00 ± 0.001	1.00 ± 0.001
nonlinear	1.19 ± 0.03	1.28 ± 0.04	1.29 ± 0.03	1.30 ± 0.03
Table 2: Covariance structure × k — ridge, 16× spread in sigma^2, shared beta

k=2	k=3	k=4	k=5
identity	1.00 ± 0.01	1.02 ± 0.01	1.04 ± 0.01	1.04 ± 0.01
diagonal, per-cluster	1.02 ± 0.01	1.05 ± 0.01	1.05 ± 0.01	1.04 ± 0.005
equicorr rho=0.2	1.01 ± 0.01	1.04 ± 0.01	1.04 ± 0.01	1.04 ± 0.01
equicorr rho=0.5	1.01 ± 0.01	1.05 ± 0.01	1.07 ± 0.01	1.06 ± 0.01
equicorr rho=0.8	1.07 ± 0.03	1.14 ± 0.03	1.11 ± 0.02	1.12 ± 0.02
equicorr, per-cluster	1.00 ± 0.01	1.03 ± 0.01	1.04 ± 0.01	1.05 ± 0.01
factor (shared W)	1.01 ± 0.002	1.00 ± 0.001	1.00 ± 0.001	1.00 ± 0.001
factor (per-cluster W)	1.01 ± 0.002	1.00 ± 0.001	1.00 ± 0.001	1.00 ± 0.001
nonlinear	1.04 ± 0.01	1.09 ± 0.02	1.13 ± 0.02	1.13 ± 0.01
Table 3: Shared vs opposite-sign beta — ridgeless, identity, equal noise

k=2	k=3	k=4	k=5
beta shared	1.13 ± 0.03	1.14 ± 0.02	1.14 ± 0.02	1.11 ± 0.01
beta opposite-sign	1.30 ± 0.07	1.43 ± 0.07	1.60 ± 0.06	1.72 ± 0.07
2. Streamlining the proof of Theorem 2.5
Both suggestions make sense, and we will make changes accordingly. First, the off-diagonal analysis will become its own lemma, since it is used in both Theorem 2.3 and Theorem 2.5 and not clearly organized as currently written.

Second, we will add a roadmap in the beginning of the proof outlining each of the blocks, and organize the blocks into sections with brief statements about what will be shown in each.

3. Getting the ridgeless results as a limit of the ridge results
We thank the reviewer for this point. It is indeed possible to first have an approximation of the ridgeless case with a proper ridge-regularized case -- i.e., a bound on the difference of prediction errors for the ridgeless estimators and ridge-regularized predictors as a function of the penalization parameters. A careful control of this difference then allows one to take the regularization to zero at a suitable rate to conclude on the ridgeless limit. This is exactly our proof strategy for the fixed effects. However, for random effects one can work with a simpler argument as demonstrated in the proof of Theorem 2.5 part (1).

4. Narrowness of the setting
We agree the framing is more general than the settings in the theorems, and will adjust the paper text accordingly. The abstract and introduction will state at the outset that the outcome model is well specified and linear, that coefficients and noise variance are shared across clusters, that cluster labels are known, and that covariate heterogeneity enters through mean shifts with a covariance common to all clusters. We will also make it clearer that the training cluster covariance 
 itself is not assumed isotropic, since all reviewers read it the other way. 2.1(i) takes any positive-definite 
 with bounded spectrum. The isotropy in 2.1(iii) is on the test point, done for analytical tractability of risk formulas. We will also add a proven-versus-conjecturedlist:

Proven, general 
 and general 
: the ensemble and pooled risk formulas (Theorems 2.3, 2.4, 2.5), random and fixed effects, arbitrary mean shifts.
Proven, general 
, 
, random effects: the dominance condition (Corollary 2.6(1)).
Proven, 
, 
: the explicit ratio (2.6(2)) and the ridgeless characterization (2.6(3)), the latter also assuming equal cluster sizes 
.
Numerical only: pooling beating ensembling under heterogeneous or non-identity 
, unequal noise, non-shared coefficients, and non-Gaussian covariates; the interpolation thresholds; fixed-effects optimal ensembling.
We are happy to run further configurations during the discussion period.

Official Review of Submission31134 by Reviewer 2nhs
Official Reviewby Reviewer 2nhs20 Jun 2026, 20:45 (modified: 24 Sept 2026, 10:16)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer 2nhsRevisions
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
Final Justification:
The authors well address my questions, so I maintain my positive score.

Rebuttal by Authors
Rebuttalby Authors (Maya Ramchandran, Rajarshi Mukherjee)27 Jul 2026, 20:44 (modified: 28 Jul 2026, 07:28)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:
Thank you for an accurate summary, and for pointing out the mismatch between our framing and our actual scope. We agree, and take each question in turn.

Q1. Which statements hold for general 
 and general 
?
Proven for general 
 and general 
, arbitrary mean shifts, random and fixed effects: Theorems 2.3 (ensemble risk), 2.4 (optimally tuned, optimally weighted ensemble) and 2.5 (pooled risk). We wrote these through 
 and 
 so that any positive-definite 
 with bounded spectrum is carried through. 
 is not assumed isotropic; it is assumed common across clusters, with heterogeneity entering through 
.
Proven for general 
, 
, random effects: Corollary 2.6(1), the necessary and sufficient condition for pooling beating ensembling
Proven for 
, 
, random effects: Corollary 2.6(2), the explicit risk ratio in the optimally tuned case, and 2.6(3), the ridgeless ratios, which additionally assume equal cluster sizes 
.
Conjectured with numerical support: dominance for non-identity or cluster-heterogeneous 
, unequal noise and non-Gaussian covariates (Q3); behavior at the interpolation thresholds; optimal ensembling under fixed effects (Q4).
We will make more explicit that each learner analyzed on its own is general in 
 and in 
; 
 allows for calculation of the explicit ratio between approaches, not the more general finding that pooling is superior to ensembling.

Q2. Make the abstract and introduction explicit about the assumptions
Agreed, and this is a framing error on our part which we will fix in the revision. The abstract and introduction will state specifically that we study a well-specified linear outcome model; coefficients and noise variance shared across clusters; known cluster labels; covariates following a subgaussian distribution and covariate heterogeneity entering through mean shifts, with 
 any positive-definite matrix but common to all clusters; and a centered, isotropic test distribution.

We will also fix the phrases you flagged, such as "arbitrary cluster-specific covariate distributions" in Section 1.1, and replace them with details above. We have provided more details as to wording in our AC response.

Q3. One simulation with covariance shifts or unequal noise levels
We have done some additional simulations that go beyond the mean shift assumption, including heterogeneous covariances, different noise levels, opposing true signal between clusters, and a higher number of clusters. Below are examples for ridge regression learners when varying the covariance structure and noise level - a full description of the experiments is in the comment for Reviewer 4ovb. For the experiments below, as well as other variations for both ridge and ridgeless, we find that the conclusions presented in the paper continue to hold - we will provide full results in the revision.

Varying covariance structure:

k=2	k=3	k=4	k=5
identity	1.12 ± 0.02	1.13 ± 0.02	1.12 ± 0.02	1.10 ± 0.01
diagonal, per-cluster	1.11 ± 0.02	1.11 ± 0.01	1.09 ± 0.01	1.06 ± 0.005
equicorr rho=0.2	1.15 ± 0.03	1.16 ± 0.02	1.15 ± 0.02	1.14 ± 0.01
equicorr rho=0.5	1.15 ± 0.03	1.19 ± 0.02	1.18 ± 0.02	1.18 ± 0.02
equicorr rho=0.8	1.25 ± 0.07	1.25 ± 0.04	1.25 ± 0.04	1.28 ± 0.05
equicorr, per-cluster	1.12 ± 0.02	1.16 ± 0.02	1.18 ± 0.02	1.19 ± 0.02
factor (shared W)	1.01 ± 0.002	1.00 ± 0.001	1.00 ± 0.001	1.00 ± 0.001
factor (per-cluster W)	1.01 ± 0.002	1.01 ± 0.001	1.00 ± 0.001	1.00 ± 0.001
nonlinear	1.19 ± 0.03	1.28 ± 0.04	1.29 ± 0.03	1.30 ± 0.03
Varying noise levels between clusters:

k=2	k=3	k=4	k=5
identity	1.00 ± 0.01	1.02 ± 0.01	1.04 ± 0.01	1.04 ± 0.01
diagonal, per-cluster	1.02 ± 0.01	1.05 ± 0.01	1.05 ± 0.01	1.04 ± 0.005
equicorr rho=0.2	1.01 ± 0.01	1.04 ± 0.01	1.04 ± 0.01	1.04 ± 0.01
equicorr rho=0.5	1.01 ± 0.01	1.05 ± 0.01	1.07 ± 0.01	1.06 ± 0.01
equicorr rho=0.8	1.07 ± 0.03	1.14 ± 0.03	1.11 ± 0.02	1.12 ± 0.02
equicorr, per-cluster	1.00 ± 0.01	1.03 ± 0.01	1.04 ± 0.01	1.05 ± 0.01
factor (shared W)	1.01 ± 0.002	1.00 ± 0.001	1.00 ± 0.001	1.00 ± 0.001
factor (per-cluster W)	1.01 ± 0.002	1.00 ± 0.001	1.00 ± 0.001	1.00 ± 0.001
nonlinear	1.04 ± 0.01	1.09 ± 0.02	1.13 ± 0.02	1.13 ± 0.01
Q4. A practical diagnostic under fixed effects
We sincerely thank the reviewer for bringing up this important aspect of practically understanding the relative merits of pooled versus merged learners in the fixed effects setting. Indeed, the subtlety in this regime pertains to the fact the prediction errors depend on complicated quantities such as 
 for 
, 
 and 
 as defined in lines 160-166 of page 4 of our original manuscript. In the proportional asymptotic regime, without further structural assumptions on 
 estimation of these quantities from data can be subtle. One can then rely on recent results from literature (Verzelen and Gassiat'18, Kong and Valiant'18, Chen, Liu, and Mukherjee'24, Celentano & Wainwright'23, Bellec & Zhang'23, and references therein)) to aim at consistent estimation of these quantities bilinear forms of 
 to get a practical guidance on the relative performance between pooled versus merged predictors.

Q5. What if clusters were estimated rather than observed?
Our hypothesis is that very little would change, supported by one of the experiments in Section 4, in which we study an ablation in which the data has two genuine clusters but the ensemble base learners are trained on a uniformly random split bearing no relation to them. Results were identical to the true-partition results for both ridge and ridgeless, and a further experiment with no cluster structure gave the same raw MSEs and ratio. Any estimated cluster would lie between the random split and correct split case, which we show to be equivalent.

Official Comment by Reviewer 2nhs
Official Commentby Reviewer 2nhs03 Aug 2026, 10:21Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Comment:
Thank you for the detailed rebuttal. The clarification that Theorems 2.3–2.5 hold for general K and a common positive-definite 
, while the explicit risk ratios require stronger assumptions, resolves my main question about the scope of the results. The planned revisions to the abstract and introduction, as well as the additional covariance- and noise-shift experiments, also address most of my framing and empirical concerns. Please distinguish clearly between conclusions established by theory and robustness supported only by the new simulations. Two points remain partially unresolved. For Q4, the cited results suggest how the relevant fixed-effects quantities might be estimated, but they do not yet give a concrete practical diagnostic. For Q5, the random-split ablation is informative, but it does not establish that every data-estimated clustering must have risk between the random and true partitions. This claim should be softened unless further theory or evidence is provided. Overall, the rebuttal resolves most of my concerns and strengthens my positive assessment. I retain my borderline-accept recommendation because the remaining limitations concern extensions beyond the paper’s proved scope, not the validity of its main theoretical results.

 Replying to Official Comment by Reviewer 2nhs
Discussions on response
Official Commentby Authors (Maya Ramchandran, Rajarshi Mukherjee)04 Aug 2026, 04:43Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Comment:
We sincerely thank the reviewer for the detailed follow up questions encouraging us think things through in more detail.

We agree with your comment about clearly distinguishing between conclusions established by the theory and the simulation robustness experiments, and will clarify that the extensions to assumption 2.1 (the more general framing) are conjectured, not proven, with some numerical support from these additional simulations.

For Q4, once the relevant fixed-effects quantities are estimated along with valid joint uncertainty quantification (as allowed by the cited literature), one can construct a test to compare membership across different regimes. Since our results pertain to first-order sharp asymptotics of the prediction errors of the learners under study, we will only need the tests to be consistent against fixed alternatives—which can be achieved by this testing procedure. We will be happy to include a detailed discussion with precise mathematical details in a revised manuscript.

We also agree that, similarly to the first point, we should clarify that the random-split ablation provides some numerical evidence that the cluster structure (or lack thereof) doesn’t change the relationship between the pooled and ensemble learners, but as you pointed out, it is not proof that every possible cluster structure would behave the same way. We will temper the claim and present this finding as a conjecture, not a proof.
