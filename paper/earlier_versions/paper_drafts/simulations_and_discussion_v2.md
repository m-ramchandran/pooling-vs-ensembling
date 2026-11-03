# Simulation Study

We conduct an extensive simulation study to validate our theoretical results and illustrate the three regimes identified in Section 4. We describe the common setup, then present results for ridgeless and ridge estimators, followed by a discussion of implications.

## Setup

We fix $n = 2000$ total training observations split equally across $k = 2$ clusters ($n_1 = n_2 = 1000$). The coefficient vector $\beta \in \mathbb{R}^p$ is drawn from $\mathcal{N}(0, I_p)$ and normalized to unit norm ($\|\beta\| = 1$). For cluster $b \in \{1, 2\}$, training data is generated as $x_i \sim \mathcal{N}(\mu_b, I_p)$ with response $y_i = x_i^\top \beta + \varepsilon_i$, where $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$. We set $\sigma = 1/\sqrt{5}$ so that $\text{SNR} = \|\beta\|^2 / \sigma^2 = 5$, matching the highest SNR considered by Hastie et al. (2022). Test data ($n_{\text{test}} = 1000$) is drawn from the same mixture distribution. All results are averaged over 20 independent trials with 95\% confidence intervals.

The cluster means $\mu_1, \mu_2$ are unit-norm random vectors with angle $\theta = 60°$ and equal norms. The **federated learner** fits a separate model on each cluster and averages predictions: $\hat{y}_{\text{fed}} = \frac{1}{2}(\hat{y}_1 + \hat{y}_2)$. The **centralized learner** fits a single model on the pooled data. For linear models, prediction averaging is equivalent to parameter averaging (as in FedAvg). Our primary metric is the **MSE ratio**: $R = \text{MSE}_{\text{fed}} / \text{MSE}_{\text{merged}}$, where $R < 1$ indicates federation wins.

## Ridgeless Regression: Sweeping Across the Interpolation Threshold

We sweep $p \in \{250, 500, 750, 1050, 1500, 2050, 3000, 4000, 5000, 6000, 7000, 8000\}$ using the ridgeless (minimum-norm least squares) estimator. This range spans three critical transitions: the cluster interpolation threshold at $p = n/k = 1000$, the merged threshold at $p = n = 2000$, and the deep overparameterized regime.

**Figure 1** displays the MSE ratio as a function of $p$. Three regimes are clearly visible:

*Regime I: Both underparameterized ($p < 1000$).* The MSE ratio increases from $1.17$ at $p = 250$ to $2.50$ at $p = 750$, confirming that centralization consistently outperforms federation when both learners are in the classical regime.

*Regime II: Staggered ($1000 < p < 2000$).* At $p = 1050$, just past the cluster interpolation threshold, the MSE ratio spikes to $9.52$: the federated learner's test MSE is nearly an order of magnitude worse than the centralized learner's. Each cluster model ($\gamma_{\text{local}} = 1.05$) is at the interpolation catastrophe while the merged model ($\gamma_{\text{global}} = 0.525$) remains safely underparameterized. By $p = 1500$, the cluster models have moved past the peak ($\gamma_{\text{local}} = 1.5$) and the ratio drops to $0.73$ — federation now outperforms centralization as cluster models enter the regime of benign overfitting while the merged model approaches its own threshold.

*Regime III: Both overparameterized ($p > 2000$).* At $p = 2050$, the merged model is at its interpolation threshold ($\gamma_{\text{global}} = 1.025$) while the cluster models have $\gamma_{\text{local}} = 2.05$. The MSE ratio plummets to $0.062$: the federated learner's MSE is approximately 16 times lower than the centralized learner's. As $p$ increases further, the merged model recovers and the ratio converges toward 1, crossing 1.0 near $p = 5000$. In the deep overparameterized regime ($p = 8000$, $\gamma_{\text{global}} = 4.0$), the ratio is $1.02$ — the two approaches are essentially equivalent.

The raw MSEs of both learners approach the null risk as $p \to \infty$, consistent with Hastie et al. (2022), which predicts $R(\gamma) \to r^2$ as $\gamma \to \infty$ when $\text{SNR} > 1$.

## Ridge Regression: Regularization Eliminates the Federation Advantage

We repeat the identical experiment using ridge regression with leave-one-out cross-validated regularization ($\lambda$ selected from 20 values spanning $[10^{-6}, 10^{6}]$, tuned independently per learner).

**Figure 2** shows the MSE ratio. The contrast with the ridgeless case is stark: the ratio remains between $1.06$ and $1.61$ for all $p$, with centralization winning uniformly. There is no crossover. At $p = 1050$, where the ridgeless ratio spiked to $9.52$, the ridge ratio is only $1.57$. At $p = 2050$, where the ridgeless ratio dropped to $0.062$, the ridge ratio is $1.34$. Ridge regression smooths over both interpolation catastrophes. This is consistent with recent work on federated ridge regression: Alsulaimawi (2026) shows that aggregating sufficient statistics recovers the exact centralized ridge solution in one round, while Fanì et al. (2024) use ridge classifiers on pre-trained features to achieve heterogeneity-immune federated learning. Both implicitly treat the centralized ridge solution as the target — our results confirm that this is justified, as centralized ridge uniformly dominates federated ridge.

**The regime in which federation outperforms centralization exists only because of the interpolation catastrophe, and regularization eliminates this catastrophe.**

## The Role of Heterogeneity

We investigate how heterogeneity affects the MSE ratio across three axes. (i) *Directional heterogeneity*: varying the angle $\theta \in [0°, 90°]$ between cluster means at $p = 3000$ with ridge regression, the MSE ratio is essentially flat ($1.09$–$1.11$), with all confidence intervals overlapping. (ii) *Signal strength*: varying the norm ratio $c \in [1, 10]$, the mean ratio increases from $1.10$ to $1.92$, but the variance grows dramatically (95\% CI at $c = 10$: $[0.21, 3.64]$), driven by heavy-tailed outliers rather than a systematic shift. (iii) *Covariance structure*: with cluster-specific block covariance ($\rho_{\max} \in [0.1, 0.9]$), the ratio ranges from $1.54$ to $1.61$ with no trend; with heterogeneous diagonal covariance (entries $\sim \text{Uniform}(1, 5)$, drawn independently per cluster), the three-regime structure is preserved with minor quantitative changes. Full heterogeneity results are in Appendix [X].

**Across all heterogeneity dimensions, the overparameterization ratio determines the regime, and heterogeneity modulates only the magnitude.**

## Discussion

Our results yield several insights for federated learning theory and practice.

**Model sizing as the primary lever.** The most consequential design choice is model complexity relative to local sample size. Our results identify a simple rule: if $\gamma_{\text{local}} < 1$, centralize; if $\gamma_{\text{global}} \approx 1$, federation provides a substantial advantage; if $\gamma_{\text{global}} \gg 1$, the choice barely matters.

**Why FedAvg works.** A persistent empirical puzzle is that simple FedAvg (McMahan et al., 2017) often performs comparably to heterogeneity-robust algorithms. Our results offer a statistical explanation: if the dominant factor is the overparameterization ratio rather than heterogeneity, algorithms addressing only heterogeneity provide limited marginal benefit.

**Interleaved double descent.** The federated and centralized learners experience their interpolation peaks at different values of $p$ ($p = n/k$ and $p = n$), creating a region where their relative performance inverts. This interleaving is a structural consequence of data splitting and connects to the subagging results of Patil et al. (2023) and Du et al. (2023), who showed that random-split ensembles can monotonize the risk curve. Our finding that heterogeneity has a second-order effect suggests these risk properties extend to structured (non-random) partitions.

**Theory-simulation agreement.** The raw MSE curves closely match the Hastie et al. (2022) risk formula. At SNR = 5, the theory predicts a local minimum at $\gamma^* \approx 1.81$, and our merged model achieves its best overparameterized performance near $\gamma_{\text{global}} = 2.0$ ($p = 4000$), consistent with this prediction.


---


# Related Work

**Federated learning and heterogeneity.** McMahan et al. (2017) introduced FedAvg; Zhao et al. (2018) showed non-IID data can severely degrade its performance. This motivated heterogeneity-robust algorithms — FedProx (Li et al., 2020a), SCAFFOLD (Karimireddy et al., 2020), FedNova (Wang et al., 2020) — and convergence analyses under heterogeneity (Li et al., 2020b). Personalization approaches train cluster-specific or per-client models: Fallah et al. (2020) use meta-learning, and Ghosh et al. (2020) propose IFCA with convergence guarantees. Our focus is not on algorithmic robustness to heterogeneity but on the statistical question of when per-cluster training outperforms pooling — and our answer is that heterogeneity is secondary to the overparameterization ratio.

**Federated ridge regression.** Alsulaimawi (2026) shows that aggregating local sufficient statistics recovers the exact centralized ridge solution in one communication round. Fanì et al. (2024) use ridge classifiers on pre-trained features to achieve heterogeneity-immune federated learning. Both treat the centralized ridge solution as the gold standard. Our results confirm this: with optimally-tuned ridge, centralization uniformly dominates federation, because regularization eliminates the interpolation threshold that gives federation its only statistical advantage.

**High-dimensional regression and double descent.** Belkin et al. (2019) identified the double descent phenomenon; Nakkiran et al. (2020) demonstrated it in deep networks; Hastie et al. (2022) provided exact asymptotic risk formulas for ridgeless least squares; Bartlett et al. (2020) gave conditions for benign overfitting. In the distributed setting, Zhang et al. (2013) and Duchi et al. (2013) characterized the statistical cost of distributing data in the classical regime. Our work extends these analyses to the overparameterized regime, where the relationship between distributed and centralized estimation is qualitatively different due to interpolation threshold staggering.

**Ensembles in overparameterized models.** Patil et al. (2023) characterized the risk of subagged ridgeless predictors, showing that subagging monotonizes the risk curve, and Du et al. (2023) proved that optimal ridgeless ensembles match optimal ridge. Our federated setting is a form of structured subagging where the partition is determined by cluster membership. Rakovic et al. (2023) also study double descent in federated linear regression, confirming that federation can outperform centralization near the interpolation threshold. We extend this by systematically varying heterogeneity across multiple dimensions and comparing ridgeless and ridge estimators to isolate the role of regularization.


---


# Limitations

Our results are for high-dimensional linear models with Gaussian features. In the NTK regime, our findings apply directly to wide neural networks, but feature learning in finite-width networks may make heterogeneity more consequential. We restrict to $k = 2$ equal-sized clusters; increasing $k$ shifts the cluster threshold to $p = n/k$ and could amplify the staggering effect, while unequal sizes create asymmetric thresholds. Our federated learner uses one-shot averaging; iterative FedAvg with multiple local steps may interact with overparameterization differently. We assume a shared $\beta$ across clusters — settings with cluster-specific models (concept drift) would fundamentally change the analysis and are where personalization methods are most naturally motivated. Finally, our ridge comparison assumes optimally-tuned $\lambda$; in practice, tuning regularization in federated settings is itself challenging, and suboptimal regularization may preserve the interpolation threshold effects we identify.


---


## References

- Alsulaimawi, Z. (2026). One-shot federated ridge regression: Exact recovery via sufficient statistic aggregation. *arXiv preprint arXiv:2601.08216*.
- Bartlett, P. L., Long, P. M., Lugosi, G., and Tsigler, A. (2020). Benign overfitting in linear regression. *Proceedings of the National Academy of Sciences*, 117(48):30063–30070.
- Belkin, M., Hsu, D., Ma, S., and Mandal, S. (2019). Reconciling modern machine learning practice and the bias-variance trade-off. *Proceedings of the National Academy of Sciences*, 116(32):15849–15854.
- Du, J.-H., Patil, P., and Kuchibhotla, A. K. (2023). Subsample ridge ensembles: Equivalences and generalized cross-validation. *Proceedings of the 40th International Conference on Machine Learning (ICML)*.
- Duchi, J. C., Jordan, M. I., Wainwright, M. J., and Zhang, Y. (2013). Information-theoretic lower bounds for distributed statistical estimation with communication constraints. *Advances in Neural Information Processing Systems (NeurIPS)*, 26.
- Fallah, A., Mokhtari, A., and Ozdaglar, A. (2020). Personalized federated learning with theoretical guarantees: A model-agnostic meta-learning approach. *Advances in Neural Information Processing Systems (NeurIPS)*, 33:3557–3568.
- Fanì, E., Camoriano, R., Caputo, B., and Ciccone, M. (2024). Accelerating heterogeneous federated learning with closed-form classifiers. *Proceedings of the 41st International Conference on Machine Learning (ICML)*.
- Ghosh, A., Chung, J., Yin, D., and Ramchandran, K. (2020). An efficient framework for clustered federated learning. *Advances in Neural Information Processing Systems (NeurIPS)*, 33.
- Hastie, T., Montanari, A., Rosset, S., and Tibshirani, R. J. (2022). Surprises in high-dimensional ridgeless least squares interpolation. *Annals of Statistics*, 50(2):949–986.
- Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S. J., Stich, S. U., and Suresh, A. T. (2020). SCAFFOLD: Stochastic controlled averaging for federated learning. *Proceedings of the 37th International Conference on Machine Learning (ICML)*.
- Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., and Smith, V. (2020a). Federated optimization in heterogeneous networks. *Proceedings of Machine Learning and Systems (MLSys)*.
- Li, X., Huang, K., Yang, W., Wang, S., and Zhang, Z. (2020b). On the convergence of FedAvg on non-IID data. *International Conference on Learning Representations (ICLR)*.
- McMahan, B., Moore, E., Ramage, D., Hampson, S., and Aguera y Arcas, B. (2017). Communication-efficient learning of deep networks from decentralized data. *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)*.
- Nakkiran, P., Kaplun, G., Bansal, Y., Yang, T., Barak, B., and Sutskever, I. (2020). Deep double descent: Where bigger models and more data can hurt. *International Conference on Learning Representations (ICLR)*.
- Patil, P., Du, J.-H., and Kuchibhotla, A. K. (2023). Bagging in overparameterized learning: Risk characterization and risk monotonization. *Journal of Machine Learning Research*, 24(319):1–113.
- Rakovic, V., Pejoski, S., Hadzi-Velkov, Z., and Denkovski, D. (2023). On the double descent effect in federated learning. *TechRxiv preprint*.
- Wang, J., Liu, Q., Liang, H., Joshi, G., and Poor, H. V. (2020). Tackling the objective inconsistency problem in heterogeneous federated optimization. *Advances in Neural Information Processing Systems (NeurIPS)*, 33.
- Zhang, Y., Duchi, J. C., and Wainwright, M. J. (2013). Communication-efficient algorithms for statistical optimization. *Journal of Machine Learning Research*, 14:3321–3363.
- Zhao, Y., Li, M., Lai, L., Suda, N., Civin, D., and Chandra, V. (2018). Federated learning with non-IID data. *arXiv preprint arXiv:1806.00582*.
