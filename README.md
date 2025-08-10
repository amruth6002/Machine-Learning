# 📊 ML / DL / AI Math Prep Guide
> A step-by-step, resource-linked roadmap to master the math needed for Machine Learning, Deep Learning, and AI interviews & implementation — without overstudying.

---

## 📍 Legend
- 🟢 **Core** – Must learn for ML/DL
- 🟡 **Useful** – Good to know, sometimes asked
- 🔵 **Applied** – Needed for implementation
- 🔴 **Optional** – Only for research-level AI

---

## **Phase 1 – Probability & Statistics for ML (Weeks 1–3)**

| Subtopic | Priority | Resource | Link | Coding Practice |
|----------|----------|----------|------|-----------------|
| Probability rules & conditional probability | 🟢 | StatQuest + Stat 110 Ch 1–2 | [StatQuest Video](https://youtu.be/UZGgVxRt0f4) · [Stat 110 Notes](https://projects.iq.harvard.edu/stat110/home) | Simulate coin tosses & dice |
| Bayes theorem & Naive Bayes | 🟢 | StatQuest | [Bayes Theorem](https://youtu.be/HZGCoVF3YvM) | Build Naive Bayes text classifier |
| Expectation, variance, covariance | 🟢 | Khan Academy | [Expected Value](https://www.khanacademy.org/math/statistics-probability/probability-library) | Compute E[X], Var(X) from dataset |
| Distributions (Normal, Bernoulli, Multinomial, Uniform) | 🟢 | Khan Academy | [Distributions Module](https://www.khanacademy.org/math/statistics-probability/modeling-distributions-of-data) | Fit distribution to data |
| CLT & Law of Large Numbers | 🟡 | Think Stats (Ch 5) | [Free PDF](https://greenteapress.com/wp/think-stats-2e/) | Simulate CLT with sample means |
| Hypothesis testing (t-test, chi-sq) | 🟡 | Khan Academy | [Hypothesis Testing](https://www.khanacademy.org/math/statistics-probability/significance-tests) | Compare 2 datasets |

---

## **Phase 2 – Linear Algebra for ML/DL (Weeks 4–6)**

| Subtopic | Priority | Resource | Link | Coding Practice |
|----------|----------|----------|------|-----------------|
| Vectors, matrices, transpose | 🟢 | 3Blue1Brown Essence of LA #1–4 | [YouTube Playlist](https://www.3blue1brown.com/topics/linear-algebra) | Implement matrix multiplication |
| Dot product & projections | 🟢 | Khan Academy | [Vectors Module](https://www.khanacademy.org/math/linear-algebra) | Cosine similarity between docs |
| Matrix multiplication & properties | 🟢 | 3Blue1Brown #5–7 | [YouTube](https://youtu.be/XkY2DOUCWMU) | Implement linear layer forward pass |
| Determinant & inverse | 🟡 | Khan Academy | [Matrix Properties](https://www.khanacademy.org/math/linear-algebra/matrix-transformations) | Solve Ax = b |
| Eigenvalues/vectors & SVD | 🟢 | 3Blue1Brown #9–14 | [YouTube](https://youtu.be/PFDu9oVAE-g) | Implement PCA from scratch |
| Orthogonality & projections | 🟢 | Khan Academy | [Orthogonal Module](https://www.khanacademy.org/math/linear-algebra/alternate-bases) | Project vector onto subspace |

---

## **Phase 3 – Calculus & Optimization for ML/DL (Weeks 7–9)**

| Subtopic | Priority | Resource | Link | Coding Practice |
|----------|----------|----------|------|-----------------|
| Derivatives & partial derivatives | 🟢 | Khan Academy | [Derivatives Module](https://www.khanacademy.org/math/calculus-1) | Differentiate cost functions |
| Chain rule & multivariable derivatives | 🟢 | Khan Academy | [Multivariable Calculus](https://www.khanacademy.org/math/multivariable-calculus) | Backprop in 2-layer NN |
| Gradient vectors | 🟢 | Khan Academy | [Gradient Module](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives) | Implement gradient descent |
| Hessian matrix & curvature | 🟡 | Khan Academy | [Multivariable Section](https://www.khanacademy.org/math/multivariable-calculus) | Visualize loss surface |
| Optimization (SGD, Momentum, Adam) | 🔵 | Andrew Ng ML Course | [Coursera Link](https://www.coursera.org/learn/machine-learning) | Train logistic regression with Adam |

---

## **Phase 4 – ML/DL-Specific Math (Weeks 10–12)**

| Subtopic | Priority | Resource | Link | Coding Practice |
|----------|----------|----------|------|-----------------|
| Linear regression (OLS, L1/L2) | 🟢 | StatQuest | [Regression Videos](https://youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF) | Implement from scratch |
| Logistic regression & sigmoid | 🟢 | StatQuest | [Logistic Regression](https://youtu.be/yIYKR4sgzI8) | Implement binary classifier |
| Softmax & cross-entropy | 🟢 | PRML (Bishop) Sec 4.3 | [Book PDF](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/) | Implement softmax classifier |
| Bias-variance tradeoff | 🟢 | StatQuest | [Bias-Variance](https://youtu.be/EuBBz3bI-aA) | Train under/overfit models |
| Activation functions | 🟢 | Goodfellow DL Book Ch 6 | [Free PDF](https://www.deeplearningbook.org/) | Compare activations in NN |
| Loss functions (MSE, MAE, CE) | 🟢 | Goodfellow DL Book Ch 6 | [Free PDF](https://www.deeplearningbook.org/) | Code custom loss |
| Regularization (L1, L2, dropout) | 🟢 | Andrew Ng ML | [Coursera Link](https://www.coursera.org/learn/machine-learning) | Apply in NN |
| Information theory basics (entropy, KL divergence) | 🟡 | Victor Lavrenko YouTube | [Playlist](https://youtube.com/playlist?list=PLBv09BD7ez_6k3cPO2pF0u2TIGU9k_F8k) | Compute KL divergence |

---

## 🔄 Review Plan
- **Weekly:** Redo 1 coding exercise from each phase  
- **Biweekly:** Solve 5 ML math interview questions (*Deep Learning Interviews* book)  
- **Monthly:** Implement a model from scratch without frameworks  

---

## 📚 Extra References
- [CS229 Stanford ML Notes](https://cs229.stanford.edu/)  
- [Dive Into Deep Learning](https://d2l.ai/)  
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)  

---
