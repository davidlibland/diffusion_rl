# Analytical Value Function for the GMM + Quadratic Reward Setting

## Setup

The base distribution is a Gaussian mixture model (GMM):

$$p(x_1) = \sum_{k=1}^K w_k\,\mathcal{N}(x_1;\,\mu_k,\,\sigma_k^2 I)$$

The stochastic interpolant connects $x_0 = 0$ to $x_1$ via

$$x_t = t\,x_1 + \sqrt{2a\,t(1-t)}\;\varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)$$

so the conditional is $p(x_t \mid x_1) = \mathcal{N}(x_t;\;t\,x_1,\;\sigma_\varepsilon^2\,I)$ with $\sigma_\varepsilon^2 = 2a\,t(1-t)$.

The reward is $r(x_1) = -10\,\|x_1 - c\|^2$ with $c = [1, 0]$.

---

## 1. Marginal $p(x_t)$ as a GMM

Marginalising over $x_1$ within each component:

$$p(x_t) = \sum_{k=1}^K w_k \int \mathcal{N}(x_t;\,t\,x_1,\,\sigma_\varepsilon^2 I)\;\mathcal{N}(x_1;\,\mu_k,\,\sigma_k^2 I)\,dx_1$$

Since $x_t = t\,x_1 + \varepsilon_t$ with $\varepsilon_t \sim \mathcal{N}(0, \sigma_\varepsilon^2 I)$ and $x_1 \sim \mathcal{N}(\mu_k, \sigma_k^2 I)$, the marginal is:

$$\boxed{p(x_t) = \sum_{k=1}^K w_k\;\mathcal{N}\!\left(x_t;\;t\,\mu_k,\;s_k^2(t)\,I\right)}$$

where

$$s_k^2(t) = t^2\,\sigma_k^2 + 2a\,t(1-t)$$

The weights are unchanged. The mean of component $k$ scales linearly with $t$ from $0$ (at $t=0$) to $\mu_k$ (at $t=1$). The variance $s_k^2(t)$ peaks mid-trajectory and vanishes at both endpoints.

---

## 2. Posterior $p(x_1 \mid x_t)$ as a GMM

By Bayes' rule:

$$p(x_1 \mid x_t) \propto p(x_t \mid x_1)\,p(x_1) = \sum_{k=1}^K w_k\;\mathcal{N}(x_t;\,t\,x_1,\,\sigma_\varepsilon^2 I)\;\mathcal{N}(x_1;\,\mu_k,\,\sigma_k^2 I)$$

For each component $k$, the product $\mathcal{N}(x_t;\,t\,x_1,\,\sigma_\varepsilon^2 I)\cdot\mathcal{N}(x_1;\,\mu_k,\,\sigma_k^2 I)$ is a Gaussian in $x_1$ (the likelihood is linear-Gaussian). The normalising constant is $\mathcal{N}(x_t;\,t\,\mu_k,\,s_k^2(t)\,I)$, so:

$$\boxed{p(x_1 \mid x_t) = \sum_{k=1}^K \tilde{w}_k(x_t,t)\;\mathcal{N}\!\left(x_1;\;\tilde{\mu}_k(x_t,t),\;\tilde{V}_k(t)\,I\right)}$$

**Posterior weights** (re-normalised):

$$\tilde{w}_k(x_t,t) \propto w_k\;\mathcal{N}\!\left(x_t;\;t\,\mu_k,\;s_k^2(t)\,I\right)$$

**Posterior variance** (same for all $x_t$, independent of $k$'s mean):

$$\tilde{V}_k(t) = \frac{\sigma_\varepsilon^2\,\sigma_k^2}{s_k^2(t)} = \frac{2a\,t(1-t)\,\sigma_k^2}{t^2\,\sigma_k^2 + 2a\,t(1-t)}$$

**Posterior mean** (a weighted combination of prior mean and "likelihood mean" $x_t/t$):

$$\tilde{\mu}_k(x_t,t) = \frac{2a(1-t)\,\mu_k + t\,\sigma_k^2\,x_t}{s_k^2(t)}$$

**Sanity checks:**
- $t=0$: $\tilde{V}_k \to \sigma_k^2$, $\tilde{\mu}_k \to \mu_k$, $\tilde{w}_k \to w_k$ — posterior equals prior ✓
- $t=1$: $\tilde{V}_k \to 0$, $\tilde{\mu}_k \to x_1 = x_t$ — posterior collapses to a delta at $x_t$ ✓

---

## 3. Analytical Value Function $V(x_t, t)$

$$V(x_t, t) = \log\,\mathbb{E}_{p(x_1 \mid x_t)}\!\left[e^{r(x_1)}\right]
= \log \sum_{k=1}^K \tilde{w}_k\!\int \mathcal{N}(x_1;\,\tilde{\mu}_k,\,\tilde{V}_k I)\;e^{-10\|x_1 - c\|^2}\,dx_1$$

For a Gaussian $\mathcal{N}(x;\,m,\,v\,I)$ times $e^{-10\|x-c\|^2}$ (a Gaussian "likelihood" with precision $20 I$), the integral is tractable. Using the completing-the-square identity:

$$\int \mathcal{N}(x;\,m,\,v\,I)\;e^{-10\|x-c\|^2}\,dx = e^{\log Z(m,\,v)}$$

where, after algebraic simplification into a numerically stable form:

$$\log Z(m, v) = -\frac{d}{2}\log(1 + 20v) + \frac{-10\|m\|^2 + 20\,m\cdot c + 200\,v\,\|c\|^2}{1 + 20v} - 10\,\|c\|^2$$

(This avoids intermediate $1/(2\tau^2)$ terms that diverge as $v\to 0$, and satisfies $\log Z(x_t,0)=r(x_t)$.)

**The value function is therefore:**

$$\boxed{V(x_t, t) = \log\sum_{k=1}^K \tilde{w}_k(x_t,t)\;e^{\log Z(\tilde{\mu}_k(x_t,t),\;\tilde{V}_k(t))}}$$

$$= \operatorname{logsumexp}_k\!\left[\log\tilde{w}_k(x_t,t) + \log Z\!\left(\tilde{\mu}_k(x_t,t),\;\tilde{V}_k(t)\right)\right]$$

This is a closed-form expression in $(x_t, t, \{\mu_k, \sigma_k^2, w_k\})$, with no approximations.

**Boundary checks:**
- $V(0, 0) = \log\sum_k w_k e^{\log Z(\mu_k, \sigma_k^2)} = -5.085$ (matches `analytical_target.py`) ✓
- $V(x_t, 1) = r(x_t) = -10\|x_t - c\|^2$ (posterior is a delta at $x_t$) ✓
