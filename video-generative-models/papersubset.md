**3. Prompt Relay**
Given a sequence of temporally-constrained text prompts $\{(p_s, [t_s^{start}, t_s^{end}])\}_{s=1}^N$, our goal is to generate a video such that each arbitrary prompt $p_s$ is realized within its designated temporal interval $[t_s^{start}, t_s^{end}]$. The generated video should preserve global coherence while ensuring that each prompt influences only its assigned temporal region.

**3.1. Preliminaries**
Cross-attention is a mechanism that enables a diffusion model to incorporate external conditioning information, such as text prompts, into the generation process. Given a latent representation at diffusion step $t$, denoted as $\phi(z_t)$, and a set of conditioning embeddings $\psi(P)$ derived from an input prompt $P$, cross-attention computes interactions between the two through learned projections.

$$ \text{Attn}(\phi(z_t), \psi(P)) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V, \quad (1) $$

where $Q = \ell_Q\phi(z_t)$ are query vectors derived from latent features, $K = \ell_K\psi(P)$ and $V = \ell_V\psi(P)$ are key and value vectors projected from the conditioning embeddings, and $d$ denotes the projection dimensionality. Each attention weight reflects how strongly a latent query attends to a particular conditioning token. Through this operation, semantic information from the conditioning input is selectively injected into the latent representation, allowing different queries to respond to different aspects of the prompt. However, because attention is computed globally over all conditioning tokens, multiple semantic concepts may compete for influence over the same latent queries. When these concepts correspond to different temporal regions, unrestricted attention can lead to interference between instructions.

**3.2. Temporal Prompt Routing**
In order to enforce the association between each prompt $p_s$ and its assigned temporal interval $[t_s^{start}, t_s^{end}]$, we introduce a penalty term $C(Q, K)$ into the cross-attention logits:

$$ \text{Attn}(\phi(z_t), \psi(P)) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}} - C(Q,K)\right)V. \quad (2) $$

The role of $C(Q, K)$ is to suppress the attention between key and query tokens whenever they do not belong to the same interval $[t_s^{start}, t_s^{end}]$. This allows each prompt to guide generation only within its intended segment, without leaking semantic concepts into other parts of the video. For any arbitrary query token indexed by $i$ and any key token $j$ belonging to $p_s$, the penalty is defined as:

$$ C(i,j) = \frac{\text{ReLU}(|f(i) - m_s| - w)^2}{2\sigma^2}, $$
$$ m_s = \frac{t_s^{start} + t_s^{end}}{2}. \quad (3) $$

Here, $f(i)$ denotes the latent frame index associated with query token $i$, and $m_s$ denotes the midpoint of the corresponding temporal segment. The parameter $w$ defines a local window around the segment midpoint within which no penalty is applied, while $\sigma$ controls the rate at which attention decays outside this window. Query tokens within the window incur zero penalty and can attend freely to their associated prompt tokens. Beyond this region, attention is smoothly attenuated as a function of the temporal distance between the query and the segment midpoint. We demonstrate in Fig. 3, that $w = L - 2$ achieves the best balance between temporal isolation and intra-segment fidelity.

We compare our approach to hard masking in Fig. 4. Hard masking sets $C(i,j) = -\infty$ for all query-key pairs where $f(i) \notin [t_s^{start}, t_s^{end}]$ and $j$ belongs to prompt $p_s$ (i.e. a query either attends fully to a prompt or is completely blocked from it). This enforces a sudden switch between prompts at segment boundaries. While hard masking eliminates cross-segment semantic interference, it creates a discontinuity at the boundary: cross-attention switches abruptly to the new prompt while self-attention remains anchored to the previous segment's visual structure, forcing the model to reconcile conflicting signals. Boundary-attention decay avoids this conflict by smoothly co-activating both neighboring prompts near the boundary, giving the model a gradual handoff region in which the transition can be planned jointly before being committed to in the visual representation.

**3.3. Boundary-Attention Decay**
To suppress semantic interference across temporal segments, attention between queries near segment boundaries and prompt tokens from neighboring segments should be negligible. We therefore choose the decay parameter $\sigma$ so that the attention prior sufficiently decreases near segment endpoints. Since our penalty subtracts $C(i, j)$ from the logits, it applies a multiplicative factor $\exp(-C(i, j))$ to the unnormalized attention scores before softmax. This prior is 1 inside the "free-attention" window and decays toward the segment boundaries. Let the endpoint distance from the segment midpoint be $L = |f(i) - m_s|$. We choose $\sigma$ such that the prior reaches a small value $\epsilon$ at the endpoints:

$$ \exp\left(-\frac{(L-w)^2}{2\sigma^2}\right) = \epsilon \Rightarrow \sigma = \frac{L-w}{\sqrt{2\ln(1/\epsilon)}}. \quad (4) $$

This formulation ensures smooth transitions between neighboring prompts while preventing destructive interference across segments. As a result, each textual instruction primarily influences its intended temporal region, allowing the model to focus on one semantic concept at a time while maintaining global temporal coherence.


---

**Figure 3. Ablation Study of the Temporal Penalty Function.**
The curves show the attention fraction retained between a query token and the prompt tokens of a given segment, as a function of the query’s latent frame offset from that segment’s midpoint $m_s$, after applying the penalty $\exp(-C(i, j))$. (Top) Effect of the window parameter $w$. $w = L - 2$ preserves full attention within the segment and only suppresses attention near the segment boundaries. (Bottom) Effect of the decay threshold $\epsilon$. Smaller values enforce stronger attenuation outside the ‘free-attention’ window; however, we find that the choice among small values has negligible perceptual impact. We adopt $\epsilon = 0.1$ as our default.

**Figure 4. Hard Masking vs Boundary-Attention Decay.** 
Hard masking enforces an abrupt semantic switch in cross-attention at segment boundaries while self-attention remains continuous across the segments. This creates a discontinuity at the boundary, forcing the model to reconcile conflicting signals (Woman eats the pasta instead of the man). Boundary-attention decay avoids this conflict by smoothly co-activating both neighboring prompts near the boundary, giving the model a gradual handoff region in which the transition can be planned jointly before being committed to in the visual representation.
