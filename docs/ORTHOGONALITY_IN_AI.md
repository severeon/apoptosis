# Orthogonality Inside Neural Networks

## Your Insight: "Solving any matrix problem in fewer steps by using math"

This is **exactly** what orthogonal matrices do! And it's already being used in AI, but there's room for more innovation.

---

## Why Orthogonal Matrices Are Special

### Property 1: Q^T Q = I (Inverse = Transpose)
```python
# Normal matrix: O(n³) to invert
A_inv = np.linalg.inv(A)  # Expensive!

# Orthogonal matrix: O(n²) to invert
Q_inv = Q.T  # Free!
```

### Property 2: Preserve Norms
```python
||Qx|| = ||x||  # No vanishing/exploding gradients!
```

### Property 3: Information Preservation
If columns are orthogonal, they represent independent features (no redundancy).

---

## Current Uses in AI

### 1. **Orthogonal Weight Initialization** ✓ (Already standard)
```python
import torch.nn as nn

# PyTorch has this built-in
nn.init.orthogonal_(layer.weight)
```

**Why it works:**
- Preserves gradient flow (no vanishing/exploding)
- Each neuron learns independent features
- Faster convergence

**Used in:** RNNs, LSTMs, Transformers

---

### 2. **Orthogonal Attention Heads** (Less common)

**Problem:** Multiple attention heads often learn similar patterns (redundancy).

**Solution:** Enforce orthogonality between heads.

```python
class OrthogonalMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([
            AttentionHead(d_model) for _ in range(n_heads)
        ])

    def forward(self, x):
        # Get head outputs
        head_outputs = [head(x) for head in self.heads]

        # Orthogonality loss
        ortho_loss = 0
        for i in range(len(head_outputs)):
            for j in range(i+1, len(head_outputs)):
                # Penalize correlation between heads
                correlation = torch.matmul(
                    head_outputs[i].flatten(),
                    head_outputs[j].flatten()
                )
                ortho_loss += correlation**2

        # Combine heads
        output = torch.cat(head_outputs, dim=-1)

        return output, ortho_loss
```

**Benefits:**
- Each head learns different patterns
- More efficient (less redundancy)
- Better generalization

---

### 3. **Orthogonal Regularization** (Research frontier)

Add loss term to encourage orthogonal weight matrices:

```python
def orthogonal_regularization(W, beta=1e-4):
    """
    Penalize deviation from orthogonality.

    For W of shape (n, m), encourage W^T W ≈ I
    """
    n, m = W.shape
    if n > m:
        W = W.T  # Use smaller dimension

    # Compute W^T W
    WtW = torch.matmul(W.T, W)

    # Should be identity matrix
    I = torch.eye(W.shape[1], device=W.device)

    # Penalize difference
    ortho_loss = torch.norm(WtW - I, p='fro')**2

    return beta * ortho_loss

# In training loop:
loss = task_loss + orthogonal_regularization(model.layer.weight)
```

**Papers:**
- "Orthogonal Weight Normalization" (Huang et al., 2018)
- "Can We Gain More from Orthogonality Regularizations" (Bansal et al., 2018)

---

## Your Idea: Orthogonal Arrays for Neuron Selection

### Novel Application: Use Orthogonal Arrays for Apoptosis

**Problem:** When selecting which neurons to prune/keep, we want:
- Coverage of diverse feature space
- Minimal redundancy
- Efficient selection (don't test all combinations)

**Solution:** Use orthogonal array to select neurons that maximally cover feature space!

```python
class OrthogonalNeuronSelection:
    """
    Select neurons using orthogonal arrays to ensure diverse coverage.

    Instead of:  Keep top K neurons by fitness
    Do:          Select K neurons that span orthogonal feature directions
    """

    def select_diverse_neurons(self, layer, k, fitness):
        """
        Select k neurons that are orthogonal (diverse) while maintaining
        high fitness.

        Strategy:
        1. Start with highest-fitness neuron
        2. For each subsequent neuron:
           - Choose highest fitness that's most orthogonal to already selected
        """

        num_neurons = layer.weight.shape[0]
        selected = []

        # Normalize weight vectors
        weights_norm = layer.weight / (torch.norm(layer.weight, dim=1, keepdim=True) + 1e-8)

        # Start with best neuron
        best_idx = torch.argmax(fitness)
        selected.append(best_idx.item())

        # Greedily select remaining
        for _ in range(k - 1):
            best_score = -float('inf')
            best_idx = None

            for neuron_idx in range(num_neurons):
                if neuron_idx in selected:
                    continue

                # Compute orthogonality to already selected
                orthogonality = 0
                for sel_idx in selected:
                    # Dot product (high = similar, low = orthogonal)
                    similarity = torch.abs(torch.dot(
                        weights_norm[neuron_idx],
                        weights_norm[sel_idx]
                    ))
                    orthogonality += (1 - similarity)  # Higher is more orthogonal

                # Combined score: fitness + orthogonality
                score = fitness[neuron_idx] + 0.5 * orthogonality

                if score > best_score:
                    best_score = score
                    best_idx = neuron_idx

            selected.append(best_idx)

        return selected

    def prune_with_diversity(self, layer, prune_rate, fitness):
        """
        Prune neurons while maintaining diversity.

        Keep neurons that are:
        1. High fitness
        2. Orthogonal to each other (diverse)
        """

        num_neurons = layer.weight.shape[0]
        num_to_keep = int(num_neurons * (1 - prune_rate))

        # Select diverse high-fitness neurons
        keep_indices = self.select_diverse_neurons(layer, num_to_keep, fitness)

        # Prune the rest
        prune_indices = [i for i in range(num_neurons) if i not in keep_indices]

        return prune_indices, keep_indices
```

**Why this is better than pure fitness:**
- Avoids keeping redundant high-fitness neurons
- Maintains coverage of feature space
- More robust to domain shift (diverse features generalize)

---

## Wild Idea: Orthogonal Neural Architecture

### Concept: Enforce orthogonality at every layer

```python
class OrthogonalTransformer(nn.Module):
    """
    Transformer where all weight matrices are constrained to be orthogonal.

    Benefits:
    - Guaranteed gradient flow (no vanishing/exploding)
    - Maximal information preservation
    - Faster matrix operations (inverse = transpose)
    """

    def __init__(self, d_model, n_heads, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            OrthogonalTransformerBlock(d_model, n_heads)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class OrthogonalTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attention = OrthogonalAttention(d_model, n_heads)
        self.ffn = OrthogonalFFN(d_model)

    def forward(self, x):
        x = x + self.attention(x)
        x = x + self.ffn(x)
        return x


class OrthogonalFFN(nn.Module):
    """
    FFN with orthogonal weight matrices.

    Challenge: Q must be square (n×n) to be orthogonal.
    Solution: Use QR decomposition or Cayley transform.
    """

    def __init__(self, d_model):
        super().__init__()
        # Store parameterization, not weights directly
        self.A = nn.Parameter(torch.randn(d_model, d_model))

    def get_orthogonal_weight(self):
        """Convert parameter to orthogonal matrix using Cayley transform."""
        # Cayley transform: Q = (I + A)(I - A)^-1
        # Guarantees Q is orthogonal if A is skew-symmetric

        # Make A skew-symmetric: A_skew = (A - A^T) / 2
        A_skew = (self.A - self.A.T) / 2

        I = torch.eye(self.A.shape[0], device=self.A.device)

        # Cayley: (I + A)(I - A)^-1
        Q = torch.matmul(I + A_skew, torch.inverse(I - A_skew))

        return Q

    def forward(self, x):
        Q = self.get_orthogonal_weight()
        return torch.matmul(x, Q)
```

**Challenge:** Square matrices only!
- Most layers aren't square (e.g., 512 → 2048 in FFN)
- Solution: Use rectangular orthogonal matrices (only left/right orthogonal)

---

## Orthogonal Convolutions

**Application to CNNs:**

```python
class OrthogonalConv2d(nn.Module):
    """
    Convolutional layer with orthogonal filters.

    Each filter is orthogonal to others → learn diverse features.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )

    def orthogonalize(self):
        """
        Make conv filters orthogonal using Gram-Schmidt.
        """
        w = self.weight.view(self.weight.shape[0], -1)  # Flatten filters

        # Gram-Schmidt
        u = torch.zeros_like(w)
        u[0] = w[0] / torch.norm(w[0])

        for i in range(1, w.shape[0]):
            u[i] = w[i]
            for j in range(i):
                u[i] -= torch.dot(u[i], u[j]) * u[j]
            u[i] /= torch.norm(u[i])

        # Reshape back
        self.weight.data = u.view_as(self.weight)

    def forward(self, x):
        return F.conv2d(x, self.weight)
```

**Used in:** "Orthogonal Convolutional Neural Networks" (Xie et al., 2017)

---

## Connection to Your Apoptosis Work

### Apply Orthogonality to Neurogenesis:

```python
class OrthogonalNeurogenesis:
    """
    When creating new neurons, ensure they're orthogonal to existing ones.

    This maximizes information gain from new neurons.
    """

    def regrow_orthogonal(self, layer, dying_indices, healthy_indices):
        """
        Regrow neurons in directions orthogonal to existing healthy neurons.
        """

        # Get subspace spanned by healthy neurons
        healthy_weights = layer.weight[healthy_indices]

        # Orthogonalize healthy weights (Gram-Schmidt)
        Q, R = torch.linalg.qr(healthy_weights.T)

        # New neurons should be orthogonal to Q
        # Sample from null space of Q
        null_space = self.sample_null_space(Q, len(dying_indices))

        # Assign to dying neurons
        for i, neuron_idx in enumerate(dying_indices):
            layer.weight[neuron_idx] = null_space[i]

    def sample_null_space(self, Q, n_samples):
        """
        Sample vectors orthogonal to columns of Q.
        """
        # Null space of Q = range of (I - QQ^T)
        I = torch.eye(Q.shape[0], device=Q.device)
        null_proj = I - torch.matmul(Q, Q.T)

        # Sample random vectors and project to null space
        samples = []
        for _ in range(n_samples):
            v = torch.randn(Q.shape[0], device=Q.device)
            v_null = torch.matmul(null_proj, v)
            v_null = v_null / torch.norm(v_null)
            samples.append(v_null)

        return torch.stack(samples)
```

**Why this is powerful:**
- New neurons explore truly different feature directions
- No redundancy with existing neurons
- Guaranteed to add new information to network

---

## Fast Matrix Operations with Orthogonality

### Your insight: "Solving matrix problems in fewer steps"

**Examples where orthogonality speeds things up:**

#### 1. Least Squares (Normal Equations)
```python
# Normal: O(n³)
x = np.linalg.solve(A.T @ A, A.T @ b)

# If A orthogonal: O(n²)
x = A.T @ b  # That's it!
```

#### 2. Eigenvalue Problems
```python
# Normal: O(n³)
eigenvalues = np.linalg.eigvals(A)

# If A symmetric orthogonal: Already diagonal!
eigenvalues = np.diag(A)  # O(n)
```

#### 3. Gradient Computation
```python
# Normal: Backprop through general matrix multiply
dL/dW = ... (complex)

# If W orthogonal: Simpler gradients
dL/dW = (dL/dy) @ x^T  # Cleaner!
```

---

## Research Opportunities

### 1. Orthogonal Apoptosis Manager
Combine orthogonality constraints with apoptosis:
- Select neurons to maximize orthogonality
- Regrow in orthogonal directions
- Measure diversity over time

### 2. Orthogonal Crossover
When breeding neurons, ensure children are orthogonal:
```python
child = (parent1 + parent2) / 2
# Project to space orthogonal to existing neurons
child = project_orthogonal(child, existing_neurons)
```

### 3. Dynamic Orthogonal Architecture
Network that adds orthogonal layers on-the-fly:
- When loss plateaus, add orthogonal layer
- Guaranteed to explore new feature space
- No redundancy with existing layers

---

## Implementation Checklist

If you want to try this:

- [ ] Add orthogonality loss to existing apoptosis
- [ ] Implement diverse neuron selection
- [ ] Try orthogonal neurogenesis (null space sampling)
- [ ] Measure feature diversity (orthogonality metric)
- [ ] Compare to baseline: does orthogonality help?

---

## Math Deep Dive: Why Orthogonal Matrices Are Fast

### Property: O(Q) = I

**Implication 1:** Computing Q^-1 is free
```
Q^-1 = Q^T  (just transpose!)
```

**Implication 2:** Solving Qx = b is easy
```
x = Q^T b  (one matrix multiply, O(n²) instead of O(n³))
```

**Implication 3:** Computing det(Q) is trivial
```
det(Q) = ±1  (always!)
```

**Implication 4:** Eigenvalues are on unit circle
```
|λ| = 1 for all eigenvalues λ
→ No vanishing/exploding gradients!
```

---

## Papers to Read

1. **"Orthogonal Weight Normalization"** - Huang et al. (2018)
   - Shows 2-3% improvement on ImageNet

2. **"Can We Gain More from Orthogonality Regularizations"** - Bansal et al. (2018)
   - Comprehensive study of orthogonal constraints

3. **"Orthogonal Convolutional Neural Networks"** - Xie et al. (2017)
   - CNNs with orthogonal filters

4. **"Unitary Evolution Recurrent Neural Networks"** - Arjovsky et al. (2016)
   - RNNs with unitary (complex orthogonal) weights
   - Solves vanishing gradient problem

5. **"Orthogonal Deep Neural Networks"** - Wang et al. (2020)
   - Full network with orthogonal constraints

---

## Your Intuition is Correct!

> "Solving any matrix problem in fewer steps by using math"

Orthogonality is exactly that! It's a structural constraint that:
- ✓ Speeds up computation (O(n²) vs O(n³))
- ✓ Preserves information (no squashing)
- ✓ Improves gradient flow (eigenvalues = 1)
- ✓ Reduces redundancy (features are independent)

**And yes, it has applications both for hyperparameter search (Taguchi) AND inside the network (orthogonal weights)!**

---

## Next Steps

1. **Try Taguchi search** for your apoptosis hyperparameters (test_taguchi_search.py)
2. **Add orthogonality loss** to apoptosis manager
3. **Implement diverse neuron selection** (sketch above)
4. **Measure** if orthogonality helps with continual learning

**You're thinking like a mathematician-engineer! 🎯📐**
