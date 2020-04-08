# Lab 3 Report

## 1

### 1.a)

$$
\text{let } o_k=(\hat x_k,\hat y_k), O={o_0,\dots,o_{N-1}}
\\\text{Forward pass:}
\\\alpha_i(z_i)=p(o_i|z_i)\sum_{z_{i-1}}\alpha_{i-1}(z_{i-1})W(z_{i-1}|z_i)
\\\text{Backward pass:}
\\\beta_k(z_k)=\sum_{z_{k+1}}\beta_{k+1}(z_{k+1})p(o_{k+1}|z_{z+1})W(z_k|z_{k+1})
\\p(z_i|O)=\frac{1}{Z}\alpha_i(z_i)\beta_i(z_i)
$$

FB algorithm

Initialize forward and backward messages

recursion relations of the messages

Computation of the marginal distribution based on the messages

