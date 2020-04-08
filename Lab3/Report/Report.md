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

## 3

### 3.a)

$$
\DeclareMathOperator{argmax}{\text{argmax}}
\text{Initialization}
\\w_1(z_1)=\log(p(z_1)p(o_1|z_1)
\\\text{Recursion}
\\w_i(z_i)=\log(p(o_i|z_i)) + \max_{z_{i-1}}\Big\{ \log(p(z_i|z_{i-1})) + w_{i-1}(z_{i-1}) \Big\}
\\\phi_i(z_i)=\argmax_{z_{i-1}}\Big[ \log(p(z_{i-1}|z_i))+w_{i-1}(z_{i-1}) \Big]
$$

