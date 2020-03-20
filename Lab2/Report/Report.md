# ECE368 Lab2 Report

Michael Vu - 1002473272

## 1

$$
\begin{align}
P(a)&=\frac{1}{2\pi\beta}\exp(-(\frac{a_0^2+a_1^2}{2\beta}))=\frac{1}{2\pi\beta}\exp(-\frac{||a||^2}{2\beta})\\

P(x_i,z_i|a)&=P(z_i|x_i,a)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp(-\frac{(z_i-a_1x_i-a_0)^2}{2\sigma^2})\\

P(x_i,z_i)&=P(z_i|x_i)=\int P(z_i|x_i,a)P(a)da=\Gamma_i\\

P(a|x,z)&=\frac{\frac{1}{2\pi\beta}\exp(-\frac{||a||^2}{2\beta})(\frac1{\sqrt{2\pi\sigma^2}})^N\exp(-\frac{1}{2\sigma^2}\sum_{i=1}^N(z_i-a_1x_i-a_0)^2)}{\prod_{i=1}^N\Gamma_i}
\end{align}
$$

Let's ignore that $\Gamma$ because it'll be constant for all of the contours.
