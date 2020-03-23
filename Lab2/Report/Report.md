# ECE368 Lab2 Report

Michael Vu - 1002473272

## 1

$$
\begin{align}
P(a)&=\frac{1}{2\pi\beta}\exp(-(\frac{a_0^2+a_1^2}{2\beta}))=\frac{1}{2\pi\beta}\exp(-\frac{||a||^2}{2\beta})\\

P(x_i,z_i|a)&=P(z_i|x_i,a)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp(-\frac{(z_i-a_1x_i-a_0)^2}{2\sigma^2})\\

P(x_i,z_i)&=P(z_i|x_i)=\int P(z_i|x_i,a)P(a)da=\Gamma_i\\

P(a|x,z)&=\frac{\frac{1}{2\pi\beta}(\frac1{\sqrt{2\pi\sigma^2}})^N\exp(-\frac{||a||^2}{2\beta}-\frac{1}{2\sigma^2}\sum_{i=1}^N(z_i-a_1x_i-a_0)^2)}{\prod_{i=1}^N\Gamma_i}\\
&=C\exp\Big(-\frac{||a||^2}{2\beta}-\frac{1}{2\sigma^2}\sum_{i=1}^N(z_i-a_1x_i-a_0)^2\Big)
\end{align}\\
$$

Let's ignore that $\Gamma$ because it'll be constant for all of the contours.

## 3

$$
\begin{align}
\text{let }D&=\{x_1,z_1,\dots,x_N,z_N\}\\
\text{let } \bold X&=\begin{bmatrix}1 & x_1\\
\vdots & \vdots\\
1 & x_N\end{bmatrix}\\
\bold {\hat{a}}&=\begin{bmatrix}a_0\\a_1\end{bmatrix}=(\bold X ^T \bold X + \frac{\sigma^2}{\beta}I)^{-1}\bold X ^T \bold z\\
P(z|x,D)&=P(a_1x+a_0+w|x,D)\sim \mathcal N(a_1x+a_0, \sigma ^2)\\
P(z|x,D)&=\frac{1}{\sqrt{2\pi}\sigma}\exp\Big(-\frac1{2\sigma^2} (z-a_1x-a_0)^2\Big)
\end{align}
$$

