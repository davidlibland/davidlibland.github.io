---
title: Saddle Points and Stochastic Gradient Descent
---
Essentially all machine learning models are trained using gradient descent. However, neural networks introduce two new challenges for gradient descent to cope with: saddle points and massive training sets. In this post we describe how these two challenges conspire together to create deadly traps, and we discuss means to escape them.

## Background: Optimizing a loss function:
Machine learning algorithms attempt to model observed data in order to predict properties about unobserved data, and in most cases they do this by:

1. Parametrizing a space of models (with some parameters $\vec{\bf{x}} = (x_1,\dots, x_m)$),
2. Defining a loss function $f(\vec{\bf{x}})\to \mathbb{R}$ which measures the total error between the model at parameters $\vec{\bf{x}} = (x_1,\dots, x_m)$ and the observed data,
3. Finding the optimal parameters $\vec{\bf{x}} = (x_1,\dots, x_m)$  which minimize the loss.

Assuming we've completed steps (1) and (2), the question then becomes: How do we minimize a high dimensional function $f(\vec{\bf{x}})\to \mathbb{R}$?

<!--![Loss Function](../images/saddle-points-and-sdg/loss_function.png)-->

## We use gradient descent:
Although there are many possible ways to minimize a function, by and far the most successful approach for machine learning is to use gradient descent. This is a greedy algorithm, where we make an initial first guess $\vec{\bf{x}}_0$ for the optimal parameters, and then iteratively modifying that guess by moving in the direction in which $f$ decreases most quickly, that is:

$$\vec{\bf{x}}_n = \vec{\bf{x}}_{n-1} - \nabla f$$

(where $\nabla f$ denotes the gradient of $f$). With some basic assumptions, the sequence of guesses $\vec{\bf{x}}_0, \vec{\bf{x}}_1, \vec{\bf{x}}_2, \vec{\bf{x}}_3, \dots$ can be shown to improve steadily, and if $f$ is sufficently nice (for example if $f$ is strongly convex), then this sequence will converge to a (local) minimum of $f$ fairly quickly.

![Gradient Descent, pictured here by a red path starting from a poor initial choice for $\vec{\bf{x}}$ (where $f(\vec{\bf{x}})$ is large) converges fairly quickly to a local minimum.](../images/saddle-points-and-sdg/gradient_descent1.png)

## Saddle points
We now introduce the first villain in our saga, saddle points, which are 
known to cause problems for gradient descent. A saddle point is a critical point[^critical_point] of a function which is neither a local minima or maxima.

[^critical_point]: That is $\vec{\bf{x}}$ is a critical point if $\nabla f=0$
 at $\vec{\bf{x}}$. Said differently $f$ is flat to first order near $\vec{\bf{x}}$.

![The red and green curves intersect at a generic saddle point in two dimensions. Along the green curve the saddle point looks like a local minimum, while it looks like a local maximum along the red curve.](../images/saddle-points-and-sdg/saddle_point.png)

Locally, up to a rotatation and translation, any function $f$ is well approximated near a saddle point by

$$\overset{concave}
{\overbrace{\frac{-a_1}{2}x_1^2+\dots +\frac{-a_k}{2}x_k^2}}
+ \overset{convex}{\overbrace{\frac{a_{k+1}}{2}x_{k+1}^2+\dots+\frac{a_m}{2}x_m^2}}$$

where $1<k<m$ and all $a_i>0$, and the translation sends the saddle point to the origin.

#### Gradient descent proceeds extremely slowly near a saddle point. 
Why? Well, $\nabla f=0$ at a saddle point $\vec{\bf{x}}$ so the gradient $\nabla f\simeq 0$ will be extremely small near the saddle point. Recall that each gradient descent update is given by $\vec{\bf{x}}_n = \vec{\bf{x}}_{n-1}- \nabla f$. So $\vec{\bf{x}}_n \simeq \vec{\bf{x}}_{n-1}$, and each update barely improves the current guess. You can see this quite clearly in the following picture:

![Gradient descent proceeds slowly near a saddle point, so the $\vec{\bf{x}}_n$ cluster more closely around the saddle points.](../images/saddle-points-and-sdg/gradient_descent2.png)

Nevertheless, if we choose $\vec{\bf{x}}_0$ randomly, then almost certainly:

#### Gradient descent will (eventually) escape saddle points  

Recall the gradient descent formula
$$\vec{\bf{x}}_n = \vec{\bf{x}}_{n-1} +\Delta \vec{\bf{x}}_{n-1}, \quad \text{ where } \quad \Delta \vec{\bf{x}}= - \nabla f$$
in coordinates, this reads:
$$\Delta x_i = - \frac{\partial f}{\partial x_i}$$

Meanwhile the local model for a saddle point at the origin is:
$$f(x_1,\dots,x_n) = \overset{concave}
{\overbrace{\frac{-a_1}{2}x_1^2+\dots +\frac{-a_k}{2}x_k^2}}
+ \overset{convex}{\overbrace{\frac{a_{k+1}}{2}x_{k+1}^2+\dots+\frac{a_m}{2}x_m^2}}$$

Now, we have 
$$\frac{\partial f}{\partial x_i} = -a_ix_i\quad \text{ for } i<=k\quad\quad
\text{ and }\quad\quad\frac{\partial f}{\partial x_i} = a_ix_i\quad \text{ for } i>k$$

So:
    $$\Delta x_i = a_ix_i \quad \text{ for } i<=k \quad \text{ (repulsive dynamics)}$$
and
$$\Delta x_i = -a_ix_i \quad \text{ for } i>k \quad \text{ (attractive dynamics)}$$

Though we are attracted to the saddle point in some directions, we are eventually ejected from the saddle point by the repulsive forces.

Finally, we should emphasize that there are **lots** of saddle points in deep neural networks.

## Typically, the loss function in deep neural networks is noisy
For most machine learning algorithms, the loss function can be written as a sum over the training data:

$$ f(\vec{\bf{x}}) = \sum_j^N f_j(\vec{\bf{x}}),$$

where $N$ is the number of training samples, and each $f_i(\vec{\bf{x}})$ measures the model's error on the $i$-th training  sample. 
<!--This allows us to compute the gradient $\nabla f$ as a sum of terms:
$$ \nabla f(\vec{\bf{x}}) = \sum_j^N \nabla f_j(\vec{\bf{x}}).$$
-->
Now, deep neural networks need massive data sets to train successfully, and so it extremely costly to evaluate the loss over the entire training set. Instead, one typically approximates the loss on a randomly chosen minibatch $J\subset \{1,\dots, N\}$ of data, 
$$ f_{J-approx}(\vec{\bf{x}}) = \sum_{j\in J} f_j(\vec{\bf{x}}).$$
This subsampling from the training data results in a _noisy_ loss function.

In turn, we modify the gradient descent algorithm as follows: on iteration, we choose a random minibatch $J\subset \{1,\dots, N\}$ and update our previous guess by:
$$\vec{\bf{x}}_n = \vec{\bf{x}}_{n-1} +\Delta \vec{\bf{x}}_{n-1}, \quad \text{ where } \quad \Delta \vec{\bf{x}}= - \nabla f_{J-approx}$$
It is worthwhile noting that minibatches chosen randomly at each iteration are mutually independent. The resulting algorithm is known as **stochastic gradient descent**.

Adding this noise, however, can have some suprising consequences near saddle points:

## Example:
Consider the following function, standard gradient descent does an excellent job of finding the minimum:

![](../images/saddle-points-and-sdg/sdg_0noise.png)

Now, we will add some noise[^medium_noise]. Notice that it doesn't roll 
immediately into the basin, but lingers along the ridge somewhat.

[^medium_noise]: By gently shaking the surface: that is, we translate it by a
 normally distributed random variable with standard deviation 1.

![](../images/saddle-points-and-sdg/sdg_1noise.png)

Adding more noise[^strong_noise] causes it to converge fairly rapidly to the 
saddle point between the two minima.

[^strong_noise]: By shaking the surface: that is, we translate it by a 
normally distributed random variable with standard deviation 2.

![](../images/saddle-points-and-sdg/sdg_2noise.png)

What is going on here? How can noise cause gradient descent to become trapped at saddle points?


<!--## Overview of talk:
- I will describe why noise can trap us at saddle points
- I will describe some ways to escape saddle points
- Open questions-->

## Why noise hurts

The local model for a saddle point in a noisy loss function is:
$$f(x_1,\dots,x_n) = \overset{concave}
{\overbrace{\frac{-a_1-\xi_1}{2}x_1^2+\dots +\frac{-a_k-\xi_k}{2}x_k^2}}
+ \overset{convex}{\overbrace{\frac{a_{k+1}-\xi_{k+1}}{2}x_{k+1}^2+\dots+\frac{a_m-\xi_m}{2}x_m^2}}\\
-\eta_1x_1-\cdots-\eta_mx_m
$$
where $\xi_i$ and $\eta_i$ are mean zero random variables encoding the noisy measurement of each coefficient.

So, we have 
$$\frac{\partial f}{\partial x_i} = \pm a_ix_i +\xi_ix_i+\eta_i$$
And the gradient descent update becomes:
$$\Delta x_i =\pm a_i x_i+ \xi_i x_i+\eta_i$$

Let's consider the seperate components
    $$ \Delta x =
    \overset{\text{Usual gradient}}{\overbrace{\pm ax}} 
    +\overset{\text{Attractive noise}}{\overbrace{\xi x}}+\overset{\text{Diffusive noise}}{\overbrace{\eta}}$$

## Attractive noise:
Let's start by considering the attractive noise in isolation, ie. $\Delta x = \xi x$. The corresponding update is
$$x_{n+1}=x_n + \xi x_n = (1+\xi) x_{n}$$

Suppose that $\xi$ equals either $-\sigma$ or $\sigma$ with equal probability. Then after 1 iteration, $x$ is rescaled by either a factor of $1-\sigma$ or $1+\sigma$ (with equal probability):

![](../images/saddle-points-and-sdg/stage_1.jpeg)
Next consider what the possiblilities are after two iterations: $x$ is rescaled by a factor of either
- $(1-\sigma)^

![Stage 2](../images/saddle-points-and-sdg/stage_2.jpeg)

![Stage 3](../images/saddle-points-and-sdg/stage_4.jpeg)

![Stage N](../images/saddle-points-and-sdg/stage_n.png)


# Diffusive noise:
Recall:
    $$ \Delta x =
    \overset{\text{Usual gradient}}{\overbrace{\pm ax}} 
    +\overset{\text{Attractive noise}}{\overbrace{\xi x}}+\overset{\text{Diffusive noise}}{\overbrace{\eta}}$$
    
So the diffusive component is:
    $$ \Delta x = \eta$$
    
At stage $N$, we have
$$x_N=x_0+\sum_{n=0}^N \Delta x_n = x_0+\sum_{i=0}^N \eta_i$$

But $\eta_i$ are iid random variables with mean zero and norm $\tau$, so $\sum_{i=0}^N \eta_i\sim N(0, \sqrt{N}\tau)$. Therefore:

$$x_N\sim N(x_0, \tau\sqrt{N})$$

In summary, $x$ diffuses at a rate of $\tau\sqrt{N}$.

## Does the diffusive or attractive noise dominate?

$$ \Delta x =
    \overset{\text{Usual gradient}}{\overbrace{\pm ax}} 
    +\overset{\text{Attractive noise}}{\overbrace{\xi x}}+\overset{\text{Diffusive noise}}{\overbrace{\eta}}$$
    
#### Replace with a stochastic differential equation:

$$dx = ax \:dt+\xi x \:dt +\eta\: dt$$
where $\xi$ and $\eta$ are white noise.

Let $\sigma$ and $\tau$ represent the scaling factors such that 
$W_t=\int_0^t\frac{\xi}{\sigma}\:dt$ and $V_t=\int_0^t\frac{\eta}{\tau}\:dt$ are standard brownian motion. Then

$$dx = ax \:dt+\sigma x \:dW +\tau\: dV$$

## Case 1: $\xi$ and $\eta$ are correlated

In this case we may assume $V = W$, so
$$dx = ax \:dt+\sigma x \:dW +\tau\: dW$$
Substituting $x = y-\tau/\sigma$, we get
$$dy = a(y-\tau/\sigma) \:dt  + \sigma y \:dW$$

This elimates the diffusive noise, replacing it by a constant rate of change of $-a\tau/\sigma$. We can solve this exactly, to get:
$$x_t = -\tau/\sigma+(x_0+\tau/\sigma)e^{(a-\sigma^2/2)t+\sigma W_t}-\frac{a\tau}{\sigma}\int_0^t e^{(a-\sigma^2/2)(t-s)+\sigma (W_t-W_s)}ds$$

$$x_t = -\tau/\sigma+(x_0+\tau/\sigma)e^{(a-\sigma^2/2)t+\sigma W_t}-\frac{a\tau}{\sigma}\int_0^t e^{(a-\sigma^2/2)(t-s)+\sigma (W_t-W_s)}ds$$

We see that the effective rate of growth is $a-\sigma^2/2$
- this rate depends only on the intensity of the attractive noise.
    - It is independent of the intensity of the diffusive noise.
- In particular, if the noise $\sigma$ is sufficiently strong, the saddle point will be attractive.

## Stationary distribution:

$$\lim_{t\to \infty} -x_t-\tau/\sigma$$ 
is inverse-gamma distributed with scale $\beta = 2\frac{\lvert a\tau\rvert}{\sigma^3}$ and shape $\alpha = 1-2a/\sigma^2$



This has no mean, but it's mode is at $-\tau/\sigma+\beta/(\alpha+1)= -\tau/\sigma-\frac{a\tau}{\sigma^3-a\sigma}$

![Inverse Gamma](../images/saddle-points-and-sdg/correlated.png)

## Case 2: $\xi$ and $\eta$ are *not* correlated
In this case, $\eta$ serves to time-average the solution
$$\hat x_t=x_0e^{(a-\sigma^2/2)t+\sigma W_t}$$
of the equation 
$$d\hat x = a\hat x \:dt+\sigma \hat x \:dW$$
Explictly, we have
$$x_t = x_0e^{(a-\sigma^2/2)t+\sigma W_t} + \int_0^te^{(a-\sigma^2/2)(t-s)+\sigma (W_t-W_s)}\eta \: ds$$


![Log Normal mixture of gaussians](../images/saddle-points-and-sdg/uncorelated.png)


## How should we escape saddle points in the presence of noise?

We need to reduce the noise!

## Options
- Perturbed Stochastic Gradient Descent
- Stochastic Variance Reduction Gradient Descent (SVRGD)
- Increase the minibatch size
- Decrease the learning rate
- Anneal the learning rate
- Use ReLu's

## Perturbed Stochastic Gradient Descent

### Idea: 
Increase the intensity of the diffusive noise $\eta$ in the equation 
$$dx = ax \:dt+\xi x \:dt +\eta\: dt$$

### Problems:
- This does not guarantee that we will escape, instead 
    - it increases the odds that we will escape, 
    - it greatly speeds how quickly we pass through non-attractive saddle points (where the attractive noise is small).


## Stochastic Variance Reduction Gradient Descent (SVRGD)
### Idea:
Suppose you have $N$ training samples.

until converged:<br>
$\quad\quad$ store the current location $x$ as a landmark:<br>
$\quad\quad$ set $x_{*} := x$<br>
$\quad\quad$ compute the full gradient $\nabla f_{full}(x_*)$ at $x_*$<br>
$\quad\quad$ for i = 1 ... N:<br>
$\quad\quad\quad\quad$ compute the approximate gradient $\nabla f_{approx}(x)$ at x<br>
$\quad\quad\quad\quad$ compute the approximate gradient $\nabla f_{approx}(x_*)$ at $x_*$<br>
$\quad\quad\quad\quad$ set $\nabla f_{VR} := \nabla f_{approx}(x)-\nabla f_{approx}(x_*)+\nabla f_{full}(x_*)$<br>
$\quad\quad\quad\quad$ set $x := x - \nabla f_{VR}$
    
    

- Interestingly, SVRGD serves correlate the two noises $\xi$ and $\eta$ in the equation $$dx = ax \:dt+\xi x \:dt +\eta\: dt$$
- Infact, at the end of every inner loop $x-x_*$ exhibits an inverse gamma distribution with scale proportionate to the distance from $x_*$ to the saddle point:


![SVRG](../images/saddle-points-and-sdg/svrg.png)


## SVGRD guarantees that you will escape the saddle point

Infact, you expect to escape geometrically (in the outer loop)

### Problems:
- Unfortuately, you may need to compute the full gradient (updating $x_*$) before each foolproof step away from the saddle point.
- This results in the same computational cost as gradient descent (which is considered too expensive for training deep neural networks).


## Increase the minibatch size

### Idea:
- Full batch Gradient Descent will never get stuck at a saddle point 
    - Why? It reduces the attactive noise to zero
- Similarly, using large minibatches greatly decreases the odds of being stuck at a saddle point 
    - Why? It reduces the attactive noise
    
### Problems:
- Unfortunately using larger minibatches becomes computationally expensive

## Decrease the learning rate

### Idea:
Instead of making the updates $$x_n = x_{n-1} +\Delta x_{n-1},$$ make smaller updates: $$x_n = x_{n-1} +\rho \Delta x_{n-1}$$ (where $0<\rho<1$ is small).

- This decreases the attractive noise by a factor of $\sqrt{\rho}$

### Problems:
- Learning takes longer
- It doesn't guarantee that we will escape, it only increases the odds

## Anneal the learning rate

### Idea:
Decrease the learning rate on a regular schedule:
$$x_n = x_{n-1} +\rho \Delta x_{n-1}$$
$$p_n = \rho_0/n$$

- This **Guarantees** tha we will escape the saddle point (eventually),
- This is recommended for other reasons too

### Problems:
- It may still take too long to escape

## Use ReLu's

### Idea:
- Regularized linear units result in piecewise linear loss functions
- This eliminates the attractive noise in the local update
- It dramatically changes the dynamics


![RELU](../images/saddle-points-and-sdg/relu.png)

## Open questions
- What are the odds of the attractive noise being significantly strong?
    - How is affected by the dimensionality?
        - Is it less likely in high dimensions?
    - How is it affected by depth vs width of the network?
- How does momentum affect the dynamics?
- Trade off between increasing batch size (reduces variance linearly) vs increasing network size (reduces odds of only bad directions, while also increasing number of saddle points).
