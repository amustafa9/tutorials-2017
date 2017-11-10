---
title: Full-Waveform Inversion - Part 3``:`` Optimization
author: |
	Philipp Witte^1^\*, Mathias Louboutin^1^, Michael Lange^2^, Navjot Kukreja^2^, Fabio Luporini^2^, Gerard Gorman^2^, and Felix J. Herrmann^1,3^\
	^1^ Seismic Laboratory for Imaging and Modeling (SLIM), The University of British Columbia \
	^2^ Imperial College London, London, UK\
	^3^ now at Georgia Institute of Technology, USA \
bibliography:
	- bib_tuto.bib
---


## Introduction

This tutorial is the third part of a full-waveform inversion (FWI) tutorial series with a step-by-step walkthrough of setting up forward and adjoint wave equations and building a basic FWI inversion framework. For discretizing and solving wave equations, we use Devito, a Python domain-specific language for automated finite-difference code generation [@lange2016dtg]. The first two parts of this tutorial [@louboutin2017fwi] demonstrated how to model seismic shot records and compute the gradient of the FWI objective function through adjoint modeling. With these two key ingredients, we will now build an inversion framework for minimizing the FWI least-squares objective function and test it on a small 2D data set using the Overthrust model.

From the optimization point of view, full-waveform inversion is an extremely challenging problem, since not only do we need to solve expensive wave equations for a large number of shot positions and iterations, but the FWI objective function is also non-convex, meaning there exist (oftentimes many) local minima and saddle points. Furthermore, FWI is typically ill-posed, which means it is not possible to uniquely recover the parametrization of the subsurface from the seismic data alone that is collected at the surface. For these reasons, FWI forms a broad field of research with the focus lying on which misfit functions to choose, optimal parameterizations of the wave equations, optimization algorithms or how to include geological constraints and penalties [e.g. @vanleeuwen2013; @warner2014; @Peters2017].

This tutorial will demonstrate how we can set up a basic FWI framework with stochastic gradient descent, which can serve for the reader as a starting point for implementing more advanced optimization methods like conjugate gradient, spectral-projected gradients or second order methods [@nocedal2006]. Building a full framework for waveform inversion, including routines for data IO and parallelization, is outside the scope of this tutorial, so we will implement our inversion framework with Julia Devito, a Julia software package for seismic modeling and inversion based on Devito. Julia Devito provides mathematical abstractions and functions wrappers that allow to implement FWI and least-squares migration algorithms that closely follow the mathematical notation, while using Devito's automatic code generation for solving the underlying wave equations.

## Optimizing the full-waveform inversion objective function


In the previous tutorial, we demonstrated how to calculate the gradient of the FWI objective function with the $\ell_2$-misfit:

```math {#FWI}
	\mathop{\hbox{minimize}}_{\mathbf{m}} \hspace{.2cm} \Phi(\mathbf{m})= \sum_{i=1}^{n_s} \frac{1}{2} \left\lVert \mathbf{d}^\mathrm{pred}_i (\mathbf{m}, \mathbf{q}) - \mathbf{d}_i^\mathrm{obs} \right\rVert_2^2,
```

where $\mathbf{d}^\mathrm{pred}_i$ and $\mathbf{d}^\mathrm{obs}_i$ are the predicted and observed seismic shot records of the $i$th source location and $\mathbf{m}$ is the velocity model in slowness squared. As mentioned in the introduction, this objective function is non-convex, making it difficult to optimize, and its properties depend on many physical and environmental factors such as the acquisition geometry, the geology of the target area or frequency content of the observed data. Even though called *full-waveform inversion*, FWI in the standard formulation (Equation #FWI) relies primarily on transmitted waves, such as diving and turning waves, while utilizing reflections for FWI is much harder and subject of current research [e.g. @xu2012full]. Turning waves however are only available in the data for sufficiently long offsets and for geological structures where the velocity increases with depth, thus limiting the effectiveness of FWI to shallow and medium target depths. 

The most straight-forward approach for optimizing the FWI objective function is with local (gradient-based) optimization methods. Unlike numerically very expensive global methods, local methods find a minimum in vicinity of the starting point, with no guarantee that the solution is in fact the global minimum. The success of FWI therefore relies heavily on the initial guess, i.e. on the accuracy of the starting model. Initial velocity models that generate predicted shot records of which the events are shifted by more than half a wavelength (widely referred to as cycle skipping), cause local optimization algorithms to converge to local minima. Despite these issues, local gradient-based optimization algorithms are still the most widely used methods in practice, because the FWI gradient is comparatively easy and cheap to compute. We will therefore demonstrate how to implement FWI using gradient descent.

Full-waveform inversion with stochastic gradient descent (SGD) and bound constraints is outlined in Algorithm 1. *Stochastic* means that rather than calculating the full gradient in each iteration as a sum over all shot locations, we compute the gradient for a randomly selected subset of source locations, thus decreasing the amount of computations. The bound constraints simply ensure that velocities never become negative or so large that our modeling engine becomes unstable. The first step of every SGD iteration is to compute the function value and gradient for the current subset of shot and sources. The inner loop $k=1 \text{ to } n_{batch}$ runs over the batchsize (number of subsampled shots) and involves modeling the $k^{th}$ shot record $\mathbf{d}_k^\mathrm{pred}$ using the forward propagator $\mathcal{F}(\mathbf{m}_0, \mathbf{q}_k)$ from the first tutorial with the current model $\mathbf{m}_0$ and source wavelet $\mathbf{q}_k$. The function value $f$ is the sum of the $\ell_2$-misfits of each individual predicted shot record and the observed shot record $\mathbf{d}_k^\mathrm{obs}$. Using the gradient operator from the second part, we then compute the gradient for each individual data residual and sum them. The operator for calculating the gradient can be written as a linear operator $\nabla \mathcal{F}^\top$ that acts on the data residual. 

The gradient itself is not an update of our velocity model and only tells us in which direction we have to move. By performing a line search, we find a scalar $\alpha$, such that updating the velocity model leads to a decrease of the FWI function value, i.e. $\Phi(\mathbf{m}_0 - \alpha \mathbf{g}) < \Phi(\mathbf{m}_0)$. However, this condition alone is not enough to ensure convergence of the algorithm, which is my in practice the *sufficient decrease* condition $\Phi(\mathbf{m}_0 - \alpha \mathbf{g}) < \Phi(\mathbf{m}_0) - c \alpha \langle \mathbf{g}, \mathbf{g}\rangle$ is used, with $c$ being a small constant such as $c=10^{-4}$ [@nocedal2006]. After finding an acceptable step length and updating the model, we apply a bound constraints operator $\mathcal{P}_{bc}$ to the update, which projects the model onto the feasible set of velocities.

#### Algorithm: {#alg_lsrtm_sgd}
| Input: observed data $\mathbf{d}^\mathrm{obs}$, source wavelets $\mathbf{q}$, initial model $\mathbf{m}_0$
| **for** \ ``j=1 \text{ to } n_{iter}``
|
| 		# Calculate FWI function value and gradient for $n_{batch}$ shots
| 		**for** \ ``k=1 \text{ to } n_{batch}``
|				$\mathbf{d}^\mathrm{pred}_k = \mathcal{F}(\mathbf{m}_0, \mathbf{q}_k)$
|				$f = f + \frac{1}{2} \| \mathbf{d}^\mathrm{pred}_k - \mathbf{d}^\mathrm{obs}_k \|^2_2$
|				$\mathbf{g} = \mathbf{g} + \nabla \mathcal{F}^\top \Big( \mathbf{d}^\mathrm{pred}_k - \mathbf{d}^\mathrm{obs}_k \Big)$
| 		**end**
| 		
|		# Line search with sufficient decrease condition
|		**while** $\Phi (\mathbf{m} - \alpha \mathbf{g}) \geq \Phi(\mathbf{m}) - c\alpha \langle \mathbf{g}, \mathbf{g} \rangle$
|				$\alpha = \tau \alpha$
|		**end**
|
|		# Bound constraints and update model
|		$\mathbf{m}_0 = \mathcal{P}_{bc} \Big( \mathbf{m}_0 - \alpha \cdot \mathbf{g}  \Big)$
| **end**
: Full-waveform inversion with stochastic gradient descent, a line search and bound constraints.


This algorithm can serve as the basis for a broader range of optimization algorithms. E.g. for $n_{batch}$ equal to the full number of shots, we get the full gradient method or by saving the gradient of the previous iteration, we can employ the nonlinear conjugate gradient algorithm for finding a modified descent direction [@Powell??].

While a serial implementation of Algorithm 1 is fairly straight forward, 

But: to run for realistic data: need grid interpolations, data interpolations, parallelization, seg-y reader/writer, (check pointing ...)
	-> Julia framework

## FWI with Julia Devito

what is Julia Devito. Inversion framework. Translate algorithm to runnable Julia code. Use Devito to solve PDE. Set up PDE in Python as function
based around linear operators for forward/adjoint modeling, demigration/migration operators, and fwi_objective function. Examples with modeling..

fwi_objective(model, source, data) -> compute gradient + function value for wavelets and data. Like Python/Devito gradient but parallel loop over shots and sum gradients/function values.

-> use Julia Devito to translate algorithm to code


Set up 

```julia
 # Input: dobs, q, model
maxiter = 20
batchsize = 20
proj(x) = reshape(median([vec(mmin), vec(x), vec(mmax)]), model.n)

for j=1:maxiter

	# select current batch of shots
	idx = randperm(dobs.nsrc)[1:batchsize]
	
	# FWI objective function value and gradient
	fval, grad = fwi_objective(model, q[idx], dobs[idx])

	# line search with sufficient decrease conditions
	alpha = backtracking_linesearch(model, q[idx], dobs[idx], fval, grad, proj; alpha=1f-6)

	# update model and bound projection
	model.m = proj(model.m - alpha*reshape(grad, model.n))
end
```

#### Figure: {#result_marmousi_sgd}
![](Figures/fwi_overthrust.pdf){width=80%}
: Overthrust velocity model (top), FWI starting model (center) and inversion result after 20 iterations of stochastic gradient descent with bound constraints (bottom).

#### Figure: {#data_marmousi_sgd}
![](Figures/shot_records.pdf){width=95%}
: "Observed" seismic shot record (right), which is modeled using the true Overthrust model. The predicted shot record using the smooth initial model (center) is missing the reflections and has an incorrect turning wave, while the shot record modeled with the inversion result (right), looks very close to the original data.

easily extend to CG, different line search ...

alternative: interface optimization libraries to access L-BSFGS, SPG,
example with minConf?



## Installation

This tutorial is based on Devito version 3.1.0. It requires the installation of the full software with examples, not only the code generation API. To install Devito, run

	git clone -b v3.1.0 https://github.com/opesci/devito
	cd devito
	conda env create -f environment.yml
	source activate devito
	pip install -e .
 
### Useful links

- [Devito documentation](http://www.opesci.org/)
- [Devito source code and examples](https://github.com/opesci/Devito)
- [Tutorial notebooks with latest Devito/master](https://github.com/opesci/Devito/examples/seismic/tutorials)


## Acknowledgments

This research was carried out as part of the SINBAD II project with the support of the member organizations of the SINBAD Consortium. This work was financially supported in part by EPSRC grant EP/L000407/1 and the Imperial College London Intel Parallel Computing Centre.

## References


