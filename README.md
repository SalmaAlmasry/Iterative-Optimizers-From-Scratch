# Iterative Optimization Algorithms From Scratch :chart_with_downwards_trend:
Implementing GD, Momentum based GD, NAG, AdaGrad, RMSProp and Adam Optimizers from scratch with python.<br> <br>
These optimization algorithms play a crucial role in training deep neural networks and optimizing complex models to find optimal values for the parameters of a model by iteratively updating them based on the gradients of the loss function. <br> <br>
Each algorithm has its strengths and is suited for different scenarios.

### Gradient Descent (GD)
A fundamental optimization algorithm used to minimize the loss function in machine learning models. It iteratively updates the model parameters in the direction of steepest descent of the loss function gradient.

### Momentum-Based Gradient Descent
Enhances GD by incorporating momentum. It introduces a momentum term that accumulates the gradients of previous iterations to determine the direction of the update. This momentum helps accelerate convergence and overcome local minima.

### Nesterov Accelerated Gradient (NAG)
Also known as Nesterov Momentum, improves upon momentum-based GD by adjusting the update direction. NAG calculates the gradient not at the current position but at a point slightly ahead in the direction of the accumulated momentum. This lookahead approach helps avoid overshooting the minimum and leads to faster convergence.

### AdaGrad (Adaptive Gradient)
Is an optimization algorithm that adapts the learning rate for each parameter based on the historical gradient information. It assigns larger learning rates to parameters with infrequent updates and smaller learning rates to parameters with frequent updates. AdaGrad performs well for sparse data and helps converge faster for steep directions.

### RMSProp (Root Mean Square Propagation)
Addresses the issue of diminishing learning rates in AdaGrad by introducing an exponentially decaying average of squared gradients. It scales the learning rate for each parameter based on the average of its recent squared gradients. RMSProp prevents the learning rate from becoming too small and performs well in non-convex optimization tasks.

### Adam (Adaptive Moment Estimation)
Combines the benefits of momentum-based GD and RMSProp. It maintains both a decaying average of past gradients (like momentum) and a decaying average of past squared gradients (like RMSProp). Adam adapts the learning rate for each parameter and provides good performance across a wide range of optimization problems.

## BFGS
BFGS is an iterative method that aims to find the optimal solution by approximating the Hessian matrix (the matrix of second partial derivatives) of the objective function. Instead of directly computing the Hessian, BFGS constructs an estimate of the Hessian using gradient information and updates this estimate at each iteration. By iteratively updating the estimate of the Hessian, BFGS efficiently converges towards the optimum.

The advantage of BFGS over other optimization algorithms, such as gradient descent, is that it can converge more quickly by approximating the curvature of the objective function. However, BFGS requires more memory as it needs to store and update the estimate of the Hessian matrix.

### Finally
Choosing the appropriate optimization algorithm depends on factors such as the problem complexity, available data, and desired convergence speed. It is essential to experiment and tune these algorithms to achieve optimal performance for specific machine learning tasks.
