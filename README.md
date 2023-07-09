# Gradient-Descent (Overview)
Gradient descent is an iterative optimization algorithm used to find the minimum of a function. It is commonly used in machine learning and deep learning for training models by adjusting their parameters to minimize a loss or cost function.

The basic idea behind gradient descent is to iteratively update the parameters of a model in the direction of the steepest descent of the cost function. The cost function measures how well the model performs on the training data, and the goal is to find the parameter values that minimize this cost.

Here's a high-level overview of how gradient descent works:

1. Initialization: Start by initializing the model's parameters with some initial values.

2. Compute the gradient: Calculate the gradient of the cost function with respect to each parameter. The gradient represents the direction of the steepest ascent, so we negate it to obtain the direction of the steepest descent.

3. Update the parameters: Adjust the parameters by taking a step in the opposite direction of the gradient. The size of the step is determined by the learning rate, which is a hyperparameter that controls the magnitude of the parameter updates.

4. Repeat steps 2 and 3: Continue to compute the gradient and update the parameters iteratively until convergence or a maximum number of iterations is reached. Convergence is typically determined by monitoring the change in the cost function or the magnitude of the gradient.

There are different variants of gradient descent, such as batch gradient descent, stochastic gradient descent (SGD), and mini-batch gradient descent. In batch gradient descent, the entire training dataset is used to compute the gradient at each iteration. In SGD, only one randomly selected sample from the training dataset is used at each iteration. Mini-batch gradient descent is a compromise between these two, where a small batch of randomly selected samples is used to compute the gradient.

Gradient descent has its challenges. It can converge slowly if the learning rate is too small or overshoot the minimum if the learning rate is too large. Choosing an appropriate learning rate is crucial. Additionally, gradient descent can get stuck in local minima, where it finds a suboptimal solution instead of the global minimum.

To mitigate some of these challenges, techniques such as learning rate schedules, momentum, and adaptive learning rates (e.g., AdaGrad, RMSprop, Adam) are often employed in practice.

Overall, gradient descent is a fundamental optimization algorithm used to train machine learning models by iteratively updating their parameters to minimize a cost function.

# What is Gradient Descent?
Gradient descent is an optimization algorithm which is commonly-used to train machine learning models and neural networks.  Training data helps these models learn over time, and the cost function within gradient descent specifically acts as a barometer, gauging its accuracy with each iteration of parameter updates. Until the function is close to or equal to zero, the model will continue to adjust its parameters to yield the smallest possible error. Once machine learning models are optimized for accuracy, they can be powerful tools for artificial intelligence (AI) and computer science applications.

# How does gradient descent work?
Before we dive into gradient descent, it may help to review some concepts from linear regression. You may recall the following formula for the slope of a line, which is y = mx + b, where m represents the slope and b is the intercept on the y-axis.

You may also recall plotting a scatterplot in statistics and finding the line of best fit, which required calculating the error between the actual output and the predicted output (y-hat) using the mean squared error formula. The gradient descent algorithm behaves similarly, but it is based on a convex function.

The starting point is just an arbitrary point for us to evaluate the performance. From that starting point, we will find the derivative (or slope), and from there, we can use a tangent line to observe the steepness of the slope. The slope will inform the updates to the parameters—i.e. the weights and bias. The slope at the starting point will be steeper, but as new parameters are generated, the steepness should gradually reduce until it reaches the lowest point on the curve, known as the point of convergence.   

Similar to finding the line of best fit in linear regression, the goal of gradient descent is to minimize the cost function or the error between predicted and actual y. In order to do this, it requires two data points—a direction and a learning rate. These factors determine the partial derivative calculations of future iterations, allowing it to gradually arrive at the local or global minimum (i.e. point of convergence).

**Learning rate**
Learning rate (also referred to as step size or the alpha) is the size of the steps that are taken to reach the minimum. This is typically a small value, and it is evaluated and updated based on the behavior of the cost function. High learning rates result in larger steps but risk overshooting the minimum. Conversely, a low learning rate has small step sizes. While it has the advantage of more precision, the number of iterations compromises overall efficiency as this takes more time and computations to reach the minimum.

**The cost (or loss) function**
The cost (or loss) function measures the difference, or error, between actual y and predicted y at its current position. This improves the machine learning model's efficacy by providing feedback to the model so that it can adjust the parameters to minimize the error and find the local or global minimum. It continuously iterates, moving along the direction of the steepest descent (or the negative gradient) until the cost function is close to or at zero. At this point, the model will stop learning. Additionally, while the terms, cost function and loss function, are considered synonymous, there is a slight difference between them. It’s worth noting that a loss function refers to the error of one training example, while a cost function calculates the average error across an entire training set.

**Convergence**
Convergence in gradient descent refers to the point at which the algorithm reaches an optimal or near-optimal solution. In the context of gradient descent, convergence means that the algorithm has found a set of parameter values that minimize the cost function to a satisfactory extent.

![image](https://github.com/TITHI-KHAN/Gradient-Descent/assets/65033964/4e0bcf9a-8a87-4509-9162-872fca00c4a3)

# How does the weight update?
In gradient descent, the weight updates are determined by the gradient of the cost function with respect to the weights of the model. The gradient indicates the direction of steepest ascent, so to minimize the cost function, the weights need to be updated in the opposite direction of the gradient.

Here's the general process for updating the weights in gradient descent:

1. Initialize the weights: Start by initializing the weights of the model with some initial values.

2. Compute the gradient: Calculate the gradient of the cost function with respect to each weight. This involves taking the partial derivative of the cost function with respect to each weight. The gradient represents the direction and magnitude of the steepest ascent.

3. Update the weights: Adjust the weights by taking a step in the opposite direction of the gradient. The size of the step is determined by the learning rate, denoted as α (alpha), which is a hyperparameter. The weight update formula is typically of the form: weight = weight - learning_rate * gradient. This formula subtracts a fraction of the gradient from the current weight value, effectively moving the weight in the direction of steepest descent.

4. Repeat steps 2 and 3: Iterate the process of computing the gradient and updating the weights until convergence is achieved or a maximum number of iterations is reached.
   
By iteratively updating the weights in the direction of the negative gradient, gradient descent aims to find the optimal or near-optimal values of the weights that minimize the cost function and improve the model's performance.

# Linear Regression using Gradient Descent 
![image](https://github.com/TITHI-KHAN/Gradient-Descent/assets/65033964/2a1f9d2f-9245-4399-b7b1-91ad0bc0c616)

**Mathematical Formula:**

![image](https://github.com/TITHI-KHAN/Gradient-Descent/assets/65033964/4f0f4476-b0b5-413b-8616-3157f6159e64)

**Important Calculus:**
![image](https://github.com/TITHI-KHAN/Gradient-Descent/assets/65033964/e97d8120-1b79-4ecc-8bf0-9309f406d2a3)

# Steps
![image](https://github.com/TITHI-KHAN/Gradient-Descent/assets/65033964/bd867024-1082-43ce-b03f-9b7fcbefad5b)

![image](https://github.com/TITHI-KHAN/Gradient-Descent/assets/65033964/05f0e439-ec09-4086-abd7-fca2870d7f17)

![image](https://github.com/TITHI-KHAN/Gradient-Descent/assets/65033964/668755d3-0630-4f01-a873-4867886cd755)

![image](https://github.com/TITHI-KHAN/Gradient-Descent/assets/65033964/13c8976a-9e8f-47fd-a3ee-392031dcd800)

![image](https://github.com/TITHI-KHAN/Gradient-Descent/assets/65033964/6127d9d0-1053-444f-9bdb-04d43edd2157)

![image](https://github.com/TITHI-KHAN/Gradient-Descent/assets/65033964/40f18810-15b4-48af-afc4-73cc0361e627)

# Linear Regression with Single Variable
![image](https://github.com/TITHI-KHAN/Gradient-Descent/assets/65033964/0329b225-ef04-4582-99aa-ba84a3914d5d)

# Linear Regression with Multiple Variables
**Mathematical Representation:**
![image](https://github.com/TITHI-KHAN/Gradient-Descent/assets/65033964/fd8eca6f-807c-49e6-b9a7-ab0357112294)

# Linear Regression with Single Vs. Multiple Variables
**Mathematical Representation:**
![image](https://github.com/TITHI-KHAN/Gradient-Descent/assets/65033964/e4126fdb-13c5-4bca-b2be-54622b04a6c3)

# R Squared Value / Model Accuracy (for Regression Algorithms)
**Mathematical Calculation:**
![image](https://github.com/TITHI-KHAN/Gradient-Descent/assets/65033964/971d42f6-97ae-4234-bfca-7f0789ef80cc)

# What are Local Minima and Global Minima in Gradient Descent?

![image](https://github.com/TITHI-KHAN/Gradient-Descent/assets/65033964/8d980be3-17b5-4f56-b075-2f1b8978bdb7)

# Local minima
The point in a curve which is minimum when compared to its preceding and succeeding points is called local minima.

# Global minima
The point in a curve which is minimum when compared to all points in the curve is called Global Minima.

**For a curve there can be more than one local minima, but it does have only one global minima.**

In gradient descent we use this local and global minima in order to decrease the loss functions.
