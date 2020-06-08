def ada_weighting(forecasts, actual, lr=0.2, p=5, prior_weights = None):
    """
    Args:
        forecasts: ndarray, dim n x m
        actual: ndarray, dim n 
        lr: float
        p: int
        
    Returns: ndarray, n by m
    """
    weights = np.zeros((forecasts.shape[0], forecasts.shape[1]))
    errors = np.abs(forecasts - actual)**p
    
    # Initialize default weights for each forecast
    if prior_weights:
        curr_weights = prior_weights
    else:
        curr_weights = [1/fcsts.shape[1]] * fcsts.shape[1]
    # Iterative updates over every timestep
    for i in range(forecasts.shape[0]):
        # Normalize errors 
        curr_errors = (errors[i:i+1] + 0.00001) #/ np.sum(errors[i:i+1])  # 0.00001 is adjusting for possible 0 error

        alpha = lr * np.log(abs(1 - curr_errors)/curr_errors)

        #Weight update
        curr_weights = curr_weights * np.exp(alpha*curr_errors)
        curr_weights /= np.sum(curr_weights)
        
        weights[i] = curr_weights
        
    return weights


def proj_prob_simplex(v):
    u = -np.sort(-v)
    rho = 0
    for j in range(1, len(v)+1):
        if ((u[j-1] + (1 / j) * (1 - np.sum(u[0:j]))) > 0):
            rho = j
    lambd = (1/rho)*(1-np.sum(u[0:rho]))
    x = np.maximum(v + lambd, 0)
    return x


def online_grad_descent(forecasts, actual, lr=0.001, p=1, prior_weights = None):
    """
    Args:
        forecasts: ndarray, dim n x m
        actual: ndarray, dim n 
        lr: float
        p: int
        
    Returns: ndarray, n by m
    """
    weights = np.zeros((forecasts.shape[0], forecasts.shape[1]))
    
    # Initialize default weights for each forecast
    if prior_weights:
        curr_weights = prior_weights
    else:
        curr_weights = [1/fcsts.shape[1]] * fcsts.shape[1]
    # Iterative (online) updates over every timestep
    for i in range(forecasts.shape[0]):
        gradient = p * ((forecasts[i:i+1] - actual[i]) * ((forecasts[i:i+1]*curr_weights) - (actual[i]*curr_weights))**(2*p-1))  / np.abs(((forecasts[i:i+1]*curr_weights) - actual[i])**p)
        
        # Update weights
        curr_weights -= lr * gradient
        
        # Project weights back to being positive and sum to 1
        curr_weights = proj_prob_simplex(curr_weights.flatten())
        curr_weights = curr_weights.reshape((1,7))

        weights[i] = curr_weights
        
    return weights

