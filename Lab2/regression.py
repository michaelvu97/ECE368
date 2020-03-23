import numpy as np
import matplotlib.pyplot as plt
import util

xv, yv = np.meshgrid(np.linspace(-1.0,1.0, 100),np.linspace(-1.0,1.0, 100), indexing='xy')
a_true = np.stack([[-0.1, -0.5]])

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here

    mean_vec = [0.0, 0.0]
    cov = [[beta, 0.0],[0.0, beta]]

    z = np.zeros((len(xv), len(yv)))

    for i in range(len(xv)):
        for j in range(len(yv)):
            a_0 = xv[i,j]
            a_1 = yv[i,j]
            z[i,j] = util.density_Gaussian(mean_vec, cov, np.stack([[a_0, a_1]]))
    
    plt.contour(xv, yv, z)
    plt.xlabel("a_0")
    plt.ylabel("a_1")
    plt.plot(a_true[0,0], a_true[0,1], 'r*')
    plt.show()

    return 
    
def posteriorDistribution(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    ### TODO: Write your code here

    res = np.zeros((len(xv), len(yv)))
    N = len(x)

    a_0_avg = 0.0
    a_1_avg = 0.0

    for k in range(len(xv)):
        for j in range(len(yv)):
            a_0 = xv[k,j]
            a_1 = yv[k,j]

            a_0_avg += a_0 / (len(xv) * len(yv))
            a_1_avg += a_1 / (len(xv) * len(yv))

            sum_thing = 0.0
            for i in range(N):
                sum_thing += np.square(z[i][0] - (a_1 * x[i][0]) - a_0)

            res[k,j] = np.exp( -((a_0**2 + a_1**2)/(2*beta)) - (sum_thing / (2*sigma2)))

    # mu_a|x,z is just a_MAP.

    X = np.zeros((N,2))
    for i in range(N):
        X[i,0] = 1.0
        X[i,1] = x[i]

    # TODO double check
    lam = sigma2 / beta

    # Comput mu
    mu = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X) + lam * np.identity(2)), np.transpose(X)),z)

    # Compute cov
    Cov = np.linalg.inv(np.transpose(X) @ X + lam * np.identity(2)) * sigma2

    plt.contour(xv, yv, res)
    plt.xlabel("a_0")
    plt.ylabel("a_1")

    plt.plot(a_true[0,0], a_true[0,1], 'r*')
    plt.plot(mu[0], mu[1], 'g*')
    plt.show()
   
    return (mu,Cov)

def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    
    N_x = len(x)

    # Reshape x
    X = np.zeros((N_x,2))
    for i in range(N_x):
        X[i,0] = 1.0
        X[i,1] = x[i]
    
    z_predictions = X @ mu

    variances = np.zeros(N_x)
    print(X[0,:].shape)
    for i in range(N_x):
        variances[i] = sigma2 + (np.transpose(X[i,:]) @ Cov @ X[i,:])

    for i in range(N_x):
        plt.plot(x[i], z_predictions[i], 'r*')
        plt.errorbar(x[i], z_predictions[i], np.sqrt(variances[i]), ecolor="red")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.ylim(-4, 4)
    plt.scatter(x_train, z_train)
    plt.show()

    return

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    priorDistribution(beta)

    ns_set = [1,5,100]

    # number of training samples used to compute posterior
    for ns in ns_set:
        # used samples
        x = x_train[0:ns]
        z = z_train[0:ns]

        # prior distribution p(a)


        # posterior distribution p(a|x,z)
        mu, Cov = posteriorDistribution(x,z,beta,sigma2)

        # distribution of the prediction
        predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)


   

    
    
    

    
