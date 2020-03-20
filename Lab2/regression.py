import numpy as np
import matplotlib.pyplot as plt
import util

xv, yv = np.meshgrid(np.linspace(-1,1),np.linspace(-1,1))

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

    z = np.zeros((len(xv), len(yv)))
    N = len(x)

    for i in range(len(xv)):
        for j in range(len(yv)):
            a_0 = xv[i,j]
            a_1 = yv[i,j]

            sum_thing = 0
            for i in range(N):
                sum_thing += (z[i][0] - a_1 * x[i][0] - a_0) ** 2

            z[i,j] = (1.0 / 2 * np.pi) * np.exp(-(a_0**2 + a_1**2)/(2*beta))*np.power(1.0/np.sqrt(2*np.pi*sigma2), N)*np.exp(-(1.0/(2*sigma2)) * sum_thing)

    plt.contour(xv, yv, z)
    plt.xlabel("a_0")
    plt.ylabel("a_1")
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
    
    
    return 

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    # number of training samples used to compute posterior
    ns  = 5
    
    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]
    
    # prior distribution p(a)
    priorDistribution(beta)
    
    # posterior distribution p(a|x,z)
    mu, Cov = posteriorDistribution(x,z,beta,sigma2)
    
    # distribution of the prediction
    predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)
    

   

    
    
    

    
