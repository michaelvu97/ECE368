import numpy as np
import matplotlib.pyplot as plt
import util

def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models
    
    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of five elments: mu_male,mu_female,cov,cov_male,cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA 
    and 1 figure for QDA in this function         
    """
    x_male = []
    x_female = []
    for i in range(len(y)):
        if y[i] == 1:
            x_male.append(x[i])
        else:
            x_female.append(x[i])

    mu_male = np.average(x_male, axis=0)
    mu_female = np.average(x_female, axis=0)

    mu = np.average(x, axis=0)
    cov = np.matmul((x - mu),np.transpose(x - mu)) / len(x)

    cov_male = np.matmul(np.transpose(x_male - mu_male), x_male - mu_male) / len(x_male)
    cov_female = np.matmul(np.transpose(x_female - mu_female), x_female - mu_female) / len(x_female)

    xv, yv = np.meshgrid(np.linspace(50, 80), np.linspace(80,280))

    print(mu_male)
    print(cov_male)

    male_density = np.zeros((len(xv), len(yv)))
    female_density = np.zeros((len(xv), len(yv)))
    for x in range(len(xv)):
        for y in range(len(yv)):
            data_point = np.ndarray((1,2))
            data_point[0,0] = xv[x,y]
            data_point[0,1] = yv[x,y]
            male_density[x,y] = util.density_Gaussian(mu_male, cov_male, data_point)
            female_density[x,y] = util.density_Gaussian(mu_female, cov_female, data_point)

    # Visualization
    fig = plt.figure()
    ax1 = fig.add_subplot("111")
    ax1.plot(np.array(x_male)[...,0], np.array(x_male)[...,1], 'bs', label='male')
    ax1.plot(np.array(x_female)[...,0], np.array(x_female)[...,1], 'ro', label='female')
    plt.contour(xv, yv, male_density)
    plt.contour(xv, yv, female_density)
    plt.xlabel("height")
    plt.ylabel("weight")
    plt.legend()
    plt.xlim(50, 80)
    plt.ylim(80,280)
    plt.show()

    return (mu_male,mu_female,cov,cov_male,cov_female)
    

def misRate(mu_male,mu_female,cov,cov_male,cov_female,x,y):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate
    
    Inputs
    ------
    mu_male,mu_female,cov,cov_male,mu_female: parameters from discrimAnalysis
    
    x: a N-by-2 2D array contains the height/weight data of the N samples  
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """
    ### TODO: Write your code here
   
    
    return (mis_lda, mis_qda)


if __name__ == '__main__':
    
    # load training data and testing data
    x_train, y_train = util.get_data_in_file('trainHeightWeight.txt')
    x_test, y_test = util.get_data_in_file('testHeightWeight.txt')
    
    # parameter estimation and visualization in LDA/QDA
    mu_male,mu_female,cov,cov_male,cov_female = discrimAnalysis(x_train,y_train)
    
    # misclassification rate computation
    mis_LDA,mis_QDA = misRate(mu_male,mu_female,cov,cov_male,cov_female,x_test,y_test)
    

    
    
    

    
