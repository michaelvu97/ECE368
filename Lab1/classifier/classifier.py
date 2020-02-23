import os.path
import numpy as np
import matplotlib.pyplot as plt
import util
import math

def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set
    
    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of 
    spam files, and the second element is a list of ham files.

    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict 
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the 
    smoothed estimates of q_d 
    """

    all_words = {}
    for i in range(2):
        counts = util.get_counts(file_lists_by_category[i])
        for word in counts:
            all_words[word] = True

    spam_frequencies = util.get_word_freq(file_lists_by_category[0])
    ham_frequencies = util.get_word_freq(file_lists_by_category[1])

    # Add non-existant words to each dict
    for word in all_words.keys():
        if word not in spam_frequencies.keys():
            spam_frequencies[word] = 0
        if word not in ham_frequencies.keys():
            ham_frequencies[word] = 0

    num_words = len(all_words.keys())

    num_spam_words = sum(spam_frequencies.values())
    num_ham_words = sum(ham_frequencies.values())

    spam_denominator = float(num_spam_words + num_words)
    ham_demoninator = float(num_ham_words + num_words)

    spam_word_probabilities_laplace = {}
    ham_word_probabilities_laplace = {}
    for pair in spam_frequencies.items():
        spam_word_probabilities_laplace[pair[0]] = (pair[1] + 1.0) / spam_denominator
    for pair in ham_frequencies.items():
        ham_word_probabilities_laplace[pair[0]] = (pair[1] + 1.0) / ham_demoninator

    probabilities_by_category = [
        spam_word_probabilities_laplace,
        ham_word_probabilities_laplace
    ]

    return probabilities_by_category

def logSum(gamma1, gamma2):
    maximum = max(gamma1, gamma2)

    return maximum + math.log(math.exp(gamma1 - maximum) + math.exp(gamma2 - maximum))

def classify_new_email(filename,probabilities_by_category,prior_by_category, gamma=1.0):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified
    probabilities_by_category: output of function learn_distributions
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the 
    parameter in the prior class distribution
    gamma: Modifies the decision boundary weight. Higher gamma=high chance of 
    HAM. Lower gamma=higher chance of SPAM.

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the 
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)], 
    representing the log posterior probabilities
    """
    ### TODO: Write your code here
    
    # Get the word count vector
    x = util.get_word_freq([filename])

    # Common for both classes, P(X)
    log_p_spam = math.log(prior_by_category[0])
    log_p_ham = math.log(prior_by_category[1])

    for word_pair in x.items():
        if word_pair[0] not in probabilities_by_category[0].keys():
            continue
        log_p_spam += word_pair[1] * math.log(probabilities_by_category[0][word_pair[0]])
        log_p_ham += word_pair[1] * math.log(probabilities_by_category[1][word_pair[0]])

    beta = logSum(log_p_spam, log_p_ham)

    log_p_spam -= beta
    log_p_ham -= beta

    # print((log_p_spam, log_p_ham))

    result_str = ""
    if log_p_spam > log_p_ham + math.log(gamma):
        result_str = "spam"
    else:
        result_str = "ham"

    classify_result = (result_str, [log_p_spam, log_p_ham])
    return classify_result

if __name__ == '__main__':
    
    # folder for training and testing 
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"

    # generate the file lists for training
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
        
    # Learn the distributions    
    probabilities_by_category = learn_distributions(file_lists)
    
    # prior class distribution
    priors_by_category = [0.5, 0.5]
    
    # Store the classification results
    performance_measures = np.zeros([2,2])
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam' 
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham' 
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam' 
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham' 

    # Classify emails from testing set and measure the performance
    for filename in (util.get_files_in_folder(test_folder)):
        # Classify
        label,log_posterior = classify_new_email(filename,
                                                 probabilities_by_category,
                                                 priors_by_category)
        
        # Measure performance (the filename indicates the true label)
        base = os.path.basename(filename)
        true_index = ('ham' in base) 
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],totals[0],correct[1],totals[1]))
    
    
    ### TODO: Write your code here to modify the decision rule such that
    ### Type 1 and Type 2 errors can be traded off, plot the trade-off curve

    errors = []
    for gamma in np.logspace(-2.5, 22):
        performance_measures = np.zeros([2,2])
        for filename in (util.get_files_in_folder(test_folder)):
            # Classify
            label,log_posterior = classify_new_email(filename,
                                                     probabilities_by_category,
                                                     priors_by_category,
                                                     gamma)
            
            # Measure performance (the filename indicates the true label)
            base = os.path.basename(filename)

            true_index = ('ham' in base) 
            guessed_index = (label == 'ham')
            performance_measures[int(true_index), int(guessed_index)] += 1

        correct = np.diag(performance_measures)
        # totals are obtained by summing across guessed labels
        totals = np.sum(performance_measures, 1)
        errors.append([totals[0] - correct[0], totals[1] - correct[1]])

    plt.plot([x[0] for x in errors], [x[1] for x in errors], marker="o")
    plt.xlabel("Type 1 Errors (Incorrectly identified as HAM)")
    plt.ylabel("Type 2 Errors (Incorrectly identified as SPAM")
    plt.margins(1, tight=False)
    plt.show()




 