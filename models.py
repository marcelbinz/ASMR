model_string_ew = '''
NUM_PARAMETERS = 1

def model(parameters, option_A, option_B):
    """
    Compute the probability of choosing Option B over Option A.

    Parameters
    ----------
    parameters : np.ndarray of shape (num_parameters,)
        Model parameters.

    option_A : np.ndarray of shape (num_trials, num_features)
        Feature matrix for Option A across trials.

    option_B : np.ndarray of shape (num_trials, num_features)
        Feature matrix for Option B across trials.

    Returns
    -------
    choice_probability_B : np.ndarray of shape (num_trials,)
        The predicted probability of choosing Option B on each trial.
    """

    # compute weighted value for each option
    value_A = option_A.sum(-1)
    value_B = option_B.sum(-1)

    # compute scaled difference in value
    scale_value_difference = parameters[0] * (value_B - value_A)

    # apply logistic function to obtain choice probabilities
    choice_probability_B = 1.0 / (1.0 + np.exp(-scale_value_difference))

    # clip probabilities to avoid numerical issues
    choice_probability_B = np.clip(choice_probability_B, 0.00001, 1 - 0.00001)

    return choice_probability_B'''

model_string_wadd = '''
NUM_PARAMETERS = 1

def model(parameters, option_A, option_B):
    """
    Compute the probability of choosing Option B over Option A.

    Parameters
    ----------
    parameters : np.ndarray of shape (num_parameters,)
        Model parameters.

    option_A : np.ndarray of shape (num_trials, num_features)
        Feature matrix for Option A across trials.

    option_B : np.ndarray of shape (num_trials, num_features)
        Feature matrix for Option B across trials.

    Returns
    -------
    choice_probability_B : np.ndarray of shape (num_trials,)
        The predicted probability of choosing Option B on each trial.
    """

    # define feature validity weights (importance of each feature)
    validities = np.array([0.9, 0.8, 0.7, 0.6])
    
    # compute weighted value for each option
    value_A = option_A @ validities
    value_B = option_B @ validities

    # compute scaled difference in value
    scale_value_difference = parameters[0] * (value_B - value_A)

    # apply logistic function to obtain choice probabilities
    choice_probability_B = 1.0 / (1.0 + np.exp(-scale_value_difference))

    # clip probabilities to avoid numerical issues
    choice_probability_B = np.clip(choice_probability_B, 0.00001, 1 - 0.00001)

    return choice_probability_B'''

model_string_ttb = '''
NUM_PARAMETERS = 1

def model(parameters, option_A, option_B):
    """
    Compute the probability of choosing Option B over Option A.

    Parameters
    ----------
    parameters : np.ndarray of shape (num_parameters,)
        Model parameters.

    option_A : np.ndarray of shape (num_trials, num_features)
        Feature matrix for Option A across trials.

    option_B : np.ndarray of shape (num_trials, num_features)
        Feature matrix for Option B across trials.

    Returns
    -------
    choice_probability_B : np.ndarray of shape (num_trials,)
        The predicted probability of choosing Option B on each trial.
    """

    # define feature validity weights (importance of each feature)
    validities = np.array([1.0, 0.5, 0.25, 0.125])
    
    # compute weighted value for each option
    value_A = option_A @ validities
    value_B = option_B @ validities

    # compute scaled difference in value
    scale_value_difference = parameters[0] * (value_B - value_A)

    # apply logistic function to obtain choice probabilities
    choice_probability_B = 1.0 / (1.0 + np.exp(-scale_value_difference))

    # clip probabilities to avoid numerical issues
    choice_probability_B = np.clip(choice_probability_B, 0.00001, 1 - 0.00001)

    return choice_probability_B'''

model_string_gecco = '''
NUM_PARAMETERS = 2

def model(parameters, optionA, optionB):
    """
    Input:
        choices - participant choices for all trials (numpy array)
        optionAs - four expert ratings for option A for all trials (numpy array)
        optionBs - four expert ratings for option B for all trials (numpy array)
        parameters - list of parameters (inverse temperature, discount factor)

    Output:
        negative log likelihood - negative log likelihood of choices conditioned on model parameters
    """
    temperature, discount_factor = parameters
    validities = np.array([0.9, 0.8, 0.7, 0.6])
    log_likelihood = 0
    value_A = np.sum(np.array(option_A) * np.array(validities) * (discount_factor ** np.arange(len(validities))), axis=-1)
    value_B = np.sum(np.array(option_B) * np.array(validities) * (discount_factor ** np.arange(len(validities))), axis=-1)
    scale_value_difference = temperature * (value_B - value_A)
    choice_probability_B = 1.0 / (1.0 + np.exp(-scale_value_difference))

    # Clip probabilities to avoid numerical issues
    choice_probability_B = np.clip(choice_probability_B, 0.00001, 1 - 0.00001)

    return choice_probability_B'''