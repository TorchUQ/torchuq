
## Foreword

In high stakes applications, people want predictions they can trust. An important aspect of trust is accurate uncertainty quantification --- a prediction should contain accurate information about what it knows, and what it does not know. 

Uncertainty quantification is a very active area of research, with hundreds of new publications each year in the machine learning community alone. This tutorial aims to cover both classical methods for uncertainty quantification (such as scoring rules, calibration, conformal inference) and as well as some recent developments (such as multi-calibration, decision calibration, advanced conformal methods). This tutorial will also focus on how to use the torchuq software package to easily implement these methods with a few lines of code. Specifically, we will go through the following topics: 

1. how to represent uncertainty, and how to learn different uncertainty predictions from data.
2. how to evaluate and visualize the quality of uncertainty predictions, such as calibration, coverage, scoring rules, etc. 
3. how to obtain calibrated probability with the general framework of conformal prediction/calibration.
4. how to measure group calibration or multi-calibration and how to implement algorithms to achieve them.

We hope this tutorial will give you an overview of the landscape in the frequentist uncertainty quantification paradigm. This tutorial does not cover other paradigms such as Bayesian methods, belief theory, betting theory, etc.

**Background** This tutorial aims to be as self contained as possible and will introduce the key concepts as we go. Required backgrounds include undergraduate level understanding of machine learning / statistics, and familiarity with Pytorch (if you have not used Pytorch before I would recommend first going through the [basic tutorial](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)). 


## Outline of the Tutorial

The tutorial consists of the following topics. You can click the link to access each tutorial. All tutorials are made with jupyter, so you can also download the repo to run the tutorial interactively. 

1. Regression
    a. Representing and evaluating predictions
    b. Learning predictors with proper scoring rules
    c. Conformal inference and calibration
    
2. Classification 
    a. Representing and evaluating predictions
    b. Calibration and conformal inference
    c. Zoo of calibration definitions
    
3. Advanced topics
    a. Multicalibration
    b. Decision calibration 
    