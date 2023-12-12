# uda_project1

# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Pipeline architecture, data, hyperparameter tuning and classification algorithm:

 **Data** is located at https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv, this is sample bank marketing training data
  
 **Tabular dataset** created from the URL location using **TabularDatasetFactory**. The data cleaning is done using the provided function _clean_data_
  
  Cleaned data is split into training and test sets (**20% test set**) and _random_state_ param is specified for reproducibility
  
**LogisticRegression** is used to train the data with 2 arguments
* *C (Inverse of regularization strength)*: It controls the trade-off between fitting the training data and keeping the model simple to avoid overfitting. Smaller values of C indicate stronger regularization, while larger values indicate weaker regularization. Suitable values for C are typically in the range of 0.01 to 100. Default value was set to 1.
* *max_iter*: It defines the maximum number of iterations for the solver to converge. If the solver fails to converge within the specified number of iterations, it will raise a warning or error. Suitable values for max_iter depend on the complexity of the data and the size of the dataset. Default value was set to 100.

**Parameter sampling** is a technique used in machine learning and optimization to explore the search space of hyperparameters and find the best combination for a given model   

**RandomParameterSampling** was used with the 2 arguments to find the best combination
* *'--C'*: choice(0.01, 0.1, 1.0, 10.0),
* *'--max_iter'*: choice(50, 100, 150, 200)    

By using RandomParameterSampling, the sampler randomly selects values for hyperparameters from the provided sets of choices. This approach allows for a more efficient and flexible exploration of the hyperparameter space, as it does not require evaluating all possible combinations exhaustively. Instead, it randomly selects a subset of combinations to evaluate, which can help save computational resources and time during the hyperparameter optimization process.

The **BanditPolicy** is a type of early termination policy used in Azure Machine Learning for hyperparameter tuning. It determines whether a particular run should be terminated early based on the performance of previous runs.

The BanditPolicy takes two parameters BanditPolicy(evaluation_interval=2, slack_factor=0.1) was used:
* *evaluation_interval*: This parameter specifies the frequency at which the policy should be applied to decide whether to terminate a run. In this case, with evaluation_interval=2, the policy will be evaluated every 2 iterations.
* *slack_factor*: The slack_factor determines the slack allowed with respect to the best performing run. If a run's performance is worse than the best performing run by more than this slack factor, the policy will terminate the run. A smaller slack_factor means a more stringent policy, while a larger slack_factor allows for more slack.

Environment was created for the training run and a **ScriptConfigObject** was used to specify the configuration details of the training job

**HyperDriveConfig** was created using the *ScriptConfigObject* object, Parameter sampler (*RandomParameterSampling*), and policy (*BanditPolicy*).

	
## HyperDrive and AutoML:
Overall, both experiments had comparable accuracy with AutoML being slighty better. AutoML used **VotingEnsemble** model which is a type of ensemble learning technique in which multiple individual models are combined to make predictions. The idea behind a voting ensemble is to leverage the collective wisdom of multiple models to improve overall prediction accuracy and robustness.

### HyperDrive
	Best Run
 	Accuracy = .9088102
 	C = 0.1
  	max_iter = 100
![Hyperdrive Best Run](/Hyperdrive%20Results.jpg)

![Hyperdrive Best Params](/Hyperdrive%20Results%20Best%20Model.jpg)

### AutoML
	Best Model = VotingEnsemble
 	Accuracy = .91794
  ![Auto ML Best Model](/AutoML%20Results%202.jpg)

### Future considerations/ enhancements
* Increase the search space: Expand the range of hyperparameters to explore. Consider adding more options, wider ranges, or different types of hyperparameters to allow for a more comprehensive search.
* Increase the resource allocation: Provide more computational resources, such as CPU cores or memory, to the AutoML experiment. This can help the algorithm explore the search space more thoroughly and speed up the optimization process.
* Adjust the experiment timeout: If the current timeout is too short, consider increasing it to allow for a longer optimization process. However, be mindful of the trade-off between time and resource allocation.
* Include more data: If possible, increase the size of your dataset or include additional relevant data sources. More data can help the AutoML algorithm learn better patterns and make more accurate predictions.
* Iterate and refine: Continuously iterate and refine AutoML experiments. Incorporate the lessons learned from previous experiments, adjust the search space, and experiment with different techniques to improve the performance and efficiency of future runs.
