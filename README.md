## For the original publication, please refer to this one and citation informatio at the bottom. 

[**Deep Neural Networks as Gaussian Processes**](https://arxiv.org/abs/1711.00165)

## Some modification in Code
1. Code for loading the CIFAR-10 dataset was provided in the uncertainty.py file. However, the CIFAR-10 dataset is also required by run_experiments.py. Instead of duplicating the same code in both uncertainty.py and run_experiments.py, the dataset-loading code was moved to load_dataset.py so that both uncertainty.py and run_experiments.py can load the CIFAR-10 dataset using a single shared implementation (lines 123–171). After this modification, the duplicated code in uncertainty.py was removed.

2. Because load_dataset.py was modified, both uncertainty.py (lines 309–316) and run_experiments.py (lines 133–140) were also modified so that they could properly import and use the functionality provided by load_dataset.py.

3. To plot the graphs using the same x-axis and y-axis ranges as those in Figure 8 of the original paper, the xlim and ylim values were hard-coded in uncertainty_plot.py (lines 271–287).

## Usage (examples)

To run `run_experiments.py`,

```
python run_experiments.py \
       --dataset=mnist\                 # for mnist data set. Use cifar10 for cifar10 dataset
       --num_train=100 \                # num_train used is 100 - 5000
       --num_eval=10000 \               # this has been fixed
       --hparams='nonlinearity=relu, depth=100,weight_var=1.79,bias_var=0.83' \      # This can be adjusted 
```


To run `uncertainty.py`,

```
python uncertainty_plot.py \
       --dataset=mnist\                 # for mnist data set. Use cifar10 for cifar10 dataset
       --num_train=5000 \                # num_train used is 100 - 5000
       --num_eval=10000 \               # this has been fixed
       --nonlinearities='tanh, relu'    # this has been fixed
       --hparams='depth=100,weight_var=1.79,bias_var=0.83' \                  # This can be adjusted.
       --output_file='/nngp/output/Figure8/uncertainty_fig8_mnist_5k.png'     # This can be changed, depending on the inputs parameters.
```


## Project 2
1. Predictive Uncertainty vs. Prediction Error
Reproduce the paper's finding (Figure 8) that the GP's predictive variance correlates with its actual squared prediction error. The original instruction was to reproduce Figure 3. However, my computing system encountered internal memory issues and terminated the jobs. Therefore, smaller training set sizes were used.

Some of the original hyperparameters did not reproduce the results exactly. For those cases, the hyperparameters were modified to generate the same or similar results. The main conclusion of Figures 3 and 8 is that prediction uncertainty is highly correlated with the empirical error on test points. My plots show the same trend, even though the correlations are not identical to those reported in the original paper.

The resulting figures and the side-by-side comparison figure can be found in /output/Figure8/.

2. Reproduction of Table 2 with a Smaller Training Set for NNGP Only
Some of the original hyperparameters did not reproduce the results exactly. For those cases, the hyperparameters were modified to generate the same or similar results in terms of accuracy.

The resulting table can be found in /output/Table2/.

3. A Unique Extension
3.1 Motivation
Most machine learning models can experience underfitting or overfitting before the final model is determined. NNGPs may also exhibit these issues. One way to address them is to adjust the complexity of the model.

For neural networks, model complexity can be adjusted by changing the width and depth of the hidden layers. Since the width of an NNGP is unlimited, changing its depth is a possible way to adjust its complexity. Therefore, this unique extension experiment investigates the effect of NNGP depth on the performance of the models and examines whether different depths are associated with underfitting or overfitting behavior.

3.2 Methods
The weight variance and bias variance were fixed, and various network depths were tested with different training set sizes to determine whether there was a trend in the depth that produced the best performance (accuracy and MSE). For this experiment, mnist and tanh were used.

Three different sets of weight variance and bias variance were tested to determine whether the observed trend was consistent across different hyperparameter settings.

33.3 Results
There are two main findings.
First, with fixed values of the variance parameters, accuracy initially increases as depth increases. Accuracy then reaches a maximum, followed by a decrease as depth continues to increase. For example, with a training set size of 400, a weight variance of 4.0, and a bias variance of 0.84, the maximum accuracy occurs at a depth of 6. When the depth is less than 6, accuracy increases as depth increases. When the depth is greater than 6, accuracy decreases as depth increases. This trend is observed across all training set sizes used in the experiment and for all combinations of weight variance and bias variance that were tested. Second, the depth at which maximum accuracy (or minimum MSE) occurs tends to become smaller as the training set size increases for all of the weight variance and bias variance combinations tested. The exception is the combination of a low weight variance (3.0) and bias variance (0.8), for which the optimal depth shows some fluctuation. Nevertheless, the overall trend is similar.

The results can be found in /output/Unique_extension/

3.4 Discussion and Conclusion
I was able to reproduce Figure 8 and Table 2. Initially, the same hyperparameters as those used in the original paper were used. The exact same results were reproduced in some cases. However, the hyperparameters had to be adjusted in other cases in order to reproduce the same or similar results. The reason why the original hyperparameters did not produce the same results in all cases is still unclear.

For the unique extension of the project, the effect of the depth of the hidden layers of the NNGP was investigated. The results suggest that there is a depth at which model performance is maximized, and that this optimal depth tends to decrease as the training set size increases.

When the depth is below this optimal point, the model may have insufficient capacity to capture the patterns in the data, which could be associated with underfitting. When the depth is above this point, the decrease in performance could be associated with overfitting. In other words, the results suggest that using too few hidden layers may limit the model's ability to learn complex patterns, while using too many hidden layers may reduce its ability to generalize.

Therefore, finding an appropriate depth is important for maximizing model performance.

3.5 Future Study
The initial plan for the unique extension of Project 2 was to investigate different activation functions, such as sigmoid and softmax. However, these activation functions are typically used in output layers. It would also be interesting to investigate how these activation functions behave when used in the hidden layers of an NNGP. In addition, the authors of the original paper suggested investigating the use of dropout as a direction for future study. Therefore, investigating different activation functions in hidden layers and the effect of dropout could be worthwhile directions for future research.

## Declaration
AI was used to check and improve the grammar and expression of this manuscript, as the author is a non-native English speaker and is still developing proficiency in English. However, all ideas, study plans, analyses, and the original manuscript were developed and written by the author. 


## Contact
Myeongseon Lee, myeongseon.lee@colorado.edu

## Citation
```
  @article{
    lee2018deep,
    title={Deep Neural Networks as Gaussian Processes},
    author={Jaehoon Lee, Yasaman Bahri, Roman Novak, Sam Schoenholz, Jeffrey Pennington, Jascha Sohl-dickstein},
    journal={International Conference on Learning Representations},
    year={2018},
    url={https://openreview.net/forum?id=B1EA-M-0Z},
  }
```

## Note

This is not an official Google product.
