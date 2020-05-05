# Neuroevolution_Forex_Prediction

### In order to run:
1. Install Anaconda for Windows with Python 3.7 (https://www.anaconda.com/products/individual)
2. Open Anaconda Prompt
#### Steps 3-8 (written):
3. Create an Anaconda environment                                                     'conda create --name pyesn_environment'
4. Activate the Anaconda environment                                                  'conda activate pyesn_environment'
5. Install pip in the environment                                                     'conda install pip'
6. Install pandas, numpy, plotly, and cufflinks in the environment using pip          'pip install numpy pandas plotly cufflinks'
7. Change directory to the one containing source files                                'cd [path to source code]'
8. Execute main.py to run program. See below for controlling program parameters       'python main.py'

#### Steps 3-8 (video):
<a href="http://www.youtube.com/watch?feature=player_embedded&v=ZGfAD9ShAr0
" target="_blank"><img src="http://img.youtube.com/vi/ZGfAD9ShAr0" 
alt="YOUTUBE CONDA TUTORIAL" width="240" height="180" border="10" /></a>

#### Step 9 (optional - if saving .txt is required). 
To save the output to a .txt file, simply add '> output.txt' to the end of the program execution command. The output will be stored in the source directory. Execute program like so: 'python main.py > output.txt'

This redirects the output to the text file, so none will be visible in Anaconda prompt; one solution to this is running the program using PyCharm, with the correct configuration. The brief video below shows how to do this.

<a href="http://www.youtube.com/watch?feature=player_embedded&v=ad0F0RDyxpg
" target="_blank"><img src="http://img.youtube.com/vi/ad0F0RDyxpg" 
alt="YOUTUBE PYCHARM TUTORIAL" width="240" height="180" border="10" /></a>
________________________________________________________________________________________________________________________________________

### How to use:
Once the above steps are complete, user interaction can take place within the 'parameters.py' file in the source directory. Here, users have access to a variety of important training and testing settings. Outlined below is each of these and their relevance to the application.

* Predictions fitting
  * prediction_window: The number of days that the network must predict  
  * n_optimizer_predictions: The number of predictions to base average mean-squared error on when testing training windows
  * smallest_possible_training_window: Smallest training window to test during window optimization
  * largest_possible_training_window: Largest training window to test during window optimization
  * training_window_test_increase: The number to increase by between window tests
  * n_fitness_predictions: The number of predictions to base average mean-squared error on when getting fitness during neuroevolution
  * n_test_predictions: More will give better representation of effectiveness during the final testing of standard/evolved Echo State Network
* Echo State Network (ESN)
  * n_reservoir: Size of ESN hidden state
  * spectral_radius: Spectral radius to use during weight initialization in the ESN hidden state
  * sparsity: Sparsity to use during weight initialization in the ESN hidden state
  * noise: Noise to use during fitting of ESN to target data
  * input_scaling: Scale of the input to hidden weights within the ESN
* Neuroevolution
  * n_generations: Number of generations to run genetic algorithm for
  * n_population: Number of members in population 
  * mutation_rate: Mutation rate by which to mutate Echo State Networks
  * crossover_rate: Multiply by 100 to give the percentage of next population which will be based on current population's best member
*
  * training_html_auto_show: If true display training plotly htmls upon creation
  * testing_html_auto_show: If true display testing plotly htmls upon creation
  * ohlc_html_auto_show: If true display OHLC html upon creation
  * window_html_auto_show: If true display window optimization html upon creation
  * results_html_auto_show: If true display results html upon creation
  * x_range_factor: Plot x-range will be (training_window x this value)
  * max_test_line_width: The maximum line width which
  * test_line_width_factor: This factor works with mean-squared error to set line width - refer to source code for the equation
* n_iterations: How many tests to run
________________________________________________________________________________________________________________________________________
Echo State Network created with PyESN; available at https://github.com/cknd/pyESN/blob/master/license.md

Copyright (c) 2015 Clemens Korndörfer

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
________________________________________________________________________________________________________________________________________
