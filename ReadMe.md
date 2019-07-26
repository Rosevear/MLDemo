This repos is a simple example of a basic ML experiment.
The original source code was taken from https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/, Jason Brownlee, on Friday July 26, 2019, and modified for its present purposes.

#########Setting up an ML development environment with Anaconda

Anaconda is both a package manager and a virtual environment manager. The former aspect allows for quick and easy installsupdating/management of hundreds of modules withi nthe python ecosystem.

Virtual environments serve as a way to isolate dependencies required for different projects, rather than installing everything globally and risking conflicts due to differing requirements between projects.

1. Download the Anaconda distribution (ideally installing for all users of the system) from https://www.anaconda.com/distribution/

2. Open the anaconda prompt (which should be installed by the anaconda distribution) and type as below to find where conda is being called from. 
Add the evealed path for the .exe file and Scripts folder to your PATH environment variable (if you want to be able to call conda from the terminal)
where conda
Result for my case: C:\ProgramData\Anaconda3\Scripts\conda.exe

So, for example, I would add C:\ProgramData\Anaconda3\Scripts\ and  C:\ProgramData\Anaconda3\Scripts\conda.exe to my windows PATH variable

3.  In order to use conda with powershell specifically one must must initialize the shell, and then restart it
conda init powershell

Other shells likely need to perform something similar, and attempting to use conda from a gien shell should result in a prompt to initialize for your respective shell

4. Create a new virtual environment with a python version of 3.6 (keras compatible version of python), to install dependencies for the current project

conda create --name <you_env_name> python=3.6

5. To activate the environment
conda activate <your_env_name>

6. To install  tensorflow: Tensorflow is an open source machine learning platform https://www.tensorflow.org/
conda install tensorflow 

7. Verify installation was successfull
python -c "import tensorflow as tf; print(tf.__version__)"

8. To install keras: Keras is a high level Deep Learning API that supports multiple backends (like Tensorflow) for performing the actual computations used in learning: https://keras.io/.
conda install keras

9. Verify keras installation
python -c "import keras; print(keras.__version__)"

10. Install sci-kit learn: scikit-learn is a traditional machine learning library, that also features a number of helpful functions for running ML experiments.: https://scikit-learn.org/stable/
conda install -c anaconda scikit-learn 

11. Validate sklearn installed
python -c "import sklearn; print(sklearn.__version__)"

######Runnning the experiment #######

To clone the repository
git clone 