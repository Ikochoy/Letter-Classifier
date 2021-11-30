** Folder Final Structure **
assignment2/
├── DataResults        # a directory that contains all the returned dataframes
├── splitted_imagesets # a directory that output images splitted into training, validation and testing
        ├── train      # a directory that contains all the training images
        ├── validation # a directory that contains all the validation images
        ├── test       # a directory that contains all the testing images
├── nonMNIST_small     # an unzipped directory of the nonMNIST_small datasets
├── nonMNIST_small.tar.gz # compressed nonMNIST_small.tar.gz datasets
├── models             # a directory that holds all the saved models
├── a2_functions.py  # python file that contains all the codes
├── requirements.txt # a txt file that contains all the packages needed
├── README.txt       # this txt file
├── report.pdf       # the report pdf file with all my solutions

a2_functions.py -- contains all the code for each implementation task (q1 to q6)

** How to run my code **

Suggested steps for running my functions:

1) Make sure you cd into the assignment2 directory

2) Create a virtual environment using the command below. Note that python
version that I am using is python3.8
```
conda create -n name python=3.8
```

3) Before running the code, activate your environment and make sure that the environment contains all the necessary packages used for computations and visualization.
Install all libraries/packages with the provided `requirements.txt`
```
pip install -r requirements.txt
```

4) To regenerate results for each of the implementation tasks, please see below.
Before running any task, make sure to
1) Unzip the notMNIST_small.tar.gz file
2) uncomment line 551 in a2_functions.py first.

For Task 1:
1) To split the images from raw nonMNIST_small directory into training, validation
and testing images, only uncomment line 557 in a2_functions.py.
2) To get the trainingloader, validationloader, and testloader, only uncomment line 560.
Then, after uncommenting out necessary codes, we can run the file by using
```
python3 a2_functions.py
```
For Task 2:
** Make sure the codes used to split the folders (line 557) is commented out **
1) Only uncomment out line 573 and line 574 in a2_functions.py.
2) Then, after uncommenting out necessary codes, we can run the file by using
```
python3 a2_functions.py
```
For Task 3:
** Make sure the codes used to split the folders (line 557) is commented out **
1) Uncomment out line 578 and line 579 in a2_functions.py.
2) Then, after uncommenting out necessary codes, we can run the file by using
```
python3 a2_functions.py
```
For Task 4:
** Make sure the codes used to split the folders (line 557) is commented out **
1) Uncomment out line 583 and 584 in a2_functions.py.
2) Then, after uncommenting out necessary codes, we can run the file by using
```
python3 a2_functions.py
```
For Task 5:
** Make sure the codes used to split the folders (line 557) is commented out **
1) Uncomment out line 588 and 589 in a2_functions.py.
2) Then, after uncommenting out necessary codes, we can run the file by using
```
python3 a2_functions.py
```

** Report.pdf **
All the visualization results in the report.pdf can be found in the output
directory

