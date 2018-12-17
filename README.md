# Capstone Project
Machine Learning Fundamentals (Codecademy)

This project makes use of most of the machine learning models taught in the course. Using the okcupid dataset, this code uses scikit-learn to analyse user essays and try to guess if the user is male or female.

## Presentation
The presentation (capstone_project_presentation.odp) was created with LibreOffice Impress. A PDF version is provided in case the .odp file doesn't open/render correctly on your computer.

## Running the Code

There is a Jupyter notebook file, and .py files provided. The Jupyter notebook file was what I used during development. The .py files are more stripped down, and don't have accompanying markdown text. You can go with either, but I would recommend opening the notebook.

After cloning this repository to your local machine, if you want to run the code, you will need to copy the okcupid dataset (profiles.csv file). This file is over 150 MB and is not provided in this repository.

The profiles.csv file can be obtained by extracting [this .zip file](https://s3.amazonaws.com/codecademy-content/programs/machine-learning/capstone/capstone_starter.zip).

The file "common.py" is not intended to be run directly. It is used by the other .py files.

Example: Running the Naive Bayes model:
`python3 naive_bayes.py`
