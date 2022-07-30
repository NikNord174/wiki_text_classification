# Text classification based on Wikipedia corpus

## Description

The project is aimed to classify Wikipedia articles by category such as science, art, ingeneering etc.

## Getting started
The project is made using Python 3.8.9 Other necessary packajes are noticed in requirements.txt.

To run the code, firstly, clone the repository from Github and switch to the new directory:

    $ git clone git@github.com/USERNAME/wiki_text_classification.git
    $ cd wiki_text_classification

Secondly, create a new environment using

    $ python3 -m venv venv

Further activate the environment by

    $ python3 source venv/bin/activate

and instal all requirements using

    $ pip3 install -r requirements.txt

To start learning type

    $ python3 main.py -i

In case you need to specify path to data, run

    $ python3 main.py -i datapath

It is anticipated that data is stored in .json (or .jsonl) file. In other cases, you need to change a little initial code in the *data_import* function in *data_preprocessing.py*
