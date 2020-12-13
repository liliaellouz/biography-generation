# Venetian Biographies Generator

## Introduction
In this project, we use generative models to come up with creative biographies of Venetian people that existed before the 20th century. Our motivation was originally to observe how such a model would pick up on underlying relationships between Venetian actors in old centuries as well as their relationships with people in the rest of the world. These underlying relationships might or might not come to light in every generated biography, but we can be sure that the model has the potential to offer fresh perspectives on historical tendencies.

The project can be roughly split into three modules:
 - an interface and webserver, to allow interaction with the model;
 - a generative model, employing a GPT-2 model finetuned to generate venetian biographies, and a BERT model finetuned to act as a realness discriminator preventing lower quality biographies of being outputted;
 - an enhancements module, which applies date and historical figures adjustments to the generated biographies in order to make them more realistic and congurent.
 
## Usage
To start the server, run the command below in the root directory.
```bash
python server.py
```

## Installation
For ease of installation, a `requirements.txt` has been automatically created from the virtual python environment used to develop the modules. To install them, simply run the command below in the root directory.
```bash
pip install -r requirements.txt 

# to install for current user only, run instead
# pip install --user -r requirements.txt 
```

In addition, in order to download the Spacy model required for the biography generation (for the enhancement module in particular), the following command should be run (after installing the libraries in `requirements.txt`):

```bash
python -m spacy download en
```

Besides the code present in this repository, weights for both the GPT-2 and BERT models need to be downloaded. The file `checkpoint.zip` can be downloaded [here](https://drive.google.com/file/d/1_NO47MbZRuySLBoiAE-JXyMY0DTYb1yu/view?usp=sharing) and should be unzipped in the root folder.


## More information
More information about this project can be found in [this wiki page](http://fdh.epfl.ch/index.php/VenBioGen).
