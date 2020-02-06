# Automated Domain-based Keyword Detection :zap:
Keyword detection for a particular domain, powered by the [PRDualRank](https://dl.acm.org/doi/10.1145/1935826.1935933) framework! 

## Contributors
Work done during Fall 2019 & Spring 2020 at [FORWARD Lab](http://www.forwarddatalab.org/) @ [UIUC](https://cs.illinois.edu/), by:
* Dipro Ray (dipror2@illinois.edu)
* Shuhan Wang (shuhanw2@illinois.edu)

## How do I run the framework?
* Install the requirements outlined in `requirements.txt` with `pip install -r requirements.txt`. You might also have to run `python -m spacy download en_core_web_md`. 
* Then, run `python3 ./final_stuff/final_framework.py [input_file] [max # of patterns] [max # of keywords] [output_file]`. (Note that `input_file` lives within `./final_stuff/data/`, and `output_file` lives within `./final_stuff/outputs/` respectively.)

## Where do I find X?
* Source code lies within `.py` files in the `./final_stuff/` directory.
* Input data, and output data are stored in `./final_stuff/data/` and `./final_stuff/outputs/` respectively.
* In-development code can be found in `./development_*/` directories.
* Archives can be found in the `./old/` subdirectory of each folder.
* Metrics and associated scripts and ipynbs lie in the `./metrics/` directory.
