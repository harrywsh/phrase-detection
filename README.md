# Automated Domain-based Keyword Detection :zap: :rocket:
Keyword detection for a particular domain, based on the [PRDualRank](https://dl.acm.org/doi/10.1145/1935826.1935933) framework! 

## Contributors
Work done during Fall 2019, Spring 2020, Summer 2020 at [FORWARD Lab](http://www.forwarddatalab.org/) @ [UIUC](https://cs.illinois.edu/), by Dipro Ray and Shuhan Wang.

## Maintainers
Dipro Ray (dipror2@illinois.edu)

## Where do I find X?
* All source code files lie within `.py` files in the `./final_stuff/` directory.
* Input data, and output data are stored in `./final_stuff/data/` and `./final_stuff/outputs/` respectively.
* [Only for repo maintainers] In-development code can be found in `./development_*/` directories.
* [Only for repo maintainers] Archives can be found in the `./old/` subdirectory of each folder.
* [Only for repo maintainers] Metrics and associated scripts and ipynbs lie in the `./metrics/` directory.

## How do I run the framework?
If you'd like to use spaCy's GPU capability, make sure you have access to a GPU. (For Nvidia users, use `nvidia-smi` to check GPU info.) Then, install CUDA Toolkit 10.2 (**note the version!** spaCy/cupy isn't compatible with the latest 11.0 version yet, as far as I know.) Use `nvcc --version` to ensure CUDA drivers (as well as the correct version) have been installed. Then, try:
```
~$ python3
Python 3.7.4 (default, Aug 13 2019, 20:35:49)
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import spacy
>>> spacy.require_gpu()
True
```
If the last line is "True" and returns no errors, you're good to go with respect to spaCy GPU compatibility. Before proceeding onto the next steps, execute `python -m spacy download en_core_web_sm`.
1. `cd` into the repository directory and run `pip install -r requirements.txt`
2. `cd` into the `final_stuff` subdirectory. This will be the main working directory henceforth.
3. `final_framework_v6.py` is the script to be run to execute the latest framework. (FYI: It uses helper scripts: `prdualrank.py`, `wikiscore.py`, `extractor_helpers.py`.)
4. Place your input data in `./data/` directory. Make sure to sanitize the input! Lower case all text, and remove all non-alphanumeric text except periods.
5. An example command to run the script is: `python3 final_framework_v6.py arxiv_titles_and_abstracts_short.txt 350 750 test_run.txt 9`
      1. `arxiv_titles_and_abstracts_short.txt`: This parameters indicates that the input data file is `./data/arxiv_titles_and_abstracts_short.txt`
      2. 350: The number of patterns to be extracted in each iteration
      3. 750: The number of keywords to be extracted in each iteration
      4. `test_run.txt`: This means that the output will be stored in `./outputs/test_run.txt`
      5. 9: This refers to the scoring method (look in `final_framework_v6.py` for details). For now, this parameter is always set to 9.
      6. An extra parameter `iter_num` exists, but it needs to be set within the source code of `final_framework_v6.py`. It affects the number of iterations to be run.
6. As mentioned earlier, your results will be stored in the designated output file.
7. Push your results, with a meaningful file name <ins>that indicates date, time, framework version, scoring method, corpus</ins> (and ensure it's in the outputs directory).
