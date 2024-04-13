# OKT
If you want to run the original OKT code refer to ``` OKT_README ``` to see how to run the file. There is an ``` env.yml ``` that has every dependency to run the code. But quick steps to run it is:
1. Run `` python main_student_model `` on the command line. All parameters can be changed in the `` configs_student_model.yaml `` file.
2. Run `` python main_okt.py `` on the command line. All parameters can be changed in the `` configs_okt.yaml ``.
3. Results are in ``` results/eval_logs.pkl ```.

# SBERT Embeddings
`` sen_trans.py `` contains code that creates new embeddings using different embeddings. Benchmarking against orignal OKT embeddings still TODO.

# Code2Vec
Previous attempt at gettings embeddings for FalconCode (not uploaded).

# FalconCode
The `` notebooks `` folder contains `` make_data.ipynb `` and `` prog2snap.ipynb ``. `` prog2snap.ipynb `` changes the FalconCode dataset to meet the minimum requirements for the Prog2Snap format. `` make_data.ipynb `` changes the FalconCode dataset to match the features of the original OKT dataset (file is still in progress). Make sure to unzip some of the FalconCode data.

# TODO
1. Finish `` sen_trans.py `` to create new embeddings for the original OKT dataset.
2. Benchmark these new embeddings vs. old embeddings.

# Questions (thinking out loud)
1. What's the difference between embeddings and code-embeddings in the original OKT dataset?
2. How are the input tensors created? Are the prompt embeddings hardcoded? Where are they calculated?
3. Can we get rid of the astnn feature and still get the model to work?
