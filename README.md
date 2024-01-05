# Gaussian Max-Value Entropy Search for Multi-Agent Bayesian Optimization

---
This repogistory is the implementation of 
[_**Gaussian Max-Value Entropy Search for Multi-Agent Bayesian Optimization**_](https://arxiv.org/abs/2303.05694)
accepted to **_[IROS 2023]_**.

The code is modified from [Distributed Bayesian Optimization for Multi-Agent Systems](https://github.com/FilipKlaesson/dbo)
created by [Filip Klaesson](https://github.com/FilipKlaesson). For the baselines, [the original documentation](https://github.com/FilipKlaesson/dbo/blob/master/README.md)
has a more clear explaination about the code. The updates are very similar to the original code. If you have 
specific questions regarding my updates of the code, feel free to open an issue or email me.

---

**dbo** is a compact python package for bayesian optimization.
It utilizes scikit-learn and gpytorch to model the surrogate function and offers 
multiple strategies to select queries.
In addition to the vanilla bayesian optimization algorithm, **dbo** offers:

* Distributed and parallel optimization
* The proposed Gaussian max-value entropy search algorithm in our paper.
* Stochastic policy evaluations
* Expected acquisition policy over pending queries
* Internal acquisition function regularization
* Regret analysis outputs

---
# Installation

This setup guide is for **Debian**-like environments. Note that if you would like to use GPU acceleartion for pytorch,
please fist refer to the official documentation to install pytorch and then install dependencies in the requirements.txt.

```bash
git clone https://github.com/mahaitongdae/dbo.git && cd dbo
pip install -r requirements.txt
python setup.py install     # use 'python setup.py develop' for development
```


---
# Running experiments

```bash
cd examples
python main_2d_json.py
```
OR for batch experiments
```bash
cd examples
chmod +x experiments.sh
./experiments.sh
```


### Output

**dbo** will create a **results** folder in the same directory as **\_\_main\_\_**. The output
(generated data/plots/gifs) will be stored in the **results** folder keyed with date and time.
For example, by running **dbo** with **\_\_main\_\_** in the **dbo** project folder, the
directory will look like this:

```
dbo
└───src  
└───examples
└───results
    └───$TEST_FUNCTION_NAME
        └───$ALGO_NAMEYYYY-MM-DD_HH:MM:SS
            └───data
            └───fig
                └───png
                └───pdf
                └───gif
```

Running <b>optimize()</b> will generate the following output:

* regret

    * File with mean regret over the n_runs together with the 95% confidence bound error (.csv)
      * Note: The 95% confidence bound is the symmetric bound on the **linear** scale.

    * Plots of mean regret together with 95% confidence bounds (.png/.pdf)

* bo*

    * Plots of every iteration in the optimization algorithm (.png/.pdf)

    * Gif of the progress in the optimization algorithm      (.gif)

Plots and gifs (except regret plot) are disabled when n_runs > 1.

 ---

# Contributing

Free free to contribute via issues or pull requests.

# Credits

Thank [Filip Klaesson](https://github.com/FilipKlaesson) for the original implementations.

