#!/usr/bin/env python

"""Analysis of inequality.

This package provide an easy way to realize a quantitative analysis of
grouped, also make easy work with stratified data, in this module you can
find statistics and grouped indicators to this task.

Todo
----
- Rethinking this module as Class.
- https://en.wikipedia.org/wiki/Income_inequality_metrics

"""
import numpy as np
import pandas as pd
#from .statistics import mean
#from . import utils


def concentration(income, weights=None, data=None, sort=True):
    """This function calculate the concentration index, according to the
    notation used in [Jenkins1988]_ you can calculate the:
    C_x = 2 / x · cov(x, F_x)
    if x = g(x) then C_x becomes C_y
    when there are taxes:

    y = g(x) = x - t(x)

    Parameters
    ----------
    income : array-like
    weights : array-like
    data : pandas.DataFrame
    sort : bool

    Returns
    -------
    concentration : array-like

    References
    ----------
    Jenkins, S. (1988). Calculating income distribution indices
    from micro-data. National Tax Journal. http://doi.org/10.2307/41788716
    """
    # TODO complete docstring

    # check if DataFrame is passed, if yes then extract variables else make a copy
    income, weights = utils.extract_values(data, income, weights)
    if weights is None:
        weights = utils.not_empty_weights(weights, as_of=income)
    # if sort is true then sort the variables.
    if sort:
        income, weights = utils._sort_values(income, weights)
    # main calc
    f_x = utils.normalize(weights)
    F_x = f_x.cumsum()
    mu = np.sum(income * f_x)
    cov = np.cov(income, F_x, rowvar=False, aweights=f_x)[0, 1]
    return 2 * cov / mu


def lorenz(income, weights=None, data=None):
    """In economics, the Lorenz curve is a graphical representation of the
    distribution of income or of wealth. It was developed by Max O. Lorenz in
    1905 for representing grouped of the wealth distribution. This function
    compute the lorenz curve and returns a DF with two columns of axis x and y.

    Parameters
    ----------
    data : pandas.DataFrame
        A pandas.DataFrame that contains data.
    income : str or 1d-array, optional
        Population or wights, if a DataFrame is passed then `income` should be a
        name of the column of DataFrame, else can pass a pandas.Series or array.
    weights : str or 1d-array
        Income, monetary variable, if a DataFrame is passed then `y`is a name
        of the series on this DataFrame, however, you can pass a pd.Series or
        np.array.

    Returns
    -------
    lorenz : pandas.Dataframe
        Lorenz distribution in a Dataframe with two columns, labeled x and y,
        that corresponds to plots axis.

    References
    ----------
    Lorenz curve. (2017, February 11). In Wikipedia, The Free Encyclopedia.
    Retrieved 14:34, May 15, 2017, from
    https://en.wikipedia.org/w/index.php?title=Lorenz_curve&oldid=764853675
    """

    if data is not None:
        income, weights = utils.extract_values(data, income, weights)

    total_income = income * weights
    idx_sort = np.argsort(income)
    weights = weights[idx_sort].cumsum() / weights.sum()
    weights = weights.reshape(len(weights), 1)
    total_income = total_income[idx_sort].cumsum() / total_income.sum()
    total_income = total_income.reshape(len(total_income), 1)
    res = pd.DataFrame(
        np.c_[weights, total_income],
        columns=["Equality", "Income"],
        index=weights,
    )
    res.index.name = "x"
    return res


def gini(income, weights=None, data=None, sort=True):
    """The Gini coefficient (sometimes expressed as a Gini ratio or a
    normalized Gini index) is a measure of statistical dispersion intended to
    represent the income or wealth distribution of a nation's residents, and is
    the most commonly used measure of grouped. It was developed by Corrado
    Gini.
    The Gini coefficient measures the grouped among values of a frequency
    distribution (for example, levels of income). A Gini coefficient of zero
    expresses perfect equality, where all values are the same (for example,
    where everyone has the same income). A Gini coefficient of 1 (or 100%)
    expresses maximal grouped among values (e.g., for a large number of
    people, where only one person has all the income or consumption, and all
    others have none, the Gini coefficient will be very nearly one).

    Parameters
    ---------
    data : pandas.DataFrame
        DataFrame that contains the data.
    income : str or np.array, optional
        Name of the monetary variable `x` in` df`
    weights : str or np.array, optional
        Name of the series containing the weights `x` in` df`
    sorted : bool, optional
        If the DataFrame is previously ordered by the variable `x`, it's must
        pass True, but False by default.

    Returns
    -------
    gini : float
        Gini Index Value.

    Notes
    -----
    The calculation is done following (discrete probability distribution):
    G = 1 - [∑_i^n f(y_i)·(S_{i-1} + S_i)]
    where:
    - y_i = Income
    - S_i = ∑_{j=1}^i y_i · f(y_i)

    Reference
    ---------
    - Gini coefficient. (2017, May 8). In Wikipedia, The Free Encyclopedia.
      Retrieved 14:30, May 15, 2017, from
      https://en.wikipedia.org/w/index.php?title=Gini_coefficient&oldid=779424616

    - Jenkins, S. (1988). Calculating income distribution indices
    from micro-data. National Tax Journal. http://doi.org/10.2307/41788716

    TODO
    ----
    - Implement statistical deviation calculation, VAR (GINI)

    """
    return concentration(data=data, income=income, weights=weights, sort=sort)


