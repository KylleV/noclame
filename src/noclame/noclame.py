"""
NoClaMe
-------

This script implements a series of metrics for node-classification.
These metrics are for binary node-classification, intended to work 
for node-classification on molecular graphs, which usually involves 
a large number of relatively small graphs (<50 vertices). 

Implemented node-classification metrics are : 

- Top-K Metric
- Node-Rank-Descent (NRD)
- Adjusted Node Proximity Index (ANPI)
- Relative Radial Confidence Score (RRCS)

Please note that NRD, ANPI and RRCS are all new metrics introduced here,
while Top-K is a previously established metric. For details on each
metric please see the readme.md file associated with the project.

Author : Victor Kyllesbech (Vrije Universiteit Amsterdam)
Github : https://github.com/KylleV/Bin-Node-Metric

Please cite 'Can graph neural networks understand chemical elements ?'
if you utilize the new metrics introduced here. 
#! update citation after publication
"""

#───────────────────────────────IMPORT───────────────────────────────#

import numpy as np
import math
from collections.abc import Iterable
from functools import wraps

# Typing
from typing import Any, Callable
from numpy.typing import NDArray

#──────────────────────────HELPER FUNCTION───────────────────────────#

def _get_iter_depth (iterable) : 
    """
    This function finds the nesting depth of a given iterable, so a
    noniterable is 0, an iterable is 1, an iterable of iterables is 2,
    and so on. Note that this function identifies the maximum depth of
    nested iterables. 


    Parameters
    ----------
    iterable : any
        Iterable to get depths of

    Returns
    -------
    int
        depth of iterable
    """

    if not isinstance(iterable, Iterable) : 
        return 0 
    
    depths = [_get_iter_depth(item) for item in iterable]
    if not depths:
        return 1
    
    return 1 + max(depths)


def _ensure_list_of_arr(args_to_convert, expected_depth = 1):
    """

    This decorator is intended to ensure the correct input format for the
    different metrics. These metrics can be used on either a single item
    or an iterable of items. This function simply checks if the input is
    a single item, and if so converts it to a list of that single item. 

    The argument names to check are listed in ``args_to_convert`` as
    strings. The expected depth of a single input is set in
    ``expected_depth``, so if set to 1 then a single input is an iterable, if
    set to 2 then a single input is an iterable of iterables. The output
    will have one more nesting level than ``expected_depth``. 
    ``expected_depth`` can be set as an integer, or a list for each item in
    ``args_to_convert``.

    Parameters
    ----------
    args_to_convert : list of str
        list of names of input arguments to analyze

    expected_depth : int or list of int, optional
        a single items nesting depth, by default 1
    """

    def conv_arg (arg_val, arg_name, exp_depth = 1) : 

        if arg_val is None : 
            return None
        
        depth = _get_iter_depth(arg_val)
        if depth == 0 : 
            raise TypeError(f"{arg_name} must be an iterable not {type(arg_val)}")
        
        if exp_depth == depth : 
            arg_val = [arg_val]
        elif depth - exp_depth != 1 : 
            raise IndexError(
                f"{arg_name} should have a nesting depth of {exp_depth} or {exp_depth + 1}, not {depth}"
            )

        arg_val = [np.asarray(itm) for itm in arg_val]
        return arg_val
    
    if isinstance(expected_depth, int) : 
        expected_depth = [expected_depth for _ in args_to_convert]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            args = list(args) # convert to list to allow assignment

            for arg_name, exp_depth in zip(args_to_convert, expected_depth):
                # Check if in keyword
                if arg_name in kwargs:
                    kwargs[arg_name] = conv_arg(kwargs[arg_name], arg_name, exp_depth)
                
                # Check if in arguments
                elif args : 
                    arg_index = func.__code__.co_varnames.index(arg_name)
                    if arg_index < len(args):
                        args[arg_index] = conv_arg(args[arg_index], arg_name, exp_depth)

            return func(*args, **kwargs)
        return wrapper
    return decorator


def _check_input_format (
        true_class          : list[list[bool]|NDArray[np.bool_]],
        pred_score          : list[list[float]|NDArray[np.float64]],
        distance_matrix     : list[NDArray[np.int64]]|None       
    ) -> None : 
    """
    Function that checks that `true_class`, `pred_score` and `distance_matrix`
    all have the same shape, and each item in them also match.

    Parameters
    ----------
    distance_matrix : list[NDArray[np.int64]],
        List of one distance matrix per graph

    true_class : list[list[bool] | NDArray[np.bool_]]
        List with on list per graph indicating the correct (boolean) class of
        each node, must have at least one positive (true) class per graph

    pred_score : list[list[float] | NDArray[np.float64]]
        List with one list per graph indicating the predicted score (float) for
        each node in that graph, with higher score indicating higher likelihood 
        of that node being the positive ('True') class
    """    

    if len(true_class) != len(pred_score) : 
        raise IndexError(f"Length mismatch: true_class={len(true_class)}, pred_score={len(pred_score)}")
    
    if (distance_matrix is not None) and len(distance_matrix) != len(true_class) : 
        raise IndexError(f"Length mismatch: true_class={len(true_class)}, distance_matrix={len(distance_matrix)}")

    for i in range(len(true_class)) : 
        
        if len(true_class[i]) != len(pred_score[i]):
            raise IndexError(
                f"Length mismatch at item {i}: true_class[{i}]={len(true_class[i])}, pred_score[{i}]={len(pred_score[i])}"
            )

        if distance_matrix is not None:
            dm = distance_matrix[i]
            if dm.shape[0] != dm.shape[1]:
                raise IndexError(f"distance_matrix[{i}] must be square, got shape {dm.shape}")
            if len(true_class[i]) != dm.shape[0]:
                raise IndexError(
                    f"Length mismatch at item {i}: true_class[{i}]={len(true_class[i])}, distance_matrix[{i}]={dm.shape}"
                )

        
_K_METHODS_DEF = {
    "pos_count"     : lambda tc,ps : int(np.sum(tc)),
    "pred_count"    : lambda tc,ps : int(np.sum([s>0 for s in ps]))
}

_AGG_METHOD = {
    "mean"      : lambda a : np.mean(a),
    "min"       : lambda a : np.min(a),
    "max"       : lambda a : np.max(a),
    "median"    : lambda a : np.median(a)
}

#──────────────────────────────METRICS───────────────────────────────#

@_ensure_list_of_arr(
    args_to_convert = ["true_class","pred_score","ignore_idx"],
    expected_depth  = 1
    )
def top_k (
        true_class      : list[list[bool]|NDArray[np.bool_]],
        pred_score      : list[list[float]|NDArray[np.float64]],
        k               : int|str|Callable[[NDArray[np.bool_],NDArray[np.float64]], int]    = 1,
        ignore_idx      : list[list[int]]|None                                              = None,
        return_arr      : bool                                                              = False,
        zero_bad_k      : bool                                                              = True,
    ) -> np.float64|NDArray[np.bool_] :
    """

    Calculates the top-k binary node-classification metrics.

    Top-K Description
    -----------------
    In a given graph each vertex is assigned one of two true classes, either
    the positive (1) or negative (0) class. A model then assigns a score to
    each vertex in the graph, with higher score indicating a vertex is more 
    likely to be the positive class (1). 

    For a single graph, the top-k metric provides a boolean, which is true if 
    within the k highest scoring vertices a true positive class exists. For 
    a set of graphs, the top-k is then the fraction of graphs where this is 
    true.

    Usage
    -----
    The ``true_class`` and ``pred_score`` provide a list containing a list or 
    array for each graph with a value per vertex, boolean for the correct class 
    and float for predicted score. The function will return the top-k for the 
    whole set, or a list of boolean if ``return_arr == True``


    Parameters
    ----------
    true_class : list[list[bool] | NDArray[np.bool_]]
        List with one list/array per graph, containing a boolean for each 
        vertex indicating if its the target 'true' class. 

    pred_score : list[list[float] | NDArray[np.float64]]
        List with one list/array per graph, containing one float for each 
        vertex indicating how likely the model thinks that vertex is the target
        'true' class 

    k : int | str | Callable[[NDArray[np.bool_],NDArray[np.float64]], int], optional
        How many vertices to include in assigning correct or not, correct 
        prediction will be if a true class is in top ``k`` highest ranked 
        vertices by `pred_score`, provided as either an integer, a function of 
        true and predicted score for a given set, or a string for a preset 
        function: 

        - ``"pos_count"``   : number of positive class
        - ``"pred_count"``  : number of predicted positive class (score > 0),

         by default 1

    ignore_idx : list[list[int]] | None, optional
        If set, provide a list with one list per graph indicating which vertex
        indices in that graph should be excluded from the analysis, if 
        ``ignore_idx == None`` all vertices are included, by default None

    return_arr : bool, optional
        If ``return_arr == True`` return an array, with a boolean for each 
        graph instead of the average of that array, by default False

    zero_bad_k : bool, optional,
        if ``k <= 0`` for a given graph, top-k cannot be calculated, if 
        ``zero_bad_k == True`` top-k for these graphs will be set to false, 
        if  ``zero_bad_k == False`` it will raise an error, by default True

    Returns
    -------
    np.float64|NDArray[np.bool_]
        Returns either the top-k metric for the dataset provided, or an array
        with a boolean for each graph, indicating if the positive, 'true' class
        was in the top-k best scoring vertices.

    """

    # Remove idx to ignore
    if ignore_idx is not None : 
        true_class = [np.delete(tc,igidx) for tc, igidx in zip(true_class,ignore_idx)]
        pred_score = [np.delete(ps,igidx) for ps, igidx in zip(pred_score,ignore_idx)]

    
    # Parse K-Function
    k_func : Callable[[NDArray[np.bool_],NDArray[np.float64]], int]|None = None

    if isinstance(k,int) : 
        k_func = lambda tc,ps : k

    elif isinstance(k, str) : 
        if k not in _K_METHODS_DEF : 
            raise KeyError(f"{k}  is not a recognized preset function")
        k_func = _K_METHODS_DEF[k]


    elif callable(k) : 
        k_func = k
    
    else : 
        raise TypeError(f"k was provided as {k} (type {type(k)}), but can only be int, str or function")

    

    _check_input_format(
        true_class      = true_class,
        pred_score      = pred_score,
        distance_matrix = None
        )
    
    # Calculate top-k for each graph
    kvs     = [int(k_func(tc,ps)) for tc, ps in zip(true_class,pred_score) ]

    if not zero_bad_k and any([kv<=0 for kv in kvs ]) : 
        bad_k = list(set([(i,kv) for i,kv in enumerate(kvs) if kv <= 0 ]))
        if all([kv<=0 for kv in kvs ]) :
            raise RuntimeError("k <= 0 for all item, k =" + ",".join([str(x[1]) for x in bad_k[:4]]) + ("..." if len(bad_k)>4 else ""))
        raise RuntimeError("k <= 0 for some items : " + ",".join([f"(idx={x[0]},k={x[1]})" for x in bad_k[:4]]) + ("..." if len(bad_k)>4 else ""))

    result  = np.array([False if kv <= 0 else np.sum(tc[np.argsort(ps)[-min(kv, len(ps)) :]]) > 0 for tc, ps,kv in zip(true_class,pred_score,kvs)])

    if return_arr : 
        return result
    
    return np.average(result)
    

@_ensure_list_of_arr(
    args_to_convert =["true_class","pred_score","ignore_idx"],
    expected_depth  = 1
    )
def node_rank_descent (
        true_class      : list[list[bool]|NDArray[np.bool_]],
        pred_score      : list[list[float]|NDArray[np.float64]],
        ignore_idx      : list[list[int]]|None                      = None,
        return_arr      : bool                                      = False,
    )  -> NDArray[np.float64] | float: 
    """
    Calculate Node-Rank-Descent (NRD) metric.

    NRD Description
    ---------------
    Node-Rank-Descent (NRD) is a metric for binary classification across 
    multiple sets (such as nodes in across multiple graphs), where its known
    at least one  positive (`true`) class is present in each set.

    For one set, NRD is provided a score for each item, where higher score 
    indicates higher likelihood of that item being the positive ('true') class,
    as well as the correct class (boolean) for each item. 

    NRD is then a measure of how far down one must go in the score list (
    high to low) before encountering a positive class. 

    This number is then adjusted based on number of elements in the set, and 
    the number of elements with positive class in the set, since the smaller 
    the set and the more positive class item in it, the more likely it is for a
    model to select a positive class by random chance. 

    Usage
    -----
    This function is provided the predicted scores as a list which for each set
    contains a list of each items in that set score, together with a similarly 
    structured list, but with true classes (boolean) instead of score. 


    Parameters
    ----------
    true_class : list[list[bool] | NDArray[np.bool_]]
        A list with one list per set, containing the correct class (boolean) 
        for each item in that set

    pred_score : list[list[float] | NDArray[np.float64]]
        A list with one list per set, containing the score (float) for each 
        for each item in that set, with higher score indicating higher
        likelihood of that item being the positive ('true') class.

    ignore_idx : list[list[int]] | None, optional
        list, with one list per set, containing indices of element in that 
        set that should be ignored in the metric calculation, all elements 
        included if ``ignore_idx == None``, by default None

    return_arr : bool, optional
        If true will return a list of the score per set instead of the average
        across all sets, by default False
 
    Returns
    -------
    NDArray[Any] | floating[Any]
        Average value across all sets, of if `return_arr == True` an array with
        the value for each individual item in the set
    """    

    nrd_score = []

    _check_input_format(
        true_class      = true_class,
        pred_score      = pred_score,
        distance_matrix = None
        )
    

    for i,(t,p) in enumerate(zip(true_class,pred_score)) : 

        if ignore_idx is not None : 
            t = np.delete(t,ignore_idx[i])
            p = np.delete(p,ignore_idx[i])

        if not np.any(t):
            raise ValueError(f"NRD requires at least one positive node per set (after ignore_idx). Item {i} has none.")

        # Get lowest rank of true positive
        positive_class_idx = np.nonzero(t)[0]
        vertex_ranking = (len(p)-1) - np.argsort(np.argsort(p))
        rank_depth = np.min(vertex_ranking[positive_class_idx]) + 1

        # Number of vertices and true positives
        v_count = len(t)
        v_positive_count = np.sum(t)

        # Calcualte nrd score
        nrd = np.prod([(v_count - v_positive_count - r)/(v_count-r) for r  in range(rank_depth)])
        nrd_score.append(nrd)


    if return_arr : 
        return np.array(nrd_score)
    return np.average(nrd_score)

nrd = node_rank_descent

@_ensure_list_of_arr(
    args_to_convert = ["distance_matrix","true_class","pred_score","ignore_idx"],
    expected_depth  = [2,1,1,1]
    )
def adjusted_node_proximity_index (
        distance_matrix     : list[NDArray[np.int64]], 
        true_class          : list[list[bool]|NDArray[np.bool_]], 
        pred_score          : list[list[float]|NDArray[np.float64]],
        ignore_idx          : list[list[int]]|None                                               = None,
        k                   : int|str|Callable[[NDArray[np.bool_],NDArray[np.float64]], int]     = 1,
        return_arr          : bool                                                               = False,
        no_neg_as_nan       : bool                                                               = False,
    )  -> NDArray[np.float64] | float: 
    """
    Calculates the Adjusted Node Proximity Index (ANPI) metric

    ANPI Description
    ----------------
    ANPI is a metric for binary node-classification. Unlike standard 
    classification in node-classification, the nodes are connected, and thus a 
    model may be able to identify a general area where the positive ('true') 
    class is located, but not specify the exact node well. 

    In ANPI, each node in a graph is given a score, with higher score indicating
    higher likelihood that, that node is the positive ('true') class. Then the
    top k nodes are selected and the minimum distance from them to any positive 
    class in the graph is calculated. 

    However the smaller the graph is, and the more nodes with positive class 
    are present, the shorter the distances will be, biasing the metric. ANPI 
    accounts for this by calculating the expectation value, or average distance
    would be for random selection.

    The final ANPI score, is then the minimum distance from any of the top-k
    scoring nodes to any correct positive-class node in the graph, divided by
    the average distance one would get from selecting k random nodes in that
    specific graph. 

    ANPI scores below one thus indicate performance better than random chance
    while ANPI score above one indicate performance worse than random chance.

    Usage
    -----
    ANPI takes a list of distance matrices, one per graph as an input together
    with a list with one list per graph with the correct class (boolean) and a
    similar shaped list with predicted scorers (float). 

    Parameters
    ----------
    distance_matrix : list[NDArray[np.int64]],
        List of one distance matrix per graph

    true_class : list[list[bool] | NDArray[np.bool_]]
        List with on list per graph indicating the correct (boolean) class of
        each node, must have at least one positive (true) class per graph

    pred_score : list[list[float] | NDArray[np.float64]]
        List with one list per graph indicating the predicted score (float) for
        each node in that graph, with higher score indicating higher likelihood 
        of that node being the positive ('True') class

    ignore_idx : list[list[int]] | None, optional
        List with one list per graph indicating the indices of nodes in that
        graph to be excluded from the metric, note that this will only affect
        which predictions to exclude and thus does not affect the average 
        distance calculations. All element included if``ignore_idx == None``,
        by default None

    k : int | str | Callable[[NDArray[np.bool_],NDArray[np.float64]], int], optional
        The number of nodes that may be selected, this will be the k nodes with
        the highest predicted score in each graph. k can be provided either as
        an integer, a function of true and predicted score for a given set, or 
        a string for a preset function: 

        - ``"pos_count"``   : number of positive class
        - ``"pred_count"``  : number of predicted positive class (score > 0)

         , by default 1

    return_arr : bool, optional
        If true will return a list of the score per set instead of the average
        across all sets, by default False

    no_neg_as_nan : bool, optional,
        If ANPI encounters a graph only with positive class nodes, the output 
        is defined as 1 (same as random selection), however, if 
        `no_neg_as_nan == True` and  `return_arr == True` will set these 
        to `np.nan` instead, useful for debugging purposes, by default False,

    Returns
    -------
    NDArray[Any] | floating[Any]
        Average value across all sets, of if `return_arr == True` an array with
        the value for each individual item in the set
    """    


    if isinstance(k, str):
        if k not in _K_METHODS_DEF:
            raise KeyError(f"{k} is not a recognized preset function")
        k = _K_METHODS_DEF[k]

    elif not isinstance(k, int) and not callable(k) : 
        raise TypeError(f"k must be a string, function or integer nok {type(k)}")


    _check_input_format(
        true_class      = true_class,
        pred_score      = pred_score,
        distance_matrix = distance_matrix
        )
    

    res = []
    for i, (dm, tc, ps) in enumerate(zip(distance_matrix, true_class, pred_score)) : 

        if callable(k) : 
            kv = k(tc,ps)
        else : 
            kv = k

        if not ignore_idx is None : 
            ps = ps.copy()
            ps[ignore_idx[i]] = -np.inf

        if kv <= 0 or kv > len(ps):
            raise ValueError(f"k must be in [1, {len(ps)}], got {kv}")

        if np.all(tc):
            res.append(np.nan if no_neg_as_nan else 1.0) # cant outperform random selection so 1
            continue

        pred_idx = np.argsort(ps)[-kv:]
        true_idx = np.nonzero(tc)[0]

        if len(true_idx) == 0:
            raise ValueError(f"ANPI requires at least one positive node per graph, no positive class node in graph {i}.")

        dist_to_pos = np.array(sorted(np.min(dm[:,true_idx],axis=1)))
        probability = np.array([(math.comb(len(dist_to_pos) - (j+1), kv - 1) / math.comb(len(dist_to_pos), kv)) for j in range(len(dist_to_pos) - kv + 1)])
        avgmin = np.sum(probability * dist_to_pos[:len(probability)])

        if avgmin == 0:
            res.append(np.nan if no_neg_as_nan else 1.0) # cant outperform random selection so 1
            continue
        res.append(np.min(dm[:,true_idx][pred_idx])/avgmin)

    if return_arr : 
        return np.array(res)
    return np.nanmean(res)


anpi = adjusted_node_proximity_index

@_ensure_list_of_arr(
    args_to_convert = ["distance_matrix","true_class","pred_score","ignore_idx"],
    expected_depth  = [2,1,1,1]
    )
def relative_radial_confidence_score (
        distance_matrix     : list[NDArray[np.int64]], 
        true_class          : list[list[bool]|NDArray[np.bool_]], 
        pred_score          : list[list[float]|NDArray[np.float64]],
        radius              : int                                           = 0, 
        strict_radius       : bool                                          = False, 
        agg_method          : str|Callable[[NDArray[np.float64]], float]    = "mean", 
        ignore_idx          : list[list[int]]|None                          = None,
        no_outer            : float|None                                    = None, 
        return_arr          :bool                                           = False
    ) -> NDArray[float] | float | None : 
    """
    Calculates Relative Radial Confidence Score (RRCS)

    RRCS Description
    ----------------
    Relative Radial Confidence Score (RRCS), is a measure of the confidence 
    of a model on a binary node-classification task. 

    For a given graph, RRCS is provided a score for each node with higher score
    indicating the model assigns higher likelihood that a node is the positive,
    ('true') class, together with the correct class of each node. 

    RRCS then calculates the average score of the positive 'True' class and the
    average score of the negative 'False' class, and returns the negative class
    score divided by the positive class score.

    In some cases a model may have a global understanding of roughly the 
    location of the positive class but may not be able to assign it well to 
    a specific node. To investigate this a radius can be set in RRCS, in which
    case it calculates the average score given to all nodes which are within 
    that radius of a positive class compared to the rest. Investigating 
    different radii can give a sense of how well the model understands where 
    the positive classes are in the graph. 

    For detailed investigations, strict radius can be applied, allowing one to
    check the score only of nodes at a given radius instead of within that 
    radius. 


    Parameters
    ----------
    distance_matrix : list[NDArray[np.int64]],
        List of one distance matrix per graph

    true_class : list[list[bool] | NDArray[np.bool_]]
        List with on list per graph indicating the correct (boolean) class of
        each node, must have at least one positive (true) class per graph

    pred_score : list[list[float] | NDArray[np.float64]]
        List with one list per graph indicating the predicted score (float) for
        each node in that graph, with higher score indicating higher likelihood 
        of that node being the positive ('True') class

    radius : int, optional
        Radius around each positive class to include in the average score of 
        the positive class, by default 0

    strict_radius : bool, optional
        If false includes all nodes within `radius` around each positive class
        or if true to include only nodes exactly at radius around each positive
        class (thus excluding positive class itself), by default False

    agg_method : str | Callable[[NDArray[np.float64]], float], optional
        Method used to aggregate the scores for positive and negative classes,
        either a function which takes a list of values or a string indicating 
        one of the preset aggregation methods : 

            - 'mean' 
            - 'min'
            - 'max'
            - 'median'
            
        , by default "mean"

    ignore_idx : list[list[int]] | None, optional
        list, with one list per set, containing indices of element in that 
        set that should be ignored in the metric calculation, all elements 
        included if is ``ignore_idx == None``, by default None

    no_outer : float|None, optional
        If either no positive or negative nodes can be found, what score to
        give that graph, if None excluded from score if 'return_arr == False'
        , by default None

    return_arr : bool, optional
        If true will return a list of the score per set instead of the average
        across all sets, by default False


    Returns
    -------
    NDArray[Any] | floating[Any] | None
        Average value across all sets, of if `return_arr == True` an array with
        the value for each individual item in the set. if `return_arr == False` 
        and `no_outer == None` then graphs with no positive or negative nodes 
        are excluded from score. Returns None if `return_arr == False` and all
        items are `None`.

    """    

    _check_input_format(
        true_class      = true_class,
        pred_score      = pred_score,
        distance_matrix = distance_matrix
        )

    res = []

    if radius < 0 : 
        raise ValueError("radius must be >= 0 ")


    def calc_with_radius (dis_mat, center_point, radius, strict_radius) : 

        if strict_radius : 
            within_r = set(np.nonzero(dis_mat[np.array(center_point)] == radius)[1])
            if radius > 0 :
                within_r = within_r.difference(set(center_point))
        else : 
            within_r = set(np.nonzero(dis_mat[np.array(center_point)] <= radius)[1])
        return within_r
    

    if isinstance(agg_method, str):
        if agg_method not in _AGG_METHOD:
            raise KeyError(f"{agg_method} is not a recognized agg_method")
        agg_method = _AGG_METHOD[agg_method]

    elif not callable(agg_method) : 
        raise TypeError("agg_method must be a string or a callable method")


     
    for mid, (dm, tc, ps) in enumerate(zip(distance_matrix, true_class, pred_score)) :

        true_site_idx  = np.nonzero(tc)[0]
        if len(true_site_idx) == 0:
            res.append(no_outer)
            continue

        within_radius  = calc_with_radius(dm,true_site_idx,radius,strict_radius)
        outside_radius = set(range(len(ps))).difference(within_radius)


        if (ignore_idx is not None) and (len(ignore_idx[mid]) > 0) : 

            ignored = set(ignore_idx[mid])
            within_radius  = within_radius.difference(ignored)
            outside_radius = outside_radius.difference(ignored)

        if len(outside_radius) == 0 or len(within_radius) == 0 : 
            res.append(no_outer)
        else : 
            outside_score   = agg_method(ps[list(outside_radius)])
            inside_score    = agg_method(ps[list(within_radius)])
            res.append(outside_score/inside_score)


    if return_arr : 
        return np.array(res)

    non_none_res = [r for r in res if (r is not None)]

    if len(non_none_res) == 0 : 
        return None

    return np.average(non_none_res)


rrcs = relative_radial_confidence_score