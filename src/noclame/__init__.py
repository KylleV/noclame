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

from .noclame import (
    top_k,
    nrd,
    anpi,
    rrcs,
    node_rank_descent,
    adjusted_node_proximity_index,
    relative_radial_confidence_score
)

__all__ = [
    "top_k",
    "nrd",
    "anpi",
    "rrcs",
    "node_rank_descent",
    "adjusted_node_proximity_index",
    "relative_radial_confidence_score"
]
