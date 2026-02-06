# Node Classification Metrics, (NoClaMe) 
A small script containing four different metrics for binary node/vertex classification: top-k and three new metrics **NRD**, **ANPI** and **RRCS**.

The metrics are designed for binary vertex classification across multiple graphs, where at least one vertex in each graph is of the positive (true/1) class. This is often the case for chemical applications such as reaction site prediction. 

In addition to the already established top-k metric, the script includes three new metrics which are: 

- **N**ode-**R**ank-**D**escent (**NRD**) : This metric is aimed at providing an alternative to top-k, without a bias towards smaller graph with many positive classes.

- **A**djusted **N**ode **P**roximity **I**ndex (**ANPI**) : A metric designed to evaluate how well the model understands global position of the positive class.

- **R**elative **R**adial **C**onfidence **S**core (**RRCS**) : **RRCS**  attempt to evaluate the confidence of the model for its predictions. 

## Citation

If any of the three new metric, **NRD**, **ANPI** or **RRCS** is used somewhere where appropriate, please cite the paper 'Can graph neural networks understand chemical elements ?' [Currently Not Published]

# Usage 

## Installation 
NoClaMe can either be installed via pip from the github repository using 

```
pip install "git+https://github.com/KylleV/noclame.git"
```

Alternatively, NoClaMe is a single python file, and `noclame.py` can be downloaded which contain all relevant functions.


## Using the Metrics
For the example below, two graphs are used, one representing ethanol, and one representing imidazole 

```python
import numpy as np

#ethanol
etoh_dm = np.array(
    [[0, 1, 2],
     [1, 0, 1],
     [2, 1, 0]]
)

etoh_true = [1  ,0  ,0]
etoh_pred = [0.4,0.2,0.3]

#imidazole
imid_dm = np.array(
   [[0, 1, 2, 2, 1],
    [1, 0, 1, 2, 2],
    [2, 1, 0, 1, 2],
    [2, 2, 1, 0, 1],
    [1, 2, 2, 1, 0]]
)

imid_true = [0  ,0  ,1  ,0  ,1]
imid_pred = [0.4,0.2,0.5,0.6,0.4]

distance_matrices = [etoh_dm  , imid_dm]
true_class        = [etoh_true, imid_true]
pred_score        = [etoh_pred, imid_pred]
```

Metrics can then be calculated as shown below.

```python
from noclame import nrd, anpi, top_k, rrcs

top_k(true_class,pred_score)
>>> 0.5

nrd(true_class,pred_score)
>>> 0.4833

anpi(distance_matrices,true_class,pred_score)
>>> 0.8333

rrcs(distance_matrices,true_class,pred_score)
>>> 0.757
```

Scores for individual graph can be retrieved by setting ``return_arr=True``

```python
from noclame import nrd, anpi, top_k, rrcs

top_k(true_class,pred_score, return_arr=True)
>>> array([ True, False])

nrd(true_class,pred_score, return_arr=True)
>><> array([0.66666667, 0.3])

anpi(distance_matrices,true_class,pred_score, return_arr=True)
>>> array([0, 1.66666667])

rrcs(distance_matrices,true_class,pred_score, return_arr=True)
>>> array([0.625, 0.88888889])
```


# Metric Description 

## General Input 
In general for all metric listed, for each graph $G = (V,E)$, for each vertex $v$, the metrics are require :

- The ground-truth binary label for each vertex, from which we can define the set of vertices with the positive (true/1) class  $V_+ \subseteq V$. 

- The model's confidence score for each vertex, where a higher score indicates higher likelihood of the vertex being the positive class.

**ANPI** and **RRCS** also require a distance matrix for each graph to be analyzed


## Top-K
Top-K metric is provided the input described above in "General Input" and for a given graph, the top-k metric then checks if with in the highest k scoring vertices that a positive class exists. The metric across a dataset is then the fraction of graphs where this is the case. 

However Top-K presents a bias towards smaller graphs and graph with many positive classes, where the chance of selecting a true positive at random is higher 

## **N**ode-**R**ank-**D**escent (**NRD**) 

**NRD** is designed to provide an alternative to top-k without the bias for small graphs with many positive classes. **NRD** functions by going down the list of vertex scores (from high to low) until a correct positive class is found.

**NRD** then ranks each vertex based on its confidence score, and starting from the highest scoring vertex goes down the list of vertices in order of rank, until a true positive is found. The rank of the first true positive, $R$, indicates how many attempts the model needed before succeeding. The value of **NRD** can then be calculated using the equation below. 

$$
\text{NRD} = \prod_{r=1}^{R} \frac{|V| - |V_+| - r }{|V| - r}
$$

Where $|V|$ indicates the graph order (number of vertices), and $\vert V_+| \vert$ the number of vertices which are the positive class. 

**NRD** can be interpreted as how much better than random chance the model is. For example, $\text{NRD} = 0.9$ means there is a $10\%$ chance of finding a true positive within $R$ attempts by random selection. Note that for finite graphs a score of $1$ is not possible, and maximal score is slightly larger for larger graphs. This does mean one should be careful if one compared **NRD** scores between two sets of graphs with very different average sizes.

##  **A**djusted **N**ode **P**roximity **I**ndex (**ANPI**)

**NRD** and top-k take into account which graph a vertex is associated with, however not how vertices are related to each other.  This allows an error type, where the predicted vertex, may not be correct, but is proximal to the real one, which for some applications may be sufficient information. This type error could indicate that the model can understand which substructure positive class vertices are located in but not where in that substructure, it understands global structure but not local ones. 

**ANPI** attempts to measure this by calculating the distance from predicted positive class to the true ones. This is done as the minimum distance from the $k$ vertices with the best score to any vertices with the positive class ($V_+$). 

This measure is affected by order/size of the graph ($\vert V \vert$) (number of vertices), the number of vertices with the positive class $\vert V_+ \vert$, as well as the selection of $k$. To correct for this the expected distance of selecting $k$ vertices by random chance in the graph is calculated by the equation below.

$$
    \overline{\min d_+(V)} = \sum_{i=1}^{|V|-k+2} d_+(v_i) \times \frac{{|V|-i \choose k}}{{|V| \choose k}} 
$$

Where $d_+(v)$ is the minimum distance from vertex $v$ to any vertex with a positive class ($V_+$). 

The **ANPI** metric is then calculated as the minimum distance for the predicted vertices, the $k$ vertices with highest score ($V_\text{pred}$) as a fraction of the expected distance $\overline{\min d_+(V)}$ in the graph

$$
\text{ANPI}_k = \frac{\min_{v_p \in V_\text{pred}} d_+(v_p)}{\overline{\min d_+(V)}}
$$

This means that an **ANPI** score above $1$ in indicative of performance worse than random change, and score below $1$ indicates performance better than random selection.


## **R**elative **R**adial **C**onfidence **S**core (**RRCS**)

Unlike the other metrics, which attempt to evaluate how well a model performance, **RRCS** attempt to investigate a model's confidence.

**RRCS** calculates the average score of vertices with the positive class divided by the average score provided to vertices with the negative class. However **RRCS** also takes a radius $r$ input, and if $r>0$ then all vertices within a distance of $r$ from the positive class will be considered positive as well. This allows you to test whether the model is confident about the local substructure even if the exact location is unclear.


