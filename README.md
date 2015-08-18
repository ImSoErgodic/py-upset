# pyUpSet
A pure-python implementation of the UpSet suite of visualisation methods by Lex, Gehlenborg et al.


## Purpose
The purpose of this package is to statically reproduce some of the visualisations that can be obtained through the 
[UpSet tool of Lex, Gehlenborg et al.](http://vcg.github.io/upset/about/#)

In particular, __pyUpSet strengthens UpSet's focus on intersections__, which motivates many of the design choices 
behind the exposed 
interface and the internal mechanics of the module. (More on this below.)

Consistently with the documentation used for Lex et al.'s UpSet, the data employed in the following examples comes 
from the movie data set of the [GroupLens Labs](http://grouplens.org/datasets/movielens).

## How it works

The current interface is very simple: Plots can be generated solely from the exposed function `plot`, whose arguments
 allow flexible customisations of the graphs. The easiest example is the plain, straightforward basic intersection 
 plot:  
```
import pyUpset as pyu
from pickle import load
with open('./test_data_dict.pckl', 'r') as f:
   data_dict = load(f)
pyu.plot(data_dict)
```
to produce
![basic plot](https://github.com/ImSoErgodic/py-upset/blob/master/pictures/basic.png "")

__N.B.:__ Notice that intersections are _exclusive_, meaning that they form a [partition](https://en.wikipedia.org/wiki/Partition_of_a_set) of the union of the base 
sets.

Displayed intersections can also be filtered or sorted by size or degree:
```
pyu.plot(data_dict, unique_keys = ['title'], sort_by='degree', inters_size_bounds=(20, 400))
```
produces
![basic filtering](https://github.com/ImSoErgodic/py-upset/blob/master/basic_filtered.png "")

The example above also uses the `unique_keys` kwarg, which specifies columns of the underlying data frames in 
`data_dict` that can be used to uniquely identify rows and possibly speed up the computation of intersections.

### Intersection highlighting

pyUpSet supports "queries", i.e. the highlighting of intersections. Intersections to highlight are specified through 
tuples. For example, the following call produces graphs where all data is highlighted that corresponds to movies 
classified as both "adventure" and "action", or "romance" and "war".
```
pyu.plot(data_dict, unique_keys = ['title'], 
         additional_plots=[{'kind':'scatter', 'data_quantities':{'x':'views', 'y':'rating_std'}},
                           {'kind':'hist', 'data_quantities':{'x':'views'}}],
         query = [('adventure', 'action'), ('romance', 'war')]
        )
```
![simple query](https://github.com/ImSoErgodic/py-upset/blob/master/pictures/query_basic.png "")

### Additional plots

It is possible to add further plots that use information contained in the data frames, as in 
```
pyu.plot(data_dict, unique_keys = ['title'], 
         additional_plots=[{'kind':'scatter', 'data_quantities':{'x':'views', 'y':'rating_std'}},
                           {'kind':'hist', 'data_quantities':{'x':'views'}}]), 
         query = [('adventure', 'action'), ('romance', 'war')]
```
This produces
![additional plots with query](https://github.com/ImSoErgodic/py-upset/blob/master/pictures/add_plots_query.png "")

The highlighting produced by the queries is passed to the additional graphs. The dictionary specifying the additional
 graphs can also take standard matplotlib arguments as kwargs:
 
 ```
 pyu.plot(data_dict, unique_keys = ['title'], 
         additional_plots=[{'kind':'scatter', 
                            'data_quantities':{'x':'views', 'y':'rating_std'},
                            'graph_properties':{'alpha':.8, 'lw':.4, 'edgecolor':'w', 's':50}},
                           {'kind':'hist', 
                            'data_quantities':{'x':'views'},
                            'graph_properties':{'bins':50}}], 
         query = [('adventure', 'action'), ('romance', 'war')])
 ```
 yields
 ![additional plots with query and properties](https://github
 .com/ImSoErgodic/py-upset/blob/master/pictures/add_plots_query_props.png "")

## A note on the input format
pyUpSet has a very specific use case: It is focussed on the study of intersections 
of sets. In order for a definition of intersection to make sense, and even more for the integration of additional 
graphs to be meaningful, it is assumed that the input data frames have properties of _homonymy_ (they contain 
columns with the same names) and _homogeneity_ (columns with the same name, intuitively, contain data of the same 
kind). While hononymy is a purely interface-dependent requirement whose aim is primarily to make pyUpSet's interface 
leaner, homogeneity has a functional role in allowing definitions of uniqueness and commonality for the data points 
in the input data frames. 

Whenever possible, pyUpSet will try to check for (and enforce) the two above properties. 
In particular, when the `unique_keys` argument of `plot` is omitted, pyUpSet will try to use all columns 
with common names across the data frames as a list of unique keys. Under the hypotheses of homogeneity and homonymy 
this should be enough for all the operations carried out by pyUpSet to complete successfully.


## Upcoming changes
Please bear in mind that pyUpset is under active development so current behaviour may change at any time. In 
particular, here is a list of changes, in no particular order, to be expected soon:
* improved OO interface for increased flexibility and customisation
* improved, automated scaling of figure and axes grid according to the number of sets, intersections and additional 
plots (at the moment manual resizing may be needed)
