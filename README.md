# pyUpSet
A pure-python implementation of the UpSet suite of visualisation methods by Lex, Gehlenborg et al.


## Purpose
The purpose of this package is to reproduce (statically) some of the visualisations that can be obtained through the UpSet tool of Lex, Gehlenborg et al. (See http://vcg.github.io/upset/about/#)

In particular, __pyUpSet's focus is on intersections__, which motivates many of the design choices behind the exposed 
interface and the internal mechanics of the module. (More on this below.)

Consistently with the documentation used for Lex et al.'s UpSet, the data employed in the following examples comes from the movie data set of the [GroupLens Labs](http://grouplens.org/datasets/movielens).

## How it works

The current interface is very simple, and that's how the module is intended to be used. Plots are produced by calling
 the function `plot`, as in 
```
import pyUpset as pyu
pyu.plot(data_dict)
```
to produce
![alt text](https://github.com/ImSoErgodic/py-upset/blob/master/basic.png "")

__N.B.:__ Notice that intersections are _exclusive_, meaning that they form a [partition](https://en.wikipedia.org/wiki/Partition_of_a_set) of the union of the base 
sets.

Displayed intersections can also be filtered or sorted by size or degree:
```
pyu.plot(data_dict, unique_keys = ['title'], sort_by='degree', inters_size_bounds=(20, 400))
```
produces
![alt text](https://github.com/ImSoErgodic/py-upset/blob/master/basic_filtered.png "")

### Additional plots

It is possible to add further plots that use information contained in the data frames, as in 
```
pyu.plot(data_dict, unique_keys = ['title'], 
         additional_plots=[{'kind':'scatter', 'data':{'x':'rating_avg', 'y':'rating_std'}},
                           {'kind':'scatter', 'data':{'x':'rating_avg', 'y':'rating_std'}}]) 
         # identical subgraphs only for demonstration purposes
```
This produces
![alt text](https://github.com/ImSoErgodic/py-upset/blob/master/basic_w_subplots.png "")

__Additional graph kinds currently supported__: `scatter`

### Intersection highlighting

pyUpSet supports the highlighting of  "queries", which are essentially a representation of an intersection as a tuple
. For example, the following call produces graphs where all data belonging to the intersection of the "romance" and 
"adventure" sets is highlighted.
```
pyu.plot(data_dict, unique_keys = ['title'],
         additional_plots=[{'kind':'scatter', 'data':{'x':'rating_avg', 'y':'rating_std'}},
                           {'kind':'scatter', 'data':{'x':'rating_avg', 'y':'rating_std'}}],
         query = [('adventure', 'action')]
        )
```
![alt text](https://github.com/ImSoErgodic/py-upset/blob/master/basic_w_subplots_query.png "")

## A note on the input format
pyUpSet has a very specific use case: It is focussed on the study of intersections 
of sets. In order for a definition of intersection to make sense, and even more for the integration of additional 
graphs to be meaningful, it is assumed that the input data frames have properties of _homonymy_ (they contain 
columns with the same names) and _homogeneity_ (columns with the same name, intuitively, contain data of the same 
kind). While hononymy is a purely interface-dependent requirement whose aim is primarily to make pyUpSet's interface 
leaner, homogeneity has a functional role in allowing definitions of uniqueness and commonality for the data points 
in the input data frames. 

Whenever possible, pyUpSet will try to check for and enforce the two above properties. 
In particular, when the `unique_keys` argument of `plot`, which is intended to specify one or more columns that can 
uniquely identify rows in the data frames, is omitted pyUpSet will try to use all columns 
with common names across the data frames as a list of unique keys. Under the hypotheses of homogeneity and homonymy 
this should be enough for all the operations carried out by pyUpSet to complete successfully.


## Upcoming changes
Please bear in mind that pyUpset is under active development so current behaviour may change at any time. In 
particular, here is a list of changes, in no particular order, to be expected soon:
* improved OO interface for increased flexibility and customisation
* universal support for all graphs callable as methods on `matplotlib` Axes
* replacement of the temporary annotation for base set names with a table, akin to that in the original UpSet
* automated scaling of figure and axes grid according to the number of sets, intersections and additional plots
