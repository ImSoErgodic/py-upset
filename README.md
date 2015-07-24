# pyUpset
A pure-python implementation of the UpSet suite of visualisation methods by Lex, Gehlenborg et al.

The purpose of this package is to reproduce (statically) some of the visualisations that can be obtained through the UpSet tool of Lex, Gehlenborg et al. (See http://vcg.github.io/upset/about/#)

At the moment the package contains a single module with the essential functionality to produce plots of intersections that reproduce those of UpSet. We plan to include further plotting facilities and support for intersection query.

## How it works

The current interface is very simple. It is possible to immediately obtain a graph going through the wrapper function `plot`, as in 
```
import pyUpset as pyu
pyu.plot(sets, set_names)
```
to produce
![alt text](https://github.com/ImSoErgodic/py-upset/blob/master/demo_plain.png "")

Displayed intersections can also be filtered or sorted by size or degree:
```
import numpy as np
pyu.plot(sets, set_names, sort_by='degree', inters_size_bounds=(20, np.inf))
```
produces
![alt text](https://github.com/ImSoErgodic/py-upset/blob/master/demo_filtered.png "")

Alternatively, one can go through the class instantiation process
```
us = pyu.UpSet(sets, set_names)
us.plot(sort_by='degree', inters_degree_bounds=(2, 4))
```
This is especially useful if one wants a finer control over the plotting process or wants to experiment with sorting. 
## Notes
Please bear in mind the following when using pyUpset:
* pyUpset is under active development so current behaviour may change at any time
* Some of the salient features of UpSet like support for additional graphs and intersection query/highlighting are not present yet. They will be introduced shortly and be built on top of pandas
* __A note on performance__: In some cases computing the intersections of numerous, large sets can be a costly operation. Since the limits of the axes in the figure are computed using the number of intersections that are within the boundaries specified by `inters_size_bounds` and `inters_degree_bounds`, if the two corresponding tuples don't change then generating new plots even for many large sets will be faster than the first run. If new boundaries are specified, then new intersections have to be computed.
