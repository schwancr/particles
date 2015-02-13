# particles
sklearn for particles

This package includes code that creates wrappers for sklearns 
estimators on the fly. The wrapper is necessary when using 
different data objects than sklearn expects.

Sklearn uses a single array X as input, which is typically
shaped [n\_points, n\_features]. However, in this case, we're
interested in studying many trajectories of indistinguishable
particles, so the input is a list of length n\_seq, where 
each item contains n\_frames snapshots of n\_particles different
particles, which each have n\_features feature values.


