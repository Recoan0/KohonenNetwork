The Kohonen Network is a self-organising map, which enables the user to visualise high-dimensional data by reducing its dimensions to a map.

For more information on how a Kohonen Network works check the [Kohonen Network notebook](kohonen.ipynb)

2 Implementations are provided in this repository:
1. An object oriented way using for-loops and cell-by-cell operations
2. A vectorized way using numpy functions and matrix operations to allow for parallel execution of operations on cells

This project shows how using vectorization and built in numpy functionality can drastically improve performance, seeing improvements in runtime of over 100x.
