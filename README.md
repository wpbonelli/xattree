# xattree

`attrs` + `xarray.DataTree` = `xattree`

"exa-tree", or "cat tree" if you like.

```python
import numpy as np
from numpy.typing import NDArray
from attrs import field, Factory
from xattree import xattree, dim, array, child

@xattree
class Grid:
    rows: int = dim(name="row", scope="root", default=3)
    cols: int = dim(name="col", scope="root", default=3)

@xattree
class Arrs:
    arr: NDArray[np.float64] = array(default=0.0, dims=("row", "col"))

@xattree
class Root:
    grid: Grid = field()
    arrs: Arrs = field()

grid = Grid()
root = Root(grid=grid)
arrs = Arrs(root)
root.data
<xarray.DataTree 'root'>
Group: /
│   Dimensions:  (row: 3, col: 3)
│   Coordinates:
│     * row      (row) int64 24B 0 1 2
│     * col      (col) int64 24B 0 1 2
├── Group: /grid
│       Attributes:
│           rows:     3
│           cols:     3
└── Group: /arrs
        Dimensions:  (row: 3, col: 3)
        Data variables:
            arr      (row, col) float64 72B 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
```