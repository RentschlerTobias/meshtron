# v2 training grid (topo order, all labelled points, no block count)

| run | params M | best epoch | val loss | val n | valid | struct ok | corner mean (median) | corner max (median) | exact conn |
|---|---|---|---|---|---|---|---|---|---|
| conn_M | 7.21 | 70 | 0.000 | 35 |  |  |  |  | 35 |
| conn_S | 1.32 | 125 | 0.000 | 35 |  |  |  |  | 35 |
| hexarow_L | 61.12 | 45 | 1.934 | 35 | 35 | 0 |  |  |  |
| hexarow_M | 7.39 | 50 | 1.629 | 35 | 35 | 17 | 151.4833984375 | 382.7200012207031 |  |
| hexarow_S | 1.57 | 135 | 1.642 | 35 | 35 | 8 | 123.38855743408203 | 388.2192687988281 |  |
| smoke_conn | 1.30 | 10 | 3.191 | 5 |  |  |  |  | 0 |
| smoke_hexarow | 7.28 | 75 | 3.681 | 5 | 5 | 0 |  |  |  |
| smoke_vertex | 7.25 | 75 | 3.531 | 5 | 5 | 5 | 133.93822703119932 | 339.52778784775114 |  |
| vertex_L | 60.91 | 30 | 2.479 | 35 | 35 | 34 | 133.00751906769753 | 361.35916021434275 |  |
| vertex_M | 7.28 | 40 | 2.392 | 35 | 35 | 34 | 138.13960356466566 | 388.2809577128514 |  |
| vertex_S | 1.52 | 85 | 2.377 | 35 | 35 | 34 | 123.6661046963461 | 337.1312722930742 |  |

corner values: distance generated vs GT corner in canonical order, dataset units x 1000.
