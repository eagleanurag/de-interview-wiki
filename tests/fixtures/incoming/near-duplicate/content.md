# Window functions versus GROUP BY

A window function aggregates across a partition but keeps the rows, where GROUP BY collapses them into one row per group. Use a window when every row still needs to be visible next to its own aggregate.
