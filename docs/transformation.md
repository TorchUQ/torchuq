# Transformations


## Transformations for regression prediction types

|              | Point  | Distribution | Interval | Quantile | Particle |
| :----------: | :----: | :----------: | :------: | :------: | :------: |
|    Point     |   -    |      No      |    No    |    No    |    No    |
| Distribution |  Yes   |      -       |   Yes    |   Yes    |   Yes    |
|   Interval   |  Yes   |      No      |    -     |    No    |    No    |
|   Quantile   | ->Dist |     Yes      |  ->Dist  |    -     |  ->Dist  |
|   Particle   | ->Dist |     Yes      |  ->Dist  |   Yes    |  ->Dist  |
|   Ensemble   | ->Dist |     Yes      |  ->Dist  |  ->Dist  |  ->Dist  |


## Transformations for classification prediction types

|             | topk | categorical | uset |
| :---------: | :--: | :---------: | :--: |
|    topk     |  -   |             |      |
| categorical | Yes  |      -      | Yes  |
|    uset     |      |             |  -   |
|  ensemble   |      |             |      |
