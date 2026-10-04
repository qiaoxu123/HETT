# Visual Attribute Census

> Development census excludes `test_unseen` by policy. The existing canonical source census records 32,326 released instructions; this taxonomy census covers the 27,045 permitted train/validation instructions.

Instructions: **27,045**

At least one lexical visual attribute: **99.91%** (27,021)

At least two lexical visual attributes: **98.78%** (26,715)

## Split coverage

| split | instructions | ≥1 visual | ≥2 visual |
|---|---:|---:|---:|
| train_seen | 21,878 | 99.91% | 98.86% |
| val_seen | 2,470 | 99.92% | 98.87% |
| val_unseen | 2,697 | 99.93% | 98.03% |

## color

| attribute | occurrences |
|---|---:|
| gray | 10,223 |
| white | 9,968 |
| black | 6,659 |
| blue | 5,848 |
| red | 4,703 |
| brown | 4,403 |
| dark | 2,416 |
| green | 1,574 |
| light | 1,496 |
| yellow | 714 |
| bright | 241 |

## size

| attribute | occurrences |
|---|---:|
| large | 2,983 |
| long | 1,769 |
| small | 1,681 |
| short | 304 |
| narrow | 224 |
| tall | 145 |
| wide | 18 |

## context

| attribute | occurrences |
|---|---:|
| road | 16,662 |
| parking_lot | 8,151 |
| vegetation | 2,615 |
| intersection | 1,568 |
| grass | 1,049 |
| bridge | 498 |
| field | 234 |
| river | 198 |
| water | 84 |
| railway | 77 |
| open_area | 27 |
| crossing | 16 |
| sports_field | 12 |

## semantic

| attribute | occurrences |
|---|---:|
| car | 15,874 |
| building | 15,792 |
| parking | 7,341 |
| house | 4,241 |
| college | 1,390 |
| church | 1,061 |
| ground | 1,005 |
| library | 645 |
| school | 452 |
| tower | 48 |
| office | 34 |
| hospital | 1 |

## roof

| attribute | occurrences |
|---|---:|
| roof | 5,800 |
| flat_roof | 71 |
| pitched_roof | 17 |

## spatial_relation

| attribute | occurrences |
|---|---:|
| near | 6,614 |
| right | 6,188 |
| front | 6,113 |
| behind | 5,805 |
| left | 5,682 |
| between | 5,030 |
| across | 2,841 |
| along | 1,606 |
| opposite | 551 |
| before | 124 |
| past | 75 |
| after | 70 |

## count_ordinal

| attribute | occurrences |
|---|---:|
| one | 4,134 |
| two | 2,908 |
| second | 1,468 |
| third | 1,146 |
| nearest | 973 |
| first | 955 |
| three | 669 |
| multiple | 277 |
| farthest | 154 |

## shape

| attribute | occurrences |
|---|---:|
| rectangular | 871 |
| square | 593 |
| round | 293 |
| l_shaped | 287 |
| courtyard | 173 |
| u_shaped | 38 |
| irregular | 2 |
| elongated | 1 |

## Frequent visual-category combinations

| combination | instructions |
|---|---:|
| color+context+semantic | 10,994 |
| color+context+roof+semantic | 3,266 |
| color+context+semantic+size | 2,808 |
| color+semantic | 2,251 |
| context+semantic | 1,582 |
| color+context+roof+semantic+size | 894 |
| color+roof+semantic | 597 |
| color+semantic+size | 553 |
| context+semantic+size | 537 |
| color+context+semantic+shape | 513 |
| semantic | 299 |
| color+context+roof+semantic+shape | 278 |
| context+semantic+shape | 255 |
| color+context | 199 |
| color+context+semantic+shape+size | 185 |
| color+semantic+shape | 183 |
| color+roof+semantic+size | 179 |
| color+context+roof+semantic+shape+size | 171 |
| context | 166 |
| semantic+size | 112 |
| context+semantic+shape+size | 100 |
| context+roof+semantic | 83 |
| color+roof+semantic+shape | 83 |
| color+semantic+shape+size | 75 |
| semantic+shape | 75 |
| color+context+size | 68 |
| context+size | 68 |
| color+context+roof | 48 |
| context+roof+semantic+size | 40 |
| color+roof+semantic+shape+size | 38 |

## Referenced-landmark co-occurrence

| attribute | occurrences with ≥1 referenced landmark | all occurrences | rate |
|---|---:|---:|---:|
| color:white | 9,939 | 9,968 | 99.71% |
| color:blue | 5,833 | 5,848 | 99.74% |
| size:large | 2,971 | 2,983 | 99.60% |
| context:parking_lot | 8,126 | 8,151 | 99.69% |
| semantic:building | 15,763 | 15,792 | 99.82% |
| semantic:parking | 7,318 | 7,341 | 99.69% |
| color:gray | 10,193 | 10,223 | 99.71% |
| color:brown | 4,388 | 4,403 | 99.66% |
| color:dark | 2,412 | 2,416 | 99.83% |
| roof:roof | 5,776 | 5,800 | 99.59% |
| context:road | 16,637 | 16,662 | 99.85% |
| semantic:house | 4,226 | 4,241 | 99.65% |
| semantic:car | 15,832 | 15,874 | 99.74% |
| context:railway | 77 | 77 | 100.00% |
| semantic:church | 1,061 | 1,061 | 100.00% |
| size:short | 300 | 304 | 98.68% |
| shape:l_shaped | 286 | 287 | 99.65% |
| shape:rectangular | 866 | 871 | 99.43% |
| color:red | 4,688 | 4,703 | 99.68% |
| color:black | 6,633 | 6,659 | 99.61% |
| color:yellow | 712 | 714 | 99.72% |
| shape:round | 293 | 293 | 100.00% |
| context:intersection | 1,568 | 1,568 | 100.00% |
| color:light | 1,489 | 1,496 | 99.53% |
| shape:square | 592 | 593 | 99.83% |
| color:bright | 241 | 241 | 100.00% |
| color:green | 1,572 | 1,574 | 99.87% |
| context:vegetation | 2,603 | 2,615 | 99.54% |
| size:small | 1,675 | 1,681 | 99.64% |
| size:long | 1,761 | 1,769 | 99.55% |
| context:grass | 1,047 | 1,049 | 99.81% |
| size:tall | 145 | 145 | 100.00% |
| context:water | 83 | 84 | 98.81% |
| roof:flat_roof | 71 | 71 | 100.00% |
| context:crossing | 16 | 16 | 100.00% |
| context:river | 197 | 198 | 99.49% |
| semantic:ground | 1,001 | 1,005 | 99.60% |
| context:open_area | 26 | 27 | 96.30% |
| size:wide | 17 | 18 | 94.44% |
| size:narrow | 223 | 224 | 99.55% |
| semantic:school | 452 | 452 | 100.00% |
| roof:pitched_roof | 17 | 17 | 100.00% |
| shape:courtyard | 173 | 173 | 100.00% |
| context:field | 233 | 234 | 99.57% |
| context:sports_field | 12 | 12 | 100.00% |
| context:bridge | 498 | 498 | 100.00% |
| shape:irregular | 2 | 2 | 100.00% |
| shape:u_shaped | 38 | 38 | 100.00% |
| semantic:office | 34 | 34 | 100.00% |
| semantic:tower | 48 | 48 | 100.00% |
| semantic:college | 1,390 | 1,390 | 100.00% |
| semantic:library | 645 | 645 | 100.00% |
| semantic:hospital | 1 | 1 | 100.00% |
| shape:elongated | 1 | 1 | 100.00% |

Counts are lexical matches, not claims of visual observability. Observability is assigned only after held-out image evaluation.
