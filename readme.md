# Using Course selection records to solve cold start problem of the recommender system on academic library books dataset

Recommend academic library book for cold-started users.

Note that we focus on the __thousand categories__(prefix-3) of 
*Chinese Classification* (https://catweb.ncl.edu.tw/class2007/96-1-1.htm) to reduce the 
sparsity of our dataset.
 

**Due to privacy issue, we cannot public the course selection data and book lending data.**

## Execution order:

### 1. ```preprocessing.ipynb```: 
It's for data preprocessing, including data mining and cleaning.

### 2. ```exploration.ipynb```: 

To give the evidences to show that __course selection records__ and __book lending records__ have some relationship.

Note that dense matrices ```usercoursedense``` and ```cate3_userdense``` are constructed during exploration part.

### 3. ```splitdata.ipynb``` :

split the data into training and testing part.

Note that it just splits the dataset but doesn't
apply any transformation to the data. 
(e.g. normalization)

### 4. ```recommendation.ipynb```:

Doing recommendation.


## recommendation strategy :

### recommend by using popluar books

recommend all cold-started users 
by using popular categories directly.

**do it in ```recommendation.ipynb```**

in the folder ```RS```

### ```courseEmd.py```(our method)

It recommends cold start users by using the records of the training users who had taken the common courses to them.  

**Can execute it directly, or call it inside ```recommendation.ipynb```**

### ```MatrixFactorization```

Reference: 

__Improving Top-N Recommendation for Cold-Start Users via
Cross-Domain Information__
(NIMA MIRBAKHSH and CHARLES X. LING, Western University)

(https://dl.acm.org/doi/10.1145/2724720)

our domain: 
- book lending records 
- course selection records.

**Can execute ```mf_based.py``` directly, or call it in ```recommendation.ipynb```**

## Kits file:
in the folder ```RS.utils```

### ```mf.py``` :
for matrix factorization by using 
alernative gradient descent

### ```evaluation.py```:

caculating the metrices $\text{precision},\text{recall},\text{F1-score}$ and plotting ROC, PR curve.

### ```dictutils.py```

I/O for json file by using python native json package.

### ```plotutils.py```

Doing visualization by using ```matplotlib.pyplot``` 

### ```dataset.py```

Some process tool to our dataset

## The visualization can be found at the director: ./result 