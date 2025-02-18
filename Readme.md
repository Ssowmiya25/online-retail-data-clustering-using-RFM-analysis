## Online Retail Data Clustering Project uaing RFM analysis

### Data Description 

- This Online Retail II data set contains all the transactions occurring for a UK-based and registered, non-store online retail between 01/12/2009 and 09/12/2011.The company mainly sells unique all-occasion gift-ware. Many customers of the company are wholesalers [Dataset link](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

### Data Exploration

##### Import Necessary Libraries

``` python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

pd.options.display.float_format = '{:20.2f}'.format

pd.set_option('display.max_columns',999)
```
![Screenshot](./Image/dataset.png)
