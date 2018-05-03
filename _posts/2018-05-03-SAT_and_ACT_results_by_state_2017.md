
# Project 1

    Pre-step: load our libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
```

## Step 1: Load the data and perform basic operations.

##### 1. Load the data in using pandas.


```python
sat = pd.read_csv('../data/sat.csv')
act = pd.read_csv('../data/act.csv')
```

##### 2. Print the first ten rows of each dataframe.


```python
print(sat.head(10))
print("\n\n")
print(act.head(10))
```

       Unnamed: 0                 State Participation  \
    0           0               Alabama            5%   
    1           1                Alaska           38%   
    2           2               Arizona           30%   
    3           3              Arkansas            3%   
    4           4            California           53%   
    5           5              Colorado           11%   
    6           6           Connecticut          100%   
    7           7              Delaware          100%   
    8           8  District of Columbia          100%   
    9           9               Florida           83%   
    
       Evidence-Based Reading and Writing  Math  Total  
    0                                 593   572   1165  
    1                                 547   533   1080  
    2                                 563   553   1116  
    3                                 614   594   1208  
    4                                 531   524   1055  
    5                                 606   595   1201  
    6                                 530   512   1041  
    7                                 503   492    996  
    8                                 482   468    950  
    9                                 520   497   1017  
    
    
    
       Unnamed: 0                 State Participation  English  Math  Reading  \
    0           0              National           60%     20.3  20.7     21.4   
    1           1               Alabama          100%     18.9  18.4     19.7   
    2           2                Alaska           65%     18.7  19.8     20.4   
    3           3               Arizona           62%     18.6  19.8     20.1   
    4           4              Arkansas          100%     18.9  19.0     19.7   
    5           5            California           31%     22.5  22.7     23.1   
    6           6              Colorado          100%     20.1  20.3     21.2   
    7           7           Connecticut           31%     25.5  24.6     25.6   
    8           8              Delaware           18%     24.1  23.4     24.8   
    9           9  District of Columbia           32%     24.4  23.5     24.9   
    
       Science  Composite  
    0     21.0       21.0  
    1     19.4       19.2  
    2     19.9       19.8  
    3     19.8       19.7  
    4     19.5       19.4  
    5     22.2       22.8  
    6     20.9       20.8  
    7     24.6       25.2  
    8     23.6       24.1  
    9     23.5       24.2  


##### 3. Describe in words what each variable (column) is.


```python
'''
SAT:
Unnamed: 0 - index column from source
State - US State
Participation - percentage of high school seniors in the state who took the SAT in 2017
Evidence-Based Reading and Writing - combined average Reading and Writing and Language scores for students from the state who took the SATs in 2017
Math - average Match scores for students from the state who took the SAT in 2017
Total - total average SAT scores for students from the state who took the SAT in 2017

source: https://blog.prepscholar.com/average-sat-scores-by-state-most-recent 
based on https://reports.collegeboard.org/sat-suite-program-results/detailed-2017-reports

ACT:
Unnamed: 0 - index column from source
State: - US State
Participation: - percentage of high school seniors in the state who took the ACT in 2017
English: - average ACT English score (1-36 scale) for students from the state who took the ACT in 2017
Math: average ACT math score (1-36 scale) for students from the state who took the ACT in 2017
Reading: average ACT reading score (1-36 scale) for students from the state who took the ACT in 2017
Science: average ACT science score (1-36 scale) for students from the state who took the ACT in 2017
Composite: average ACT composite score (1-36 scale) for students from the state who took the ACT in 2017

Source: https://blog.prepscholar.com/act-scores-by-state-averages-highs-and-lows
based on http://www.act.org/content/act/en/research/condition-of-college-and-career-readiness-2017.html 


'''
```




    '\nSAT:\nUnnamed: 0 - index column from source\nState - US State\nParticipation - percentage of high school seniors in the state who took the SAT in 2017\nEvidence-Based Reading and Writing - combined average Reading and Writing and Language scores for students from the state who took the SATs in 2017\nMath - average Match scores for students from the state who took the SAT in 2017\nTotal - total average SAT scores for students from the state who took the SAT in 2017\n\nsource: https://blog.prepscholar.com/average-sat-scores-by-state-most-recent \nbased on https://reports.collegeboard.org/sat-suite-program-results/detailed-2017-reports\n\nACT:\nUnnamed: 0 - index column from source\nState: - US State\nParticipation: - percentage of high school seniors in the state who took the ACT in 2017\nEnglish: - average ACT English score (1-36 scale) for students from the state who took the ACT in 2017\nMath: average ACT math score (1-36 scale) for students from the state who took the ACT in 2017\nReading: average ACT reading score (1-36 scale) for students from the state who took the ACT in 2017\nScience: average ACT science score (1-36 scale) for students from the state who took the ACT in 2017\nComposite: average ACT composite score (1-36 scale) for students from the state who took the ACT in 2017\n\nSource: https://blog.prepscholar.com/act-scores-by-state-averages-highs-and-lows\nbased on http://www.act.org/content/act/en/research/condition-of-college-and-career-readiness-2017.html \n\n\n'



##### 4. Does the data look complete? Are there any obvious issues with the observations?


```python
# checking for null values
print(f"SAT null values: {sat.isnull().sum().sum() } - ACT null values: {act.isnull().sum().sum()}")

print(f"SAT shape: {sat.shape} \nACT shape: {act.shape}")
```

    SAT null values: 0 - ACT null values: 0
    SAT shape: (51, 6) 
    ACT shape: (52, 8)


##### 5. Print the types of each column.


```python
# checking data types
print("SAT\n", sat.dtypes) 
print("\nACT\n", act.dtypes)

# looks like maybe we should convert participation to a float in both datasets
```

    SAT
     Unnamed: 0                             int64
    State                                 object
    Participation                         object
    Evidence-Based Reading and Writing     int64
    Math                                   int64
    Total                                  int64
    dtype: object
    
    ACT
     Unnamed: 0         int64
    State             object
    Participation     object
    English          float64
    Math             float64
    Reading          float64
    Science          float64
    Composite        float64
    dtype: object


##### 6. Do any types need to be reassigned? If so, go ahead and do it.


```python
# Drop junk columns (Unnamed=0)
#df.drop([col name], axis=1, inplace=True)
sat.drop('Unnamed: 0', axis=1, inplace=True)
act.drop('Unnamed: 0', axis=1, inplace=True)


```


```python
# Convert "Participation" values to floats (SAT, ACT)
# model: boston.DIS = boston.DIS.map(lambda x: float(x.replace(',', '.')))
# need to divide values by 100 since they were percents!
sat.Participation = sat.Participation.map(lambda x: float(x.strip('%'))/100)
act.Participation = act.Participation.map(lambda x: float(x.strip('%'))/100)

```

##### 7. Create a dictionary for each column mapping the State to its respective value for that column. (For example, you should have three SAT dictionaries.)


```python
# setting State as our index first and then doing a Series dictionary
sat.set_index('State').to_dict('Series')

act.set_index('State').to_dict('Series')
```




    {'Composite': State
     National                21.0
     Alabama                 19.2
     Alaska                  19.8
     Arizona                 19.7
     Arkansas                19.4
     California              22.8
     Colorado                20.8
     Connecticut             25.2
     Delaware                24.1
     District of Columbia    24.2
     Florida                 19.8
     Georgia                 21.4
     Hawaii                  19.0
     Idaho                   22.3
     Illinois                21.4
     Indiana                 22.6
     Iowa                    21.9
     Kansas                  21.7
     Kentucky                20.0
     Louisiana               19.5
     Maine                   24.3
     Maryland                23.6
     Massachusetts           25.4
     Michigan                24.1
     Minnesota               21.5
     Mississippi             18.6
     Missouri                20.4
     Montana                 20.3
     Nebraska                21.4
     Nevada                  17.8
     New Hampshire           25.5
     New Jersey              23.9
     New Mexico              19.7
     New York                24.2
     North Carolina          19.1
     North Dakota            20.3
     Ohio                    22.0
     Oklahoma                19.4
     Oregon                  21.8
     Pennsylvania            23.7
     Rhode Island            24.0
     South Carolina          18.7
     South Dakota            21.8
     Tennessee               19.8
     Texas                   20.7
     Utah                    20.3
     Vermont                 23.6
     Virginia                23.8
     Washington              21.9
     West Virginia           20.4
     Wisconsin               20.5
     Wyoming                 20.2
     Name: Composite, dtype: float64, 'English': State
     National                20.3
     Alabama                 18.9
     Alaska                  18.7
     Arizona                 18.6
     Arkansas                18.9
     California              22.5
     Colorado                20.1
     Connecticut             25.5
     Delaware                24.1
     District of Columbia    24.4
     Florida                 19.0
     Georgia                 21.0
     Hawaii                  17.8
     Idaho                   21.9
     Illinois                21.0
     Indiana                 22.0
     Iowa                    21.2
     Kansas                  21.1
     Kentucky                19.6
     Louisiana               19.4
     Maine                   24.2
     Maryland                23.3
     Massachusetts           25.4
     Michigan                24.1
     Minnesota               20.4
     Mississippi             18.2
     Missouri                19.8
     Montana                 19.0
     Nebraska                20.9
     Nevada                  16.3
     New Hampshire           25.4
     New Jersey              23.8
     New Mexico              18.6
     New York                23.8
     North Carolina          17.8
     North Dakota            19.0
     Ohio                    21.2
     Oklahoma                18.5
     Oregon                  21.2
     Pennsylvania            23.4
     Rhode Island            24.0
     South Carolina          17.5
     South Dakota            20.7
     Tennessee               19.5
     Texas                   19.5
     Utah                    19.5
     Vermont                 23.3
     Virginia                23.5
     Washington              20.9
     West Virginia           20.0
     Wisconsin               19.7
     Wyoming                 19.4
     Name: English, dtype: float64, 'Math': State
     National                20.7
     Alabama                 18.4
     Alaska                  19.8
     Arizona                 19.8
     Arkansas                19.0
     California              22.7
     Colorado                20.3
     Connecticut             24.6
     Delaware                23.4
     District of Columbia    23.5
     Florida                 19.4
     Georgia                 20.9
     Hawaii                  19.2
     Idaho                   21.8
     Illinois                21.2
     Indiana                 22.4
     Iowa                    21.3
     Kansas                  21.3
     Kentucky                19.4
     Louisiana               18.8
     Maine                   24.0
     Maryland                23.1
     Massachusetts           25.3
     Michigan                23.7
     Minnesota               21.5
     Mississippi             18.1
     Missouri                19.9
     Montana                 20.2
     Nebraska                20.9
     Nevada                  18.0
     New Hampshire           25.1
     New Jersey              23.8
     New Mexico              19.4
     New York                24.0
     North Carolina          19.3
     North Dakota            20.4
     Ohio                    21.6
     Oklahoma                18.8
     Oregon                  21.5
     Pennsylvania            23.4
     Rhode Island            23.3
     South Carolina          18.6
     South Dakota            21.5
     Tennessee               19.2
     Texas                   20.7
     Utah                    19.9
     Vermont                 23.1
     Virginia                23.3
     Washington              21.9
     West Virginia           19.4
     Wisconsin               20.4
     Wyoming                 19.8
     Name: Math, dtype: float64, 'Participation': State
     National                0.60
     Alabama                 1.00
     Alaska                  0.65
     Arizona                 0.62
     Arkansas                1.00
     California              0.31
     Colorado                1.00
     Connecticut             0.31
     Delaware                0.18
     District of Columbia    0.32
     Florida                 0.73
     Georgia                 0.55
     Hawaii                  0.90
     Idaho                   0.38
     Illinois                0.93
     Indiana                 0.35
     Iowa                    0.67
     Kansas                  0.73
     Kentucky                1.00
     Louisiana               1.00
     Maine                   0.08
     Maryland                0.28
     Massachusetts           0.29
     Michigan                0.29
     Minnesota               1.00
     Mississippi             1.00
     Missouri                1.00
     Montana                 1.00
     Nebraska                0.84
     Nevada                  1.00
     New Hampshire           0.18
     New Jersey              0.34
     New Mexico              0.66
     New York                0.31
     North Carolina          1.00
     North Dakota            0.98
     Ohio                    0.75
     Oklahoma                1.00
     Oregon                  0.40
     Pennsylvania            0.23
     Rhode Island            0.21
     South Carolina          1.00
     South Dakota            0.80
     Tennessee               1.00
     Texas                   0.45
     Utah                    1.00
     Vermont                 0.29
     Virginia                0.29
     Washington              0.29
     West Virginia           0.69
     Wisconsin               1.00
     Wyoming                 1.00
     Name: Participation, dtype: float64, 'Reading': State
     National                21.4
     Alabama                 19.7
     Alaska                  20.4
     Arizona                 20.1
     Arkansas                19.7
     California              23.1
     Colorado                21.2
     Connecticut             25.6
     Delaware                24.8
     District of Columbia    24.9
     Florida                 21.0
     Georgia                 22.0
     Hawaii                  19.2
     Idaho                   23.0
     Illinois                21.6
     Indiana                 23.2
     Iowa                    22.6
     Kansas                  22.3
     Kentucky                20.5
     Louisiana               19.8
     Maine                   24.8
     Maryland                24.2
     Massachusetts           25.9
     Michigan                24.5
     Minnesota               21.8
     Mississippi             18.8
     Missouri                20.8
     Montana                 21.0
     Nebraska                21.9
     Nevada                  18.1
     New Hampshire           26.0
     New Jersey              24.1
     New Mexico              20.4
     New York                24.6
     North Carolina          19.6
     North Dakota            20.5
     Ohio                    22.5
     Oklahoma                20.1
     Oregon                  22.4
     Pennsylvania            24.2
     Rhode Island            24.7
     South Carolina          19.1
     South Dakota            22.3
     Tennessee               20.1
     Texas                   21.1
     Utah                    20.8
     Vermont                 24.4
     Virginia                24.6
     Washington              22.1
     West Virginia           21.2
     Wisconsin               20.6
     Wyoming                 20.8
     Name: Reading, dtype: float64, 'Science': State
     National                21.0
     Alabama                 19.4
     Alaska                  19.9
     Arizona                 19.8
     Arkansas                19.5
     California              22.2
     Colorado                20.9
     Connecticut             24.6
     Delaware                23.6
     District of Columbia    23.5
     Florida                 19.4
     Georgia                 21.3
     Hawaii                  19.3
     Idaho                   22.1
     Illinois                21.3
     Indiana                 22.3
     Iowa                    22.1
     Kansas                  21.7
     Kentucky                20.1
     Louisiana               19.6
     Maine                   23.7
     Maryland                 2.3
     Massachusetts           24.7
     Michigan                23.8
     Minnesota               21.6
     Mississippi             18.8
     Missouri                20.5
     Montana                 20.5
     Nebraska                21.5
     Nevada                  18.2
     New Hampshire           24.9
     New Jersey              23.2
     New Mexico              20.0
     New York                23.9
     North Carolina          19.3
     North Dakota            20.6
     Ohio                    22.0
     Oklahoma                19.6
     Oregon                  21.7
     Pennsylvania            23.3
     Rhode Island            23.4
     South Carolina          18.9
     South Dakota            22.0
     Tennessee               19.9
     Texas                   20.9
     Utah                    20.6
     Vermont                 23.2
     Virginia                23.5
     Washington              22.0
     West Virginia           20.5
     Wisconsin               20.9
     Wyoming                 20.6
     Name: Science, dtype: float64}



##### 8. Create one dictionary where each key is the column name, and each value is an iterable (a list or a Pandas Series) of all the values in that column.


```python
sat.to_dict('list')
```




    {'Evidence-Based Reading and Writing': [593,
      547,
      563,
      614,
      531,
      606,
      530,
      503,
      482,
      520,
      535,
      544,
      513,
      559,
      542,
      641,
      632,
      631,
      611,
      513,
      536,
      555,
      509,
      644,
      634,
      640,
      605,
      629,
      563,
      532,
      530,
      577,
      528,
      546,
      635,
      578,
      530,
      560,
      540,
      539,
      543,
      612,
      623,
      513,
      624,
      562,
      561,
      541,
      558,
      642,
      626],
     'Math': [572,
      533,
      553,
      594,
      524,
      595,
      512,
      492,
      468,
      497,
      515,
      541,
      493,
      556,
      532,
      635,
      628,
      616,
      586,
      499,
      52,
      551,
      495,
      651,
      607,
      631,
      591,
      625,
      553,
      520,
      526,
      561,
      523,
      535,
      621,
      570,
      517,
      548,
      531,
      524,
      521,
      603,
      604,
      507,
      614,
      551,
      541,
      534,
      528,
      649,
      604],
     'Participation': [0.05,
      0.38,
      0.3,
      0.03,
      0.53,
      0.11,
      1.0,
      1.0,
      1.0,
      0.83,
      0.61,
      0.55,
      0.93,
      0.09,
      0.63,
      0.02,
      0.04,
      0.04,
      0.04,
      0.95,
      0.69,
      0.76,
      1.0,
      0.03,
      0.02,
      0.03,
      0.1,
      0.03,
      0.26,
      0.96,
      0.7,
      0.11,
      0.67,
      0.49,
      0.02,
      0.12,
      0.07,
      0.43,
      0.65,
      0.71,
      0.5,
      0.03,
      0.05,
      0.62,
      0.03,
      0.6,
      0.65,
      0.64,
      0.14,
      0.03,
      0.03],
     'State': ['Alabama',
      'Alaska',
      'Arizona',
      'Arkansas',
      'California',
      'Colorado',
      'Connecticut',
      'Delaware',
      'District of Columbia',
      'Florida',
      'Georgia',
      'Hawaii',
      'Idaho',
      'Illinois',
      'Indiana',
      'Iowa',
      'Kansas',
      'Kentucky',
      'Louisiana',
      'Maine',
      'Maryland',
      'Massachusetts',
      'Michigan',
      'Minnesota',
      'Mississippi',
      'Missouri',
      'Montana',
      'Nebraska',
      'Nevada',
      'New Hampshire',
      'New Jersey',
      'New Mexico',
      'New York',
      'North Carolina',
      'North Dakota',
      'Ohio',
      'Oklahoma',
      'Oregon',
      'Pennsylvania',
      'Rhode Island',
      'South Carolina',
      'South Dakota',
      'Tennessee',
      'Texas',
      'Utah',
      'Vermont',
      'Virginia',
      'Washington',
      'West Virginia',
      'Wisconsin',
      'Wyoming'],
     'Total': [1165,
      1080,
      1116,
      1208,
      1055,
      1201,
      1041,
      996,
      950,
      1017,
      1050,
      1085,
      1005,
      1115,
      1074,
      1275,
      1260,
      1247,
      1198,
      1012,
      1060,
      1107,
      1005,
      1295,
      1242,
      1271,
      1196,
      1253,
      1116,
      1052,
      1056,
      1138,
      1052,
      1081,
      1256,
      1149,
      1047,
      1108,
      1071,
      1062,
      1064,
      1216,
      1228,
      1020,
      1238,
      1114,
      1102,
      1075,
      1086,
      1291,
      1230]}



##### 9. Merge the dataframes on the state column.


```python
# Making a merged dataframe joining on 'State'
merged = pd.merge(sat, act, on="State", suffixes=('_sat', '_act'))




```

##### 10. Change the names of the columns so you can distinguish between the SAT columns and the ACT columns.


```python
# Specifying sat and act in every column
merged = merged.rename(columns={'Evidence-Based Reading and Writing': 'Reading_writing_sat', 
             'Total':'Total_sat', 'English':'English_act', 
             'Reading':'Reading_act', 
             'Science': 'Science_act', 
             'Composite':'Composite_act'})
```

##### 11. Print the minimum and maximum of each numeric column in the data frame.


```python
# This gives us min and max for every column

numeric_columns = ['Reading_writing_sat', 'Math_sat', 'Total_sat', 'English_act', 'Math_act', 'Reading_act', 'Science_act', 'Composite_act']
merged[numeric_columns].aggregate([np.min, np.max])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Reading_writing_sat</th>
      <th>Math_sat</th>
      <th>Total_sat</th>
      <th>English_act</th>
      <th>Math_act</th>
      <th>Reading_act</th>
      <th>Science_act</th>
      <th>Composite_act</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>amin</th>
      <td>482</td>
      <td>52</td>
      <td>950</td>
      <td>16.3</td>
      <td>18.0</td>
      <td>18.1</td>
      <td>2.3</td>
      <td>17.8</td>
    </tr>
    <tr>
      <th>amax</th>
      <td>644</td>
      <td>651</td>
      <td>1295</td>
      <td>25.5</td>
      <td>25.3</td>
      <td>26.0</td>
      <td>24.9</td>
      <td>25.5</td>
    </tr>
  </tbody>
</table>
</div>



##### 12. Write a function using only list comprehensions, no loops, to compute standard deviation. Using this function, calculate the standard deviation of each numeric column in both data sets. Add these to a list called `sd`.

$$\sigma = \sqrt{\frac{1}{n}\sum_{i=1}^n(x_i - \mu)^2}$$


```python
# helped by: https://codereview.stackexchange.com/questions/9222/calculating-population-standard-deviation

def std_dev(series):
    mean = np.mean(series)
    n=len(series)
    sum_of_squares = sum([(item-mean) ** 2 for item in series])
    std = (sum_of_squares/len(series))**0.5
    return std

numeric_columns = ['Participation_sat', 'Reading_writing_sat', 'Math_sat', 'Total_sat', 'Participation_act', 'English_act', 'Math_act', 'Reading_act', 'Science_act', 'Composite_act']

sd = [std_dev(merged[item]) for item in numeric_columns]

```

## Step 2: Manipulate the dataframe

##### 13. Turn the list `sd` into a new observation in your dataset.


```python
# make a dictionary of colnames with vals 
# dictionary = dict(zip(keys, values)) 
new_row = dict(zip(numeric_columns, sd))

# try to add that to the df and fill in the missing values (?)
merged.append(new_row, ignore_index=True)

#np.std(merged['Total_sat'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Participation_sat</th>
      <th>Reading_writing_sat</th>
      <th>Math_sat</th>
      <th>Total_sat</th>
      <th>Participation_act</th>
      <th>English_act</th>
      <th>Math_act</th>
      <th>Reading_act</th>
      <th>Science_act</th>
      <th>Composite_act</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>0.050000</td>
      <td>593.00000</td>
      <td>572.000000</td>
      <td>1165.000000</td>
      <td>1.000000</td>
      <td>18.900000</td>
      <td>18.400000</td>
      <td>19.700000</td>
      <td>19.400000</td>
      <td>19.200000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alaska</td>
      <td>0.380000</td>
      <td>547.00000</td>
      <td>533.000000</td>
      <td>1080.000000</td>
      <td>0.650000</td>
      <td>18.700000</td>
      <td>19.800000</td>
      <td>20.400000</td>
      <td>19.900000</td>
      <td>19.800000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arizona</td>
      <td>0.300000</td>
      <td>563.00000</td>
      <td>553.000000</td>
      <td>1116.000000</td>
      <td>0.620000</td>
      <td>18.600000</td>
      <td>19.800000</td>
      <td>20.100000</td>
      <td>19.800000</td>
      <td>19.700000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>0.030000</td>
      <td>614.00000</td>
      <td>594.000000</td>
      <td>1208.000000</td>
      <td>1.000000</td>
      <td>18.900000</td>
      <td>19.000000</td>
      <td>19.700000</td>
      <td>19.500000</td>
      <td>19.400000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>California</td>
      <td>0.530000</td>
      <td>531.00000</td>
      <td>524.000000</td>
      <td>1055.000000</td>
      <td>0.310000</td>
      <td>22.500000</td>
      <td>22.700000</td>
      <td>23.100000</td>
      <td>22.200000</td>
      <td>22.800000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Colorado</td>
      <td>0.110000</td>
      <td>606.00000</td>
      <td>595.000000</td>
      <td>1201.000000</td>
      <td>1.000000</td>
      <td>20.100000</td>
      <td>20.300000</td>
      <td>21.200000</td>
      <td>20.900000</td>
      <td>20.800000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Connecticut</td>
      <td>1.000000</td>
      <td>530.00000</td>
      <td>512.000000</td>
      <td>1041.000000</td>
      <td>0.310000</td>
      <td>25.500000</td>
      <td>24.600000</td>
      <td>25.600000</td>
      <td>24.600000</td>
      <td>25.200000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Delaware</td>
      <td>1.000000</td>
      <td>503.00000</td>
      <td>492.000000</td>
      <td>996.000000</td>
      <td>0.180000</td>
      <td>24.100000</td>
      <td>23.400000</td>
      <td>24.800000</td>
      <td>23.600000</td>
      <td>24.100000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>District of Columbia</td>
      <td>1.000000</td>
      <td>482.00000</td>
      <td>468.000000</td>
      <td>950.000000</td>
      <td>0.320000</td>
      <td>24.400000</td>
      <td>23.500000</td>
      <td>24.900000</td>
      <td>23.500000</td>
      <td>24.200000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Florida</td>
      <td>0.830000</td>
      <td>520.00000</td>
      <td>497.000000</td>
      <td>1017.000000</td>
      <td>0.730000</td>
      <td>19.000000</td>
      <td>19.400000</td>
      <td>21.000000</td>
      <td>19.400000</td>
      <td>19.800000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Georgia</td>
      <td>0.610000</td>
      <td>535.00000</td>
      <td>515.000000</td>
      <td>1050.000000</td>
      <td>0.550000</td>
      <td>21.000000</td>
      <td>20.900000</td>
      <td>22.000000</td>
      <td>21.300000</td>
      <td>21.400000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Hawaii</td>
      <td>0.550000</td>
      <td>544.00000</td>
      <td>541.000000</td>
      <td>1085.000000</td>
      <td>0.900000</td>
      <td>17.800000</td>
      <td>19.200000</td>
      <td>19.200000</td>
      <td>19.300000</td>
      <td>19.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Idaho</td>
      <td>0.930000</td>
      <td>513.00000</td>
      <td>493.000000</td>
      <td>1005.000000</td>
      <td>0.380000</td>
      <td>21.900000</td>
      <td>21.800000</td>
      <td>23.000000</td>
      <td>22.100000</td>
      <td>22.300000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Illinois</td>
      <td>0.090000</td>
      <td>559.00000</td>
      <td>556.000000</td>
      <td>1115.000000</td>
      <td>0.930000</td>
      <td>21.000000</td>
      <td>21.200000</td>
      <td>21.600000</td>
      <td>21.300000</td>
      <td>21.400000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Indiana</td>
      <td>0.630000</td>
      <td>542.00000</td>
      <td>532.000000</td>
      <td>1074.000000</td>
      <td>0.350000</td>
      <td>22.000000</td>
      <td>22.400000</td>
      <td>23.200000</td>
      <td>22.300000</td>
      <td>22.600000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Iowa</td>
      <td>0.020000</td>
      <td>641.00000</td>
      <td>635.000000</td>
      <td>1275.000000</td>
      <td>0.670000</td>
      <td>21.200000</td>
      <td>21.300000</td>
      <td>22.600000</td>
      <td>22.100000</td>
      <td>21.900000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Kansas</td>
      <td>0.040000</td>
      <td>632.00000</td>
      <td>628.000000</td>
      <td>1260.000000</td>
      <td>0.730000</td>
      <td>21.100000</td>
      <td>21.300000</td>
      <td>22.300000</td>
      <td>21.700000</td>
      <td>21.700000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Kentucky</td>
      <td>0.040000</td>
      <td>631.00000</td>
      <td>616.000000</td>
      <td>1247.000000</td>
      <td>1.000000</td>
      <td>19.600000</td>
      <td>19.400000</td>
      <td>20.500000</td>
      <td>20.100000</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Louisiana</td>
      <td>0.040000</td>
      <td>611.00000</td>
      <td>586.000000</td>
      <td>1198.000000</td>
      <td>1.000000</td>
      <td>19.400000</td>
      <td>18.800000</td>
      <td>19.800000</td>
      <td>19.600000</td>
      <td>19.500000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Maine</td>
      <td>0.950000</td>
      <td>513.00000</td>
      <td>499.000000</td>
      <td>1012.000000</td>
      <td>0.080000</td>
      <td>24.200000</td>
      <td>24.000000</td>
      <td>24.800000</td>
      <td>23.700000</td>
      <td>24.300000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Maryland</td>
      <td>0.690000</td>
      <td>536.00000</td>
      <td>52.000000</td>
      <td>1060.000000</td>
      <td>0.280000</td>
      <td>23.300000</td>
      <td>23.100000</td>
      <td>24.200000</td>
      <td>2.300000</td>
      <td>23.600000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Massachusetts</td>
      <td>0.760000</td>
      <td>555.00000</td>
      <td>551.000000</td>
      <td>1107.000000</td>
      <td>0.290000</td>
      <td>25.400000</td>
      <td>25.300000</td>
      <td>25.900000</td>
      <td>24.700000</td>
      <td>25.400000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Michigan</td>
      <td>1.000000</td>
      <td>509.00000</td>
      <td>495.000000</td>
      <td>1005.000000</td>
      <td>0.290000</td>
      <td>24.100000</td>
      <td>23.700000</td>
      <td>24.500000</td>
      <td>23.800000</td>
      <td>24.100000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Minnesota</td>
      <td>0.030000</td>
      <td>644.00000</td>
      <td>651.000000</td>
      <td>1295.000000</td>
      <td>1.000000</td>
      <td>20.400000</td>
      <td>21.500000</td>
      <td>21.800000</td>
      <td>21.600000</td>
      <td>21.500000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Mississippi</td>
      <td>0.020000</td>
      <td>634.00000</td>
      <td>607.000000</td>
      <td>1242.000000</td>
      <td>1.000000</td>
      <td>18.200000</td>
      <td>18.100000</td>
      <td>18.800000</td>
      <td>18.800000</td>
      <td>18.600000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Missouri</td>
      <td>0.030000</td>
      <td>640.00000</td>
      <td>631.000000</td>
      <td>1271.000000</td>
      <td>1.000000</td>
      <td>19.800000</td>
      <td>19.900000</td>
      <td>20.800000</td>
      <td>20.500000</td>
      <td>20.400000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Montana</td>
      <td>0.100000</td>
      <td>605.00000</td>
      <td>591.000000</td>
      <td>1196.000000</td>
      <td>1.000000</td>
      <td>19.000000</td>
      <td>20.200000</td>
      <td>21.000000</td>
      <td>20.500000</td>
      <td>20.300000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Nebraska</td>
      <td>0.030000</td>
      <td>629.00000</td>
      <td>625.000000</td>
      <td>1253.000000</td>
      <td>0.840000</td>
      <td>20.900000</td>
      <td>20.900000</td>
      <td>21.900000</td>
      <td>21.500000</td>
      <td>21.400000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Nevada</td>
      <td>0.260000</td>
      <td>563.00000</td>
      <td>553.000000</td>
      <td>1116.000000</td>
      <td>1.000000</td>
      <td>16.300000</td>
      <td>18.000000</td>
      <td>18.100000</td>
      <td>18.200000</td>
      <td>17.800000</td>
    </tr>
    <tr>
      <th>29</th>
      <td>New Hampshire</td>
      <td>0.960000</td>
      <td>532.00000</td>
      <td>520.000000</td>
      <td>1052.000000</td>
      <td>0.180000</td>
      <td>25.400000</td>
      <td>25.100000</td>
      <td>26.000000</td>
      <td>24.900000</td>
      <td>25.500000</td>
    </tr>
    <tr>
      <th>30</th>
      <td>New Jersey</td>
      <td>0.700000</td>
      <td>530.00000</td>
      <td>526.000000</td>
      <td>1056.000000</td>
      <td>0.340000</td>
      <td>23.800000</td>
      <td>23.800000</td>
      <td>24.100000</td>
      <td>23.200000</td>
      <td>23.900000</td>
    </tr>
    <tr>
      <th>31</th>
      <td>New Mexico</td>
      <td>0.110000</td>
      <td>577.00000</td>
      <td>561.000000</td>
      <td>1138.000000</td>
      <td>0.660000</td>
      <td>18.600000</td>
      <td>19.400000</td>
      <td>20.400000</td>
      <td>20.000000</td>
      <td>19.700000</td>
    </tr>
    <tr>
      <th>32</th>
      <td>New York</td>
      <td>0.670000</td>
      <td>528.00000</td>
      <td>523.000000</td>
      <td>1052.000000</td>
      <td>0.310000</td>
      <td>23.800000</td>
      <td>24.000000</td>
      <td>24.600000</td>
      <td>23.900000</td>
      <td>24.200000</td>
    </tr>
    <tr>
      <th>33</th>
      <td>North Carolina</td>
      <td>0.490000</td>
      <td>546.00000</td>
      <td>535.000000</td>
      <td>1081.000000</td>
      <td>1.000000</td>
      <td>17.800000</td>
      <td>19.300000</td>
      <td>19.600000</td>
      <td>19.300000</td>
      <td>19.100000</td>
    </tr>
    <tr>
      <th>34</th>
      <td>North Dakota</td>
      <td>0.020000</td>
      <td>635.00000</td>
      <td>621.000000</td>
      <td>1256.000000</td>
      <td>0.980000</td>
      <td>19.000000</td>
      <td>20.400000</td>
      <td>20.500000</td>
      <td>20.600000</td>
      <td>20.300000</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Ohio</td>
      <td>0.120000</td>
      <td>578.00000</td>
      <td>570.000000</td>
      <td>1149.000000</td>
      <td>0.750000</td>
      <td>21.200000</td>
      <td>21.600000</td>
      <td>22.500000</td>
      <td>22.000000</td>
      <td>22.000000</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Oklahoma</td>
      <td>0.070000</td>
      <td>530.00000</td>
      <td>517.000000</td>
      <td>1047.000000</td>
      <td>1.000000</td>
      <td>18.500000</td>
      <td>18.800000</td>
      <td>20.100000</td>
      <td>19.600000</td>
      <td>19.400000</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Oregon</td>
      <td>0.430000</td>
      <td>560.00000</td>
      <td>548.000000</td>
      <td>1108.000000</td>
      <td>0.400000</td>
      <td>21.200000</td>
      <td>21.500000</td>
      <td>22.400000</td>
      <td>21.700000</td>
      <td>21.800000</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Pennsylvania</td>
      <td>0.650000</td>
      <td>540.00000</td>
      <td>531.000000</td>
      <td>1071.000000</td>
      <td>0.230000</td>
      <td>23.400000</td>
      <td>23.400000</td>
      <td>24.200000</td>
      <td>23.300000</td>
      <td>23.700000</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Rhode Island</td>
      <td>0.710000</td>
      <td>539.00000</td>
      <td>524.000000</td>
      <td>1062.000000</td>
      <td>0.210000</td>
      <td>24.000000</td>
      <td>23.300000</td>
      <td>24.700000</td>
      <td>23.400000</td>
      <td>24.000000</td>
    </tr>
    <tr>
      <th>40</th>
      <td>South Carolina</td>
      <td>0.500000</td>
      <td>543.00000</td>
      <td>521.000000</td>
      <td>1064.000000</td>
      <td>1.000000</td>
      <td>17.500000</td>
      <td>18.600000</td>
      <td>19.100000</td>
      <td>18.900000</td>
      <td>18.700000</td>
    </tr>
    <tr>
      <th>41</th>
      <td>South Dakota</td>
      <td>0.030000</td>
      <td>612.00000</td>
      <td>603.000000</td>
      <td>1216.000000</td>
      <td>0.800000</td>
      <td>20.700000</td>
      <td>21.500000</td>
      <td>22.300000</td>
      <td>22.000000</td>
      <td>21.800000</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Tennessee</td>
      <td>0.050000</td>
      <td>623.00000</td>
      <td>604.000000</td>
      <td>1228.000000</td>
      <td>1.000000</td>
      <td>19.500000</td>
      <td>19.200000</td>
      <td>20.100000</td>
      <td>19.900000</td>
      <td>19.800000</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Texas</td>
      <td>0.620000</td>
      <td>513.00000</td>
      <td>507.000000</td>
      <td>1020.000000</td>
      <td>0.450000</td>
      <td>19.500000</td>
      <td>20.700000</td>
      <td>21.100000</td>
      <td>20.900000</td>
      <td>20.700000</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Utah</td>
      <td>0.030000</td>
      <td>624.00000</td>
      <td>614.000000</td>
      <td>1238.000000</td>
      <td>1.000000</td>
      <td>19.500000</td>
      <td>19.900000</td>
      <td>20.800000</td>
      <td>20.600000</td>
      <td>20.300000</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Vermont</td>
      <td>0.600000</td>
      <td>562.00000</td>
      <td>551.000000</td>
      <td>1114.000000</td>
      <td>0.290000</td>
      <td>23.300000</td>
      <td>23.100000</td>
      <td>24.400000</td>
      <td>23.200000</td>
      <td>23.600000</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Virginia</td>
      <td>0.650000</td>
      <td>561.00000</td>
      <td>541.000000</td>
      <td>1102.000000</td>
      <td>0.290000</td>
      <td>23.500000</td>
      <td>23.300000</td>
      <td>24.600000</td>
      <td>23.500000</td>
      <td>23.800000</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Washington</td>
      <td>0.640000</td>
      <td>541.00000</td>
      <td>534.000000</td>
      <td>1075.000000</td>
      <td>0.290000</td>
      <td>20.900000</td>
      <td>21.900000</td>
      <td>22.100000</td>
      <td>22.000000</td>
      <td>21.900000</td>
    </tr>
    <tr>
      <th>48</th>
      <td>West Virginia</td>
      <td>0.140000</td>
      <td>558.00000</td>
      <td>528.000000</td>
      <td>1086.000000</td>
      <td>0.690000</td>
      <td>20.000000</td>
      <td>19.400000</td>
      <td>21.200000</td>
      <td>20.500000</td>
      <td>20.400000</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Wisconsin</td>
      <td>0.030000</td>
      <td>642.00000</td>
      <td>649.000000</td>
      <td>1291.000000</td>
      <td>1.000000</td>
      <td>19.700000</td>
      <td>20.400000</td>
      <td>20.600000</td>
      <td>20.900000</td>
      <td>20.500000</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Wyoming</td>
      <td>0.030000</td>
      <td>626.00000</td>
      <td>604.000000</td>
      <td>1230.000000</td>
      <td>1.000000</td>
      <td>19.400000</td>
      <td>19.800000</td>
      <td>20.800000</td>
      <td>20.600000</td>
      <td>20.200000</td>
    </tr>
    <tr>
      <th>51</th>
      <td>NaN</td>
      <td>0.349291</td>
      <td>45.21697</td>
      <td>84.072555</td>
      <td>91.583511</td>
      <td>0.318242</td>
      <td>2.330488</td>
      <td>1.962462</td>
      <td>2.046903</td>
      <td>3.151108</td>
      <td>2.000786</td>
    </tr>
  </tbody>
</table>
</div>



##### 14. Sort the dataframe by the values in a numeric column (e.g. observations descending by SAT participation rate)


```python
merged.sort_values('Math_sat', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Participation_sat</th>
      <th>Reading_writing_sat</th>
      <th>Math_sat</th>
      <th>Total_sat</th>
      <th>Participation_act</th>
      <th>English_act</th>
      <th>Math_act</th>
      <th>Reading_act</th>
      <th>Science_act</th>
      <th>Composite_act</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23</th>
      <td>Minnesota</td>
      <td>0.03</td>
      <td>644</td>
      <td>651</td>
      <td>1295</td>
      <td>1.00</td>
      <td>20.4</td>
      <td>21.5</td>
      <td>21.8</td>
      <td>21.6</td>
      <td>21.5</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Wisconsin</td>
      <td>0.03</td>
      <td>642</td>
      <td>649</td>
      <td>1291</td>
      <td>1.00</td>
      <td>19.7</td>
      <td>20.4</td>
      <td>20.6</td>
      <td>20.9</td>
      <td>20.5</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Iowa</td>
      <td>0.02</td>
      <td>641</td>
      <td>635</td>
      <td>1275</td>
      <td>0.67</td>
      <td>21.2</td>
      <td>21.3</td>
      <td>22.6</td>
      <td>22.1</td>
      <td>21.9</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Missouri</td>
      <td>0.03</td>
      <td>640</td>
      <td>631</td>
      <td>1271</td>
      <td>1.00</td>
      <td>19.8</td>
      <td>19.9</td>
      <td>20.8</td>
      <td>20.5</td>
      <td>20.4</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Kansas</td>
      <td>0.04</td>
      <td>632</td>
      <td>628</td>
      <td>1260</td>
      <td>0.73</td>
      <td>21.1</td>
      <td>21.3</td>
      <td>22.3</td>
      <td>21.7</td>
      <td>21.7</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Nebraska</td>
      <td>0.03</td>
      <td>629</td>
      <td>625</td>
      <td>1253</td>
      <td>0.84</td>
      <td>20.9</td>
      <td>20.9</td>
      <td>21.9</td>
      <td>21.5</td>
      <td>21.4</td>
    </tr>
    <tr>
      <th>34</th>
      <td>North Dakota</td>
      <td>0.02</td>
      <td>635</td>
      <td>621</td>
      <td>1256</td>
      <td>0.98</td>
      <td>19.0</td>
      <td>20.4</td>
      <td>20.5</td>
      <td>20.6</td>
      <td>20.3</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Kentucky</td>
      <td>0.04</td>
      <td>631</td>
      <td>616</td>
      <td>1247</td>
      <td>1.00</td>
      <td>19.6</td>
      <td>19.4</td>
      <td>20.5</td>
      <td>20.1</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Utah</td>
      <td>0.03</td>
      <td>624</td>
      <td>614</td>
      <td>1238</td>
      <td>1.00</td>
      <td>19.5</td>
      <td>19.9</td>
      <td>20.8</td>
      <td>20.6</td>
      <td>20.3</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Mississippi</td>
      <td>0.02</td>
      <td>634</td>
      <td>607</td>
      <td>1242</td>
      <td>1.00</td>
      <td>18.2</td>
      <td>18.1</td>
      <td>18.8</td>
      <td>18.8</td>
      <td>18.6</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Tennessee</td>
      <td>0.05</td>
      <td>623</td>
      <td>604</td>
      <td>1228</td>
      <td>1.00</td>
      <td>19.5</td>
      <td>19.2</td>
      <td>20.1</td>
      <td>19.9</td>
      <td>19.8</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Wyoming</td>
      <td>0.03</td>
      <td>626</td>
      <td>604</td>
      <td>1230</td>
      <td>1.00</td>
      <td>19.4</td>
      <td>19.8</td>
      <td>20.8</td>
      <td>20.6</td>
      <td>20.2</td>
    </tr>
    <tr>
      <th>41</th>
      <td>South Dakota</td>
      <td>0.03</td>
      <td>612</td>
      <td>603</td>
      <td>1216</td>
      <td>0.80</td>
      <td>20.7</td>
      <td>21.5</td>
      <td>22.3</td>
      <td>22.0</td>
      <td>21.8</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Colorado</td>
      <td>0.11</td>
      <td>606</td>
      <td>595</td>
      <td>1201</td>
      <td>1.00</td>
      <td>20.1</td>
      <td>20.3</td>
      <td>21.2</td>
      <td>20.9</td>
      <td>20.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>0.03</td>
      <td>614</td>
      <td>594</td>
      <td>1208</td>
      <td>1.00</td>
      <td>18.9</td>
      <td>19.0</td>
      <td>19.7</td>
      <td>19.5</td>
      <td>19.4</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Montana</td>
      <td>0.10</td>
      <td>605</td>
      <td>591</td>
      <td>1196</td>
      <td>1.00</td>
      <td>19.0</td>
      <td>20.2</td>
      <td>21.0</td>
      <td>20.5</td>
      <td>20.3</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Louisiana</td>
      <td>0.04</td>
      <td>611</td>
      <td>586</td>
      <td>1198</td>
      <td>1.00</td>
      <td>19.4</td>
      <td>18.8</td>
      <td>19.8</td>
      <td>19.6</td>
      <td>19.5</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>0.05</td>
      <td>593</td>
      <td>572</td>
      <td>1165</td>
      <td>1.00</td>
      <td>18.9</td>
      <td>18.4</td>
      <td>19.7</td>
      <td>19.4</td>
      <td>19.2</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Ohio</td>
      <td>0.12</td>
      <td>578</td>
      <td>570</td>
      <td>1149</td>
      <td>0.75</td>
      <td>21.2</td>
      <td>21.6</td>
      <td>22.5</td>
      <td>22.0</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>New Mexico</td>
      <td>0.11</td>
      <td>577</td>
      <td>561</td>
      <td>1138</td>
      <td>0.66</td>
      <td>18.6</td>
      <td>19.4</td>
      <td>20.4</td>
      <td>20.0</td>
      <td>19.7</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Illinois</td>
      <td>0.09</td>
      <td>559</td>
      <td>556</td>
      <td>1115</td>
      <td>0.93</td>
      <td>21.0</td>
      <td>21.2</td>
      <td>21.6</td>
      <td>21.3</td>
      <td>21.4</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Nevada</td>
      <td>0.26</td>
      <td>563</td>
      <td>553</td>
      <td>1116</td>
      <td>1.00</td>
      <td>16.3</td>
      <td>18.0</td>
      <td>18.1</td>
      <td>18.2</td>
      <td>17.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arizona</td>
      <td>0.30</td>
      <td>563</td>
      <td>553</td>
      <td>1116</td>
      <td>0.62</td>
      <td>18.6</td>
      <td>19.8</td>
      <td>20.1</td>
      <td>19.8</td>
      <td>19.7</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Vermont</td>
      <td>0.60</td>
      <td>562</td>
      <td>551</td>
      <td>1114</td>
      <td>0.29</td>
      <td>23.3</td>
      <td>23.1</td>
      <td>24.4</td>
      <td>23.2</td>
      <td>23.6</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Massachusetts</td>
      <td>0.76</td>
      <td>555</td>
      <td>551</td>
      <td>1107</td>
      <td>0.29</td>
      <td>25.4</td>
      <td>25.3</td>
      <td>25.9</td>
      <td>24.7</td>
      <td>25.4</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Oregon</td>
      <td>0.43</td>
      <td>560</td>
      <td>548</td>
      <td>1108</td>
      <td>0.40</td>
      <td>21.2</td>
      <td>21.5</td>
      <td>22.4</td>
      <td>21.7</td>
      <td>21.8</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Virginia</td>
      <td>0.65</td>
      <td>561</td>
      <td>541</td>
      <td>1102</td>
      <td>0.29</td>
      <td>23.5</td>
      <td>23.3</td>
      <td>24.6</td>
      <td>23.5</td>
      <td>23.8</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Hawaii</td>
      <td>0.55</td>
      <td>544</td>
      <td>541</td>
      <td>1085</td>
      <td>0.90</td>
      <td>17.8</td>
      <td>19.2</td>
      <td>19.2</td>
      <td>19.3</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>North Carolina</td>
      <td>0.49</td>
      <td>546</td>
      <td>535</td>
      <td>1081</td>
      <td>1.00</td>
      <td>17.8</td>
      <td>19.3</td>
      <td>19.6</td>
      <td>19.3</td>
      <td>19.1</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Washington</td>
      <td>0.64</td>
      <td>541</td>
      <td>534</td>
      <td>1075</td>
      <td>0.29</td>
      <td>20.9</td>
      <td>21.9</td>
      <td>22.1</td>
      <td>22.0</td>
      <td>21.9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alaska</td>
      <td>0.38</td>
      <td>547</td>
      <td>533</td>
      <td>1080</td>
      <td>0.65</td>
      <td>18.7</td>
      <td>19.8</td>
      <td>20.4</td>
      <td>19.9</td>
      <td>19.8</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Indiana</td>
      <td>0.63</td>
      <td>542</td>
      <td>532</td>
      <td>1074</td>
      <td>0.35</td>
      <td>22.0</td>
      <td>22.4</td>
      <td>23.2</td>
      <td>22.3</td>
      <td>22.6</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Pennsylvania</td>
      <td>0.65</td>
      <td>540</td>
      <td>531</td>
      <td>1071</td>
      <td>0.23</td>
      <td>23.4</td>
      <td>23.4</td>
      <td>24.2</td>
      <td>23.3</td>
      <td>23.7</td>
    </tr>
    <tr>
      <th>48</th>
      <td>West Virginia</td>
      <td>0.14</td>
      <td>558</td>
      <td>528</td>
      <td>1086</td>
      <td>0.69</td>
      <td>20.0</td>
      <td>19.4</td>
      <td>21.2</td>
      <td>20.5</td>
      <td>20.4</td>
    </tr>
    <tr>
      <th>30</th>
      <td>New Jersey</td>
      <td>0.70</td>
      <td>530</td>
      <td>526</td>
      <td>1056</td>
      <td>0.34</td>
      <td>23.8</td>
      <td>23.8</td>
      <td>24.1</td>
      <td>23.2</td>
      <td>23.9</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Rhode Island</td>
      <td>0.71</td>
      <td>539</td>
      <td>524</td>
      <td>1062</td>
      <td>0.21</td>
      <td>24.0</td>
      <td>23.3</td>
      <td>24.7</td>
      <td>23.4</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>California</td>
      <td>0.53</td>
      <td>531</td>
      <td>524</td>
      <td>1055</td>
      <td>0.31</td>
      <td>22.5</td>
      <td>22.7</td>
      <td>23.1</td>
      <td>22.2</td>
      <td>22.8</td>
    </tr>
    <tr>
      <th>32</th>
      <td>New York</td>
      <td>0.67</td>
      <td>528</td>
      <td>523</td>
      <td>1052</td>
      <td>0.31</td>
      <td>23.8</td>
      <td>24.0</td>
      <td>24.6</td>
      <td>23.9</td>
      <td>24.2</td>
    </tr>
    <tr>
      <th>40</th>
      <td>South Carolina</td>
      <td>0.50</td>
      <td>543</td>
      <td>521</td>
      <td>1064</td>
      <td>1.00</td>
      <td>17.5</td>
      <td>18.6</td>
      <td>19.1</td>
      <td>18.9</td>
      <td>18.7</td>
    </tr>
    <tr>
      <th>29</th>
      <td>New Hampshire</td>
      <td>0.96</td>
      <td>532</td>
      <td>520</td>
      <td>1052</td>
      <td>0.18</td>
      <td>25.4</td>
      <td>25.1</td>
      <td>26.0</td>
      <td>24.9</td>
      <td>25.5</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Oklahoma</td>
      <td>0.07</td>
      <td>530</td>
      <td>517</td>
      <td>1047</td>
      <td>1.00</td>
      <td>18.5</td>
      <td>18.8</td>
      <td>20.1</td>
      <td>19.6</td>
      <td>19.4</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Georgia</td>
      <td>0.61</td>
      <td>535</td>
      <td>515</td>
      <td>1050</td>
      <td>0.55</td>
      <td>21.0</td>
      <td>20.9</td>
      <td>22.0</td>
      <td>21.3</td>
      <td>21.4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Connecticut</td>
      <td>1.00</td>
      <td>530</td>
      <td>512</td>
      <td>1041</td>
      <td>0.31</td>
      <td>25.5</td>
      <td>24.6</td>
      <td>25.6</td>
      <td>24.6</td>
      <td>25.2</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Texas</td>
      <td>0.62</td>
      <td>513</td>
      <td>507</td>
      <td>1020</td>
      <td>0.45</td>
      <td>19.5</td>
      <td>20.7</td>
      <td>21.1</td>
      <td>20.9</td>
      <td>20.7</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Maine</td>
      <td>0.95</td>
      <td>513</td>
      <td>499</td>
      <td>1012</td>
      <td>0.08</td>
      <td>24.2</td>
      <td>24.0</td>
      <td>24.8</td>
      <td>23.7</td>
      <td>24.3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Florida</td>
      <td>0.83</td>
      <td>520</td>
      <td>497</td>
      <td>1017</td>
      <td>0.73</td>
      <td>19.0</td>
      <td>19.4</td>
      <td>21.0</td>
      <td>19.4</td>
      <td>19.8</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Michigan</td>
      <td>1.00</td>
      <td>509</td>
      <td>495</td>
      <td>1005</td>
      <td>0.29</td>
      <td>24.1</td>
      <td>23.7</td>
      <td>24.5</td>
      <td>23.8</td>
      <td>24.1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Idaho</td>
      <td>0.93</td>
      <td>513</td>
      <td>493</td>
      <td>1005</td>
      <td>0.38</td>
      <td>21.9</td>
      <td>21.8</td>
      <td>23.0</td>
      <td>22.1</td>
      <td>22.3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Delaware</td>
      <td>1.00</td>
      <td>503</td>
      <td>492</td>
      <td>996</td>
      <td>0.18</td>
      <td>24.1</td>
      <td>23.4</td>
      <td>24.8</td>
      <td>23.6</td>
      <td>24.1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>District of Columbia</td>
      <td>1.00</td>
      <td>482</td>
      <td>468</td>
      <td>950</td>
      <td>0.32</td>
      <td>24.4</td>
      <td>23.5</td>
      <td>24.9</td>
      <td>23.5</td>
      <td>24.2</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Maryland</td>
      <td>0.69</td>
      <td>536</td>
      <td>52</td>
      <td>1060</td>
      <td>0.28</td>
      <td>23.3</td>
      <td>23.1</td>
      <td>24.2</td>
      <td>2.3</td>
      <td>23.6</td>
    </tr>
  </tbody>
</table>
</div>



##### 15. Use a boolean filter to display only observations with a score above a certain threshold (e.g. only states with a participation rate above 50%)


```python
# Shows states above the mean for SAT reading & writing, ordered by that stat in descending order

merged.loc[merged['Reading_writing_sat'] > np.mean(merged['Reading_writing_sat'])].sort_values('Reading_writing_sat', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Participation_sat</th>
      <th>Reading_writing_sat</th>
      <th>Math_sat</th>
      <th>Total_sat</th>
      <th>Participation_act</th>
      <th>English_act</th>
      <th>Math_act</th>
      <th>Reading_act</th>
      <th>Science_act</th>
      <th>Composite_act</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23</th>
      <td>Minnesota</td>
      <td>0.03</td>
      <td>644</td>
      <td>651</td>
      <td>1295</td>
      <td>1.00</td>
      <td>20.4</td>
      <td>21.5</td>
      <td>21.8</td>
      <td>21.6</td>
      <td>21.5</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Wisconsin</td>
      <td>0.03</td>
      <td>642</td>
      <td>649</td>
      <td>1291</td>
      <td>1.00</td>
      <td>19.7</td>
      <td>20.4</td>
      <td>20.6</td>
      <td>20.9</td>
      <td>20.5</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Iowa</td>
      <td>0.02</td>
      <td>641</td>
      <td>635</td>
      <td>1275</td>
      <td>0.67</td>
      <td>21.2</td>
      <td>21.3</td>
      <td>22.6</td>
      <td>22.1</td>
      <td>21.9</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Missouri</td>
      <td>0.03</td>
      <td>640</td>
      <td>631</td>
      <td>1271</td>
      <td>1.00</td>
      <td>19.8</td>
      <td>19.9</td>
      <td>20.8</td>
      <td>20.5</td>
      <td>20.4</td>
    </tr>
    <tr>
      <th>34</th>
      <td>North Dakota</td>
      <td>0.02</td>
      <td>635</td>
      <td>621</td>
      <td>1256</td>
      <td>0.98</td>
      <td>19.0</td>
      <td>20.4</td>
      <td>20.5</td>
      <td>20.6</td>
      <td>20.3</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Mississippi</td>
      <td>0.02</td>
      <td>634</td>
      <td>607</td>
      <td>1242</td>
      <td>1.00</td>
      <td>18.2</td>
      <td>18.1</td>
      <td>18.8</td>
      <td>18.8</td>
      <td>18.6</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Kansas</td>
      <td>0.04</td>
      <td>632</td>
      <td>628</td>
      <td>1260</td>
      <td>0.73</td>
      <td>21.1</td>
      <td>21.3</td>
      <td>22.3</td>
      <td>21.7</td>
      <td>21.7</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Kentucky</td>
      <td>0.04</td>
      <td>631</td>
      <td>616</td>
      <td>1247</td>
      <td>1.00</td>
      <td>19.6</td>
      <td>19.4</td>
      <td>20.5</td>
      <td>20.1</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Nebraska</td>
      <td>0.03</td>
      <td>629</td>
      <td>625</td>
      <td>1253</td>
      <td>0.84</td>
      <td>20.9</td>
      <td>20.9</td>
      <td>21.9</td>
      <td>21.5</td>
      <td>21.4</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Wyoming</td>
      <td>0.03</td>
      <td>626</td>
      <td>604</td>
      <td>1230</td>
      <td>1.00</td>
      <td>19.4</td>
      <td>19.8</td>
      <td>20.8</td>
      <td>20.6</td>
      <td>20.2</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Utah</td>
      <td>0.03</td>
      <td>624</td>
      <td>614</td>
      <td>1238</td>
      <td>1.00</td>
      <td>19.5</td>
      <td>19.9</td>
      <td>20.8</td>
      <td>20.6</td>
      <td>20.3</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Tennessee</td>
      <td>0.05</td>
      <td>623</td>
      <td>604</td>
      <td>1228</td>
      <td>1.00</td>
      <td>19.5</td>
      <td>19.2</td>
      <td>20.1</td>
      <td>19.9</td>
      <td>19.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>0.03</td>
      <td>614</td>
      <td>594</td>
      <td>1208</td>
      <td>1.00</td>
      <td>18.9</td>
      <td>19.0</td>
      <td>19.7</td>
      <td>19.5</td>
      <td>19.4</td>
    </tr>
    <tr>
      <th>41</th>
      <td>South Dakota</td>
      <td>0.03</td>
      <td>612</td>
      <td>603</td>
      <td>1216</td>
      <td>0.80</td>
      <td>20.7</td>
      <td>21.5</td>
      <td>22.3</td>
      <td>22.0</td>
      <td>21.8</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Louisiana</td>
      <td>0.04</td>
      <td>611</td>
      <td>586</td>
      <td>1198</td>
      <td>1.00</td>
      <td>19.4</td>
      <td>18.8</td>
      <td>19.8</td>
      <td>19.6</td>
      <td>19.5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Colorado</td>
      <td>0.11</td>
      <td>606</td>
      <td>595</td>
      <td>1201</td>
      <td>1.00</td>
      <td>20.1</td>
      <td>20.3</td>
      <td>21.2</td>
      <td>20.9</td>
      <td>20.8</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Montana</td>
      <td>0.10</td>
      <td>605</td>
      <td>591</td>
      <td>1196</td>
      <td>1.00</td>
      <td>19.0</td>
      <td>20.2</td>
      <td>21.0</td>
      <td>20.5</td>
      <td>20.3</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>0.05</td>
      <td>593</td>
      <td>572</td>
      <td>1165</td>
      <td>1.00</td>
      <td>18.9</td>
      <td>18.4</td>
      <td>19.7</td>
      <td>19.4</td>
      <td>19.2</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Ohio</td>
      <td>0.12</td>
      <td>578</td>
      <td>570</td>
      <td>1149</td>
      <td>0.75</td>
      <td>21.2</td>
      <td>21.6</td>
      <td>22.5</td>
      <td>22.0</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>New Mexico</td>
      <td>0.11</td>
      <td>577</td>
      <td>561</td>
      <td>1138</td>
      <td>0.66</td>
      <td>18.6</td>
      <td>19.4</td>
      <td>20.4</td>
      <td>20.0</td>
      <td>19.7</td>
    </tr>
  </tbody>
</table>
</div>



## Step 3: Visualize the data

##### 16. Using MatPlotLib and PyPlot, plot the distribution of the Rate columns for both SAT and ACT using histograms. (You should have two histograms. You might find [this link](https://matplotlib.org/users/pyplot_tutorial.html#working-with-multiple-figures-and-axes) helpful in organizing one plot above the other.) 


```python
plt.subplot(211)
plt.hist(merged.Participation_sat)
plt.title('SAT')
plt.subplot(212)
plt.hist(merged.Participation_act)
plt.title('ACT')
plt.subplots_adjust(hspace=1)
plt.show()
```


![png](/images/SAT_and_ACT_results_by_state_2017_files/SAT_and_ACT_results_by_state_2017_38_0.png)


##### 17. Plot the Math(s) distributions from both data sets.


```python
f, (ax1, ax2) = plt.subplots(2)
plt.subplots_adjust(hspace=1)
sns.distplot(merged.Math_sat, bins=30, ax=ax1, kde=False, fit=stats.norm).set_title("SAT: Math distribution")
ax1.set_xlabel('State Average Math Scores, 2017\nline = normal distribution')
sns.distplot(merged.Math_act, bins = 20, ax=ax2, kde=False, fit=stats.norm).set_title("ACT: Math distribution")
ax2.set_xlabel('State Average Math Scores, 2017\nline = normal distribution')
```




    Text(0.5,0,'State Average Math Scores, 2017\nline = normal distribution')




![png](/images/SAT_and_ACT_results_by_state_2017_files/SAT_and_ACT_results_by_state_2017_40_1.png)


##### 18. Plot the Verbal distributions from both data sets.


```python
f, (ax1, ax2) = plt.subplots(2)
plt.subplots_adjust(hspace=1)
sns.distplot(merged.Reading_writing_sat, bins=20, ax=ax1, kde=False, fit=stats.norm).set_title("SAT: Reading & Writing distribution")
ax1.set_xlabel('State Average Combined Reading & Writing Scores, 2017\nline = normal distribution')
sns.distplot(merged.English_act, bins = 20, ax=ax2, kde=False, fit=stats.norm).set_title("ACT: English distribution")
ax2.set_xlabel('State Average English Scores, 2017\nline = normal distribution')
```




    Text(0.5,0,'State Average English Scores, 2017\nline = normal distribution')




![png](/images/SAT_and_ACT_results_by_state_2017_files/SAT_and_ACT_results_by_state_2017_42_1.png)


##### 19. When we make assumptions about how data are distributed, what is the most common assumption?


```python
# Normal distribution
```

##### 20. Does this assumption hold true for any of our columns? Which?


```python
from scipy import stats as stats
for column in merged.drop(['State'], axis=1).columns.values:
    result = stats.mstats.normaltest(merged[column], axis=0)
    if result[1] > .05:
        print(f"{column.title()} has a normal distribution")
    
#result = stats.mstats.normaltest(merged['English_act'], axis=0)
```

    English_Act has a normal distribution



```python
# The ACT English score has a normal distribution
```

##### 21. Plot some scatterplots examining relationships between all variables.


```python
sns.pairplot(merged)
plt.show()
```


![png](/images/SAT_and_ACT_results_by_state_2017_files/SAT_and_ACT_results_by_state_2017_49_0.png)


##### 22. Are there any interesting relationships to note?


```python
# In general state test score averages seem to be correlated across subjects with a strong linear relationship
# There is a "cluster" phenomenon with SAT Math and ACT Science scores that looks interesting.
```

##### 23. Create box plots for each variable. 


```python
# This needs work, I know!

merged.plot.box()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a193d2e10>




![png](/images/SAT_and_ACT_results_by_state_2017_files/SAT_and_ACT_results_by_state_2017_53_1.png)


##### BONUS: Using Tableau, create a heat map for each variable using a map of the US. 

## Step 4: Descriptive and Inferential Statistics

##### 24. Summarize each distribution. As data scientists, be sure to back up these summaries with statistics. (Hint: What are the three things we care about when describing distributions?)


```python
merged.describe().T

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Participation_sat</th>
      <td>51.0</td>
      <td>0.398039</td>
      <td>0.352766</td>
      <td>0.02</td>
      <td>0.04</td>
      <td>0.38</td>
      <td>0.66</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Reading_writing_sat</th>
      <td>51.0</td>
      <td>569.117647</td>
      <td>45.666901</td>
      <td>482.00</td>
      <td>533.50</td>
      <td>559.00</td>
      <td>613.00</td>
      <td>644.0</td>
    </tr>
    <tr>
      <th>Math_sat</th>
      <td>51.0</td>
      <td>547.627451</td>
      <td>84.909119</td>
      <td>52.00</td>
      <td>522.00</td>
      <td>548.00</td>
      <td>599.00</td>
      <td>651.0</td>
    </tr>
    <tr>
      <th>Total_sat</th>
      <td>51.0</td>
      <td>1126.098039</td>
      <td>92.494812</td>
      <td>950.00</td>
      <td>1055.50</td>
      <td>1107.00</td>
      <td>1212.00</td>
      <td>1295.0</td>
    </tr>
    <tr>
      <th>Participation_act</th>
      <td>51.0</td>
      <td>0.652549</td>
      <td>0.321408</td>
      <td>0.08</td>
      <td>0.31</td>
      <td>0.69</td>
      <td>1.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>English_act</th>
      <td>51.0</td>
      <td>20.931373</td>
      <td>2.353677</td>
      <td>16.30</td>
      <td>19.00</td>
      <td>20.70</td>
      <td>23.30</td>
      <td>25.5</td>
    </tr>
    <tr>
      <th>Math_act</th>
      <td>51.0</td>
      <td>21.182353</td>
      <td>1.981989</td>
      <td>18.00</td>
      <td>19.40</td>
      <td>20.90</td>
      <td>23.10</td>
      <td>25.3</td>
    </tr>
    <tr>
      <th>Reading_act</th>
      <td>51.0</td>
      <td>22.013725</td>
      <td>2.067271</td>
      <td>18.10</td>
      <td>20.45</td>
      <td>21.80</td>
      <td>24.15</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>Science_act</th>
      <td>51.0</td>
      <td>21.041176</td>
      <td>3.182463</td>
      <td>2.30</td>
      <td>19.90</td>
      <td>21.30</td>
      <td>22.75</td>
      <td>24.9</td>
    </tr>
    <tr>
      <th>Composite_act</th>
      <td>51.0</td>
      <td>21.519608</td>
      <td>2.020695</td>
      <td>17.80</td>
      <td>19.80</td>
      <td>21.40</td>
      <td>23.60</td>
      <td>25.5</td>
    </tr>
  </tbody>
</table>
</div>




```python


'''
SAT Participation: Participation for the SAT varies widely, from near-zero to 100%. The distribution is skewed right due to a large group of states with very low participation (these states tend to have high participation rates in the ACT) 
SAT Reading & Writing: Reading and writing has a fairly narrow spread, with STD of 45.67 vs a mean of 569.12
SAT Math: Math SAT scores are somewhat more spread than Reading & Writing, with STD of 84.91 vs a mean of 546.63
SAT Total: SAT total average has a mean of 1126.10 with STD of 92.49
ACT Participation: Similar to SAT particpation, varies widely - although less widely than SAT participation 
ACT English: Medium spread (2.35 STD vs mean of 20.93), has a normal distribution
ACT Math: Similar to English, medium spread (2.35 STD vs mean of 21.18)
ACT Science: Skewed slightly left
ACT Composite: close to normal distribution
'''

```




    NormaltestResult(statistic=4.945616782086351, pvalue=0.08434764489205682)



##### 25. Summarize each relationship. Be sure to back up these summaries with statistics.

##### 26. Execute a hypothesis test comparing the SAT and ACT participation rates. Use $\alpha = 0.05$. Be sure to interpret your results.


```python
# best test is just to see if the participation rates are correlated, like so:
merged.corr(method='pearson', min_periods=1)
# result - SAT participation is strongly negatively correlated with ACT participation (-0.84)
# however, the question asked for hypothesis testing...
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Participation_sat</th>
      <th>Reading_writing_sat</th>
      <th>Math_sat</th>
      <th>Total_sat</th>
      <th>Participation_act</th>
      <th>English_act</th>
      <th>Math_act</th>
      <th>Reading_act</th>
      <th>Science_act</th>
      <th>Composite_act</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Participation_sat</th>
      <td>1.000000</td>
      <td>-0.874326</td>
      <td>-0.566558</td>
      <td>-0.867540</td>
      <td>-0.841234</td>
      <td>0.686889</td>
      <td>0.710697</td>
      <td>0.705352</td>
      <td>0.248553</td>
      <td>0.694748</td>
    </tr>
    <tr>
      <th>Reading_writing_sat</th>
      <td>-0.874326</td>
      <td>1.000000</td>
      <td>0.628405</td>
      <td>0.996661</td>
      <td>0.716153</td>
      <td>-0.461345</td>
      <td>-0.486126</td>
      <td>-0.488441</td>
      <td>-0.135461</td>
      <td>-0.470382</td>
    </tr>
    <tr>
      <th>Math_sat</th>
      <td>-0.566558</td>
      <td>0.628405</td>
      <td>1.000000</td>
      <td>0.632648</td>
      <td>0.507670</td>
      <td>-0.345342</td>
      <td>-0.340906</td>
      <td>-0.363099</td>
      <td>0.594714</td>
      <td>-0.346335</td>
    </tr>
    <tr>
      <th>Total_sat</th>
      <td>-0.867540</td>
      <td>0.996661</td>
      <td>0.632648</td>
      <td>1.000000</td>
      <td>0.701477</td>
      <td>-0.441947</td>
      <td>-0.454116</td>
      <td>-0.466558</td>
      <td>-0.121783</td>
      <td>-0.445020</td>
    </tr>
    <tr>
      <th>Participation_act</th>
      <td>-0.841234</td>
      <td>0.716153</td>
      <td>0.507670</td>
      <td>0.701477</td>
      <td>1.000000</td>
      <td>-0.843501</td>
      <td>-0.861114</td>
      <td>-0.866620</td>
      <td>-0.304992</td>
      <td>-0.858134</td>
    </tr>
    <tr>
      <th>English_act</th>
      <td>0.686889</td>
      <td>-0.461345</td>
      <td>-0.345342</td>
      <td>-0.441947</td>
      <td>-0.843501</td>
      <td>1.000000</td>
      <td>0.967803</td>
      <td>0.985999</td>
      <td>0.403456</td>
      <td>0.990856</td>
    </tr>
    <tr>
      <th>Math_act</th>
      <td>0.710697</td>
      <td>-0.486126</td>
      <td>-0.340906</td>
      <td>-0.454116</td>
      <td>-0.861114</td>
      <td>0.967803</td>
      <td>1.000000</td>
      <td>0.979630</td>
      <td>0.412318</td>
      <td>0.990451</td>
    </tr>
    <tr>
      <th>Reading_act</th>
      <td>0.705352</td>
      <td>-0.488441</td>
      <td>-0.363099</td>
      <td>-0.466558</td>
      <td>-0.866620</td>
      <td>0.985999</td>
      <td>0.979630</td>
      <td>1.000000</td>
      <td>0.401097</td>
      <td>0.995069</td>
    </tr>
    <tr>
      <th>Science_act</th>
      <td>0.248553</td>
      <td>-0.135461</td>
      <td>0.594714</td>
      <td>-0.121783</td>
      <td>-0.304992</td>
      <td>0.403456</td>
      <td>0.412318</td>
      <td>0.401097</td>
      <td>1.000000</td>
      <td>0.408656</td>
    </tr>
    <tr>
      <th>Composite_act</th>
      <td>0.694748</td>
      <td>-0.470382</td>
      <td>-0.346335</td>
      <td>-0.445020</td>
      <td>-0.858134</td>
      <td>0.990856</td>
      <td>0.990451</td>
      <td>0.995069</td>
      <td>0.408656</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# H0: NULL Hypothesis:  the mean of the participation rates is equal
# H1: Alternative Hypothesis: there is a statistically significant difference between the means

# Calculating the SAT Participation Sample Mean - we will use this as mu in our test
sat_mean = np.mean(merged['Participation_sat'])

# Running a t-test using the ACT participation rate as our sample and SAT participation rate as population 
stats.ttest_1samp(merged['Participation_act'], sat_mean, axis=0)

# Given the resulting pvalue of 7.48, which is well above our alpha of 0.05, we can reject the null hypothesis and concclude that the population means are not identical

```




    Ttest_1sampResult(statistic=5.6549966804539755, pvalue=7.484459723240645e-07)



##### 27. Generate and interpret 95% confidence intervals for SAT and ACT participation rates.


```python
sat_std = np.std(merged['Participation_sat'])
act_std = np.std(merged['Participation_act'])
sat_conf = stats.norm.interval(0.95, loc=sat_mean, scale=sat_std)
act_conf = stats.norm.interval(0.95, loc=act_mean, scale=act_std)
print(f"SAT participation 95% confidence interval: {sat_conf}")
print(f"ACT participation 95% confidence interval: {act_conf}")
```

    SAT participation 95% confidence interval: (-0.28655799147447736, 1.0826364228470264)
    ACT participation 95% confidence interval: (0.028806636506970462, 1.2762914027087158)


##### 28. Given your answer to 26, was your answer to 27 surprising? Why?


```python
# It is not suprising that the confidence intervals are different given that we proved that the means are different
```

##### 29. Is it appropriate to generate correlation between SAT and ACT math scores? Why?


```python
# It is appropriate to measure correlation, because they are different measurements from the same population (students in 2017)
merged.loc[:,['Math_sat','Math_act']].corr(method='pearson')
# However we see from the results that there is only a weak negative correlation between them (-0.34)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Math_sat</th>
      <th>Math_act</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Math_sat</th>
      <td>1.000000</td>
      <td>-0.340906</td>
    </tr>
    <tr>
      <th>Math_act</th>
      <td>-0.340906</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



##### 30. Suppose we only seek to understand the relationship between SAT and ACT data in 2017. Does it make sense to conduct statistical inference given the data we have? Why?


```python
# Yes, it does - all of our data is from 2017
```
