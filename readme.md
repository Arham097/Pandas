# Pandas: A Comprehensive Guide

## Introduction
Pandas is a fast, powerful, flexible, and easy-to-use open-source data analysis and manipulation tool built on top of the Python programming language. This repository is dedicated to exploring various concepts of Pandas, starting with **Series**, and will be expanded further with other Pandas functionalities such as DataFrames.

For understanding **Series**, we will work with the following datasets:
- **movies.csv**
- **kohli_ipl.csv** (referred to as `vk`)
- **subs.csv**

## Importing Pandas
To begin, import the required libraries:
```python
import numpy as np
import pandas as pd
```

## Pandas Series
A **Series** in Pandas is a one-dimensional labeled array that can hold data of any type. Below are different ways to create a Series.

### Series from Lists
#### Creating a Series from a List
```python
# String values
country = ['India', 'Pakistan', 'Iran', 'USA']
pd.Series(country)
```
```python
# Integer values
runs = [10, 20, 43, 22, 57, 82]
pd.Series(runs)
```
```python
# Custom indexing
marks = [100, 84, 78, 74, 91]
subjects = ['maths', 'chemistry', 'physics', 'pst', 'english']
pd.Series(marks, index=subjects)
```
```python
# Setting a name for Series
marks = pd.Series(marks, index=subjects, name="Marksheet")
print(marks)
```

### Series from Dictionary
```python
marks = {
  'maths': 100,
  'chemistry': 84,
  'physics': 78,
  'ost': 74,
  'english': 91
}
marks_series = pd.Series(marks, name="Arham's Marks")
print(marks_series)
```

## Series Attributes
Series comes with useful attributes such as:
```python
# Getting size
marks_series.size

# Getting data type
type(marks_series)

# Checking if values are unique
marks_series.is_unique
```

## Loading Data from CSV
We can load real-world data using `pd.read_csv()`:
```python
subs = pd.read_csv('subs.csv', squeeze=True)
vk = pd.read_csv('kohli_ipl.csv', index_col='match_no', squeeze=True)
movies = pd.read_csv('bollywood.csv', index_col='movie', squeeze=True)
```

## Series Methods
Pandas provides multiple built-in methods for analysis:
```python
# Display first n rows
subs.head()

# Display last n rows
vk.tail(10)

# Getting unique values and their counts
movies.value_counts()
```
```python
# Sorting values
vk.sort_values(ascending=False).head(1).values[0]

# Sorting by index
movies.sort_index(ascending=False, inplace=True)
```

## Mathematical Operations on Series
```python
# Basic statistics
print(subs.sum())
print(subs.mean())
print(subs.median())
print(subs.var())
print(subs.std())
```
## Series with Python Functionalities

Pandas Series integrates well with core Python functionalities, allowing seamless interaction using built-in functions and operators.

### Basic Python Functions with Series
```python
print(len(subs))  # Number of elements in Series
print(type(subs))  # Data type of Series
print(dir(subs))   # Available attributes and methods
print(sorted(subs)) # Sorting values
print(min(subs))   # Minimum value
print(max(subs))   # Maximum value
```

### Type Conversion of Series
```python
print(list(marks_series))
print(dict(marks_series))
```

### Membership Operators with Series
```python
print('maths' in marks_series) # check in index 
print(100 in marks_series.values)  # check in values
```

### Iterating Over a Series
```python
for i in movies.index:
    print(i)  # Printing movie names (index values)
```

### Arithmetic Operations on Series (Broadcasting)

Mathematical operations on a Series apply element-wise.

```python
print(100 + marks_series)  # Adds 100 to each value
print(100 - marks_series)  # Subtracts 100 from each value
```
### Relational Operators on Series
Series supports relational operators, which return boolean values.

```python
print(marks_series >= 80)  # Checks which subjects have marks >= 80
```



## Series Indexing and Slicing
Series supports advanced indexing and slicing techniques:
```python
# Accessing elements
print(movies[2])
print(movies[-1])  # Negative indexing works for labeled indexes
```
```python
# Slicing
print(vk[5:16])
print(vk[-5:])  # Last five values
```

## Editing a Series
```python
# Modifying values using indexing
marks_series[1] = 100
print(marks_series)
marks_series['urdu'] = 83  # Adds new entry
print(marks_series)

# Modifying using slicing
marks_series[3:5] = 0
print(marks_series)
```

## Boolean Indexing
Using conditions to filter data:
```python
# Finding the number of 50s and 100s scored by Kohli
print("50's Scored by Kohli:", vk[(vk>=50) & (vk<100)].size)
print("100's Scored by Kohli:", vk[(vk>=100)].size)

# Find no. of ducks
print("No. of ducks scored by Kohli:", vk[vk==0].size)

# Count number of day when I had more than 200 subs a day
print(subs[subs>200].size)
```

## Plotting Graphs with Series
```python
subs.plot()
movies.value_counts().head(20).plot(kind='bar')
```

## Advanced Series Methods
Pandas provides additional useful methods:
```python
# Type conversion using astype
subs.astype('int16')

# Checking values in a given range using between
vk[vk.between(51,99)].size

# Clipping values to a range
subs.clip(100,200)

# drop_duplicates => removes duplicate values
temp = pd.Series([1,1,2,2,3,3,4,4])
print(temp)
temp.drop_duplicates() # keep first occurence and delete other occurences
temp.drop_duplicates(keep='last')  # keep last occurence and delete other occurences

# another function is ``duplicated`` -> return boolean series where duplicated values are true and other are false

print(temp.duplicated())
print(temp.duplicated().sum()) # tell no of duplicated items

# isin -> checks the no. of values exist in data or not
print(vk)
print(vk[vk.isin([51,99])])
```
```python
# isnull -> return boolean series where null values true and other are false
temp2 = pd.Series([1,2,3,np.nan,5,6,np.nan,8,np.nan,10])
print(temp2)
print(temp2.isnull()) 
print(temp2.isnull().sum()) # count the number of null values

# dropna -> drop all the null values 
print(temp2)
temp2.dropna()

# fillna -> fills the null values with desired value
print(temp2) 
print(temp2.fillna(temp2.mean())) # fills null values with the series' mean

# copy => creates a copy of data => helps in manipulation on data without affecting original one
vk2 = vk.head(10).copy()
vk2[1] = 100
print(vk2)
print(vk)
```

## Using `apply()` for Custom Functions
Applying custom logic to Series:
```python
# get the first name of actors in upper case
movies.apply(lambda x: x.split()[0].upper())
```
```python
# Labeling days based on subscription count
subs.apply(lambda x: 'Good Day' if x > 200 else 'Bad Day')
```

# DataFrame Important Functions

## Required Datasets

- `ipl-matches.csv`
- `movies.csv`
- `batsman_runs_series.csv`

Ensure these datasets are available in the same directory as your script for proper functioning.

## Table of Contents

- [value\_counts()](#value_counts)
- [sort\_values()](#sort_values)
- [rank()](#rank)
- [sort\_index()](#sort_index)
- [set\_index()](#set_index)
- [reset\_index()](#reset_index)
- [rename()](#rename)
- [nunique()](#nunique)
- [isnull() / notnull()](#isnull-notnull)
- [dropna()](#dropna)
- [fillna()](#fillna)
- [drop\_duplicates()](#drop_duplicates)
- [drop()](#drop)
- [apply()](#apply)

## sort\_values()

The `sort_values()` function sorts a DataFrame or Series by the specified labels along a given axis.

```python
movies = pd.read_csv('movies.csv')

# Sort by single column
movies.sort_values('title_x', ascending=False)

# Sort by multiple columns
movies.sort_values(['year_of_release', 'title_x'], ascending=[True, False])
```

---

## rank()

The `rank()` function provides rank values to data points within a Series or DataFrame.

```python
batsman = pd.read_csv('batsman_runs_series.csv')
batsman['rank'] = batsman['batsman_run'].rank(ascending=False)
batsman.sort_values('rank')
```

---

## sort\_index()

The `sort_index()` function sorts a DataFrame or Series by its index.

```python
marks = pd.Series({'maths': 67, 'english': 57, 'science': 89, 'hindi': 100})
marks.sort_index(ascending=False)

movies.sort_index(ascending=False)
```

---

## set\_index()

The `set_index()` function is used to set the DataFrame index using an existing column.

```python
batsman.set_index('batter', inplace=True)
```

---

## reset\_index()

The `reset_index()` function resets the index of a DataFrame, converting it back to the default integer index. Can convert Series to DataFrame.

```python
batsman.reset_index(inplace=True)
# how to replace existing index without loosing
batsman.reset_index().set_index('rank')
# series to datafram using reset_index
a = pd.Series([1,2,3,4])
print(a)
a.reset_index()
marks_series.reset_index()
```

---

## rename()

The `rename()` function is used to rename rows or columns.

```python
movies.rename(columns={'imdb_id': 'imdb'}, inplace=True)
movies.rename(index={'Uri: The Surgical Strike': 'Uri'}, inplace=True)
```

---

## nunique()

The `nunique()` function returns the number of unique values in a Series or DataFrame.

```python
ipl['Season'].nunique()
```

---

## isnull() / notnull()

These functions detect missing values in a DataFrame or Series.

```python
students['name'].isnull()  # Returns True for NaN values
students['name'].notnull()  # Returns True for non-NaN values

students.isnull().sum()  # Total count of missing values per column
```

---

## dropna()

The `dropna()` function is used to remove missing values from a DataFrame or Series. By default, it removes rows containing any null value.

```python
students.dropna()  # Removes rows with any NaN value
students.dropna(subset=['marks'])  # Removes rows where 'marks' is NaN
students.dropna(how='all')  # Removes rows where all values are NaN
```

---

## fillna()

The `fillna()` function replaces missing values with a specified value or strategy (like forward fill or backward fill).

```python
students.fillna(0)  # Replaces all NaN values with 0
students.fillna(method='ffill')  # Forward fills NaN values with the last valid observation
students.fillna({'marks': 0, 'name': 'Unknown'})  # Custom fill for specific columns
```

---

## drop\_duplicates()

The `drop_duplicates()` function removes duplicate rows from a DataFrame. Duplicate detection can be limited to specific columns.

```python
students.drop_duplicates()  # Drops fully duplicated rows
students.drop_duplicates(subset=['name'], keep='first')  # Keeps the first occurrence of each duplicate
students.drop_duplicates(subset=['name'], keep='last')  # Keeps the last occurrence of each duplicate
```

---

## drop()

The `drop()` function removes rows or columns by labels or index values.

```python
students.drop('marks', axis=1)  # Drops the 'marks' column
students.drop([0, 2])  # Drops rows by index labels 0 and 2
students.drop(index=[1, 3], columns=['marks', 'name'])  # Drops specific rows and columns simultaneously
```

---

## apply()

The `apply()` function allows applying a function along a particular axis (rows or columns) of a DataFrame.

```python
students['marks'].apply(lambda x: x * 2)  # Applies a lambda function to each element of 'marks'
students.apply(sum, axis=0)  # Applies the sum function column-wise
students.apply(sum, axis=1)  # Applies the sum function row-wise
```

## Group-By Objects

This section covers various operations that can be performed on GroupBy objects using the pandas library. It demonstrates grouping data by specific columns, applying aggregation functions, looping over groups, and working with multiple columns for advanced analysis.

### Importing Dataset
```python
movies = pd.read_csv('imdb-top-1000.csv')
movies.head(2)
```

### Creating GroupBy Objects
```python
genre = movies.groupby('Genre')
```

## Applying Built-In Aggregation Functions

```python
# Applying standard deviation aggregation to all groups
genre.std()

# Find top 3 Genres by total earning
movies.groupby('Genre')['Gross'].sum().sort_values(ascending=False).head(3)

# Find the genre with the highest average IMDB rating
movies.groupby('Genre')['IMDB_Rating'].mean().sort_values(ascending=False).head(1)

# Find the most popular director by total votes
movies.groupby('Director')['No_of_Votes'].sum().sort_values(ascending=False).head(1)

# Find the highest-rated movie of each genre
movies.groupby('Genre')['IMDB_Rating'].max()

# Find the number of movies done by each actor
movies.groupby('Star1')['Runtime'].count().sort_values(ascending=False)
```

## GroupBy Attributes and Methods
- `len`: Find total number of groups.
- `size`: Find the number of items in each group.
- `first()` / `last()`: Access the first or last item of each group.
- `nth()`: Retrieve the nth item of each group.
- `get_group()`: Retrieve a specific group.
- `groups`: Dictionary with group labels and indices.
- `describe()`: Generate descriptive statistics for each group.
- `sample()`: Randomly sample rows from each group.
- `nunique()`: Count unique values in each group.

### Example Usage
```python
# Number of unique groups
len(movies.groupby('Genre'))  # Output: 14
movies['Genre'].nunique()      # Output: 14

# Size of each group
movies.groupby('Genre').size()  # Returns number of rows in each group

# First and Last items of each group
movies.groupby('Genre').first()
movies.groupby('Genre').last()

# nth Item of each group
movies.groupby('Genre').nth(5)  # Retrieves the 6th movie in each group

# Using get_group() method
movies.groupby('Genre').get_group('Drama')

# Accessing groups attribute
# It gives dictionary with key is Genre and value will be the list containing index where same Genre present
movies.groupby('Genre').groups

# Descriptive statistics for each group
movies.groupby('Genre').describe()

# Random Sampling from groups
movies.groupby('Genre').sample(2, replace=True)
# replce parameter Allow or disallow sampling of the same row more than once as Action appear multiple times without replace true 2 action movies not goint to show and throw error
```

## Aggregation Methods
Aggregation can be done using dictionaries, lists, or a combination of both.

```python
movies.groupby('Genre').agg({
    'Runtime': 'mean',
    'IMDB_Rating': 'mean',
    'No_of_Votes': 'sum',
    'Gross': 'sum',
    'Metascore': 'min'
})

movies.groupby('Genre').agg(['min', 'max'])
```

## Looping Over Groups
```python
# Finding highest-rated movie of each genre

genres = movies.groupby('Genre')
df = pd.DataFrame(columns=movies.columns)
for group, data in genres:
    df = pd.concat([df, data[data['IMDB_Rating'] == data['IMDB_Rating'].max()]])

df
```

## Split-Apply-Combine Mechanism
The split-apply-combine mechanism involves splitting a dataset, applying a function to each group independently, and then combining the results into a DataFrame.

```python
# Apply a function to each group

def startWithA(group):
    return group['Series_Title'].str.startswith('A').sum()

genres.apply(startWithA)
```

## GroupBy with Multiple Columns
Grouping by multiple columns allows for more detailed analysis.

```python
# Grouping by Director and Star1

duo = movies.groupby(['Director', 'Star1'])

duo.size()
duo.get_group(('Aamir Khan', 'Amole Gupte'))
```

### Example Tasks
- Find the most earning actor-director combination.
- Find the best actor-genre combo in terms of average Metascore.
- Apply multiple aggregation functions to a grouped dataset.

```python
# find the most earning actor->director combo
duo['Gross'].sum().sort_values(ascending=False).head(1)


# Find the best actor-genre combo in terms of average Metascore.
combo = movies.groupby(['Star1', 'Genre'])
combo['Metascore'].mean().sort_values(ascending=False).head(1)

# Apply multiple aggregation functions to a grouped dataset.
duo.agg(['min', 'max', 'mean'])
```

This section provides a comprehensive understanding of GroupBy operations with pandas, showcasing various methods and attributes to manipulate and analyze datasets effectively.


# Merging, Joining, and Concatenating in Pandas

This script demonstrates various techniques for merging, joining, and concatenating DataFrames using the Pandas library. It includes examples of concatenation, joins (inner, left, right, outer), and self-joins, highlighting their differences and use cases.


## Datasets Used
The following CSV files are loaded using `pd.read_csv()`:
- `courses.csv`
- `students.csv`
- `reg-month1.csv` (November registrations)
- `reg-month2.csv` (December registrations)
- `matches.csv`
- `deliveries.csv`

## Code Explanation

### Imports
```python
import pandas as pd
```

### Loading DataFrames
```python
courses = pd.read_csv('courses.csv')
students = pd.read_csv('students.csv')
nov = pd.read_csv('reg-month1.csv')
dec = pd.read_csv('reg-month2.csv')
matches = pd.read_csv('matches.csv')
delivery = pd.read_csv('deliveries.csv')
```

## Concatenation

### `pd.concat()`
Concatenates two or more DataFrames with the same number of columns vertically.

```python
pd.concat([nov, dec], ignore_index=True)
```

- `ignore_index=True`: Assigns a continuous index across concatenated DataFrames.

### `append()`
An alternative to `pd.concat()` that works similarly.

```python
nov.append(dec, ignore_index=True)
```

### MultiIndex Concatenation
```python
multi = pd.concat([nov, dec], keys=['Nov', 'Dec'])  # Creates a MultiIndex DataFrame
multi.loc[('Dec', 3)]  # Fetching value by MultiIndex
```

### Horizontal Concatenation
Concatenates DataFrames along columns using `axis=1`.

```python
pd.concat([nov, dec], axis=1)
```

## Joins

### Inner Join
Returns only the rows where keys match in both DataFrames.

```python
regs = pd.concat([nov, dec], ignore_index=True)
students.merge(regs, how='inner', on='student_id')
```

### Left Join
Returns all rows from the left DataFrame, with matching rows from the right. Missing values are filled with NaN.

```python
courses.merge(regs, how='left', on='course_id')
```

### Right Join
Returns all rows from the right DataFrame, with matching rows from the left. Missing values are filled with NaN.

```python
temp_df = pd.DataFrame({
    'student_id': [26, 27, 28],
    'name': ['Nitish', 'Ankit', 'Rahul'],
    'partner': [28, 26, 17]
})

students = pd.concat([students, temp_df], ignore_index=True)
students.merge(regs, how='right', on='student_id')
```

### Outer Join
Returns all rows from both DataFrames, filling non-matching entries with NaN.

```python
students.merge(regs, how='outer', on='student_id')
```

### Self Join
Joining a DataFrame with itself to compare rows within the same table, usually using `merge()` with different aliases.

```python
students.merge(students, right_on='partner', left_on='student_id')[['name_x', 'name_y']]
```

## MultiIndex Objects

### 1. MultiIndex Series

multiindex series (also known as Hierarchical Indexing) allow multiple index levels within a single index.

**How to create a MultiIndex object:**
1. `pd.MultiIndex.from_tuples()`
```python
index_val = [('cse',2019),('cse',2020),('cse',2021),('cse',2022),
             ('ece',2019),('ece',2020),('ece',2021),('ece',2022)]
multiindex = pd.MultiIndex.from_tuples(index_val)
multiindex
```

2. `pd.MultiIndex.from_product()`
```python
index = pd.MultiIndex.from_product([['cse','ece'],[2019,2020,2021,2022]])
```

**Creating a Series with MultiIndex object:**
```python
s = pd.Series([1,2,3,4,5,6,7,8], index=multiindex)
s
```

**Fetching items from the Series:**
```python
print(s['ece'])           # Returns series with 'ece' index
print(s[('ece',2020)])    # Returns a specific value
```

### 2. MultiIndex DataFrame

**MultiIndex DataFrame from rows perspective:**
```python
branchdf1 = pd.DataFrame([
  [1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16]
], index=multiindex, columns=['avg_package','students'])
branchdf1
```

```python
branchdf1['students']  # gives MultiIndex series
```

**MultiIndex DataFrame from columns perspective:**
```python
branch_df2 = pd.DataFrame([
    [1,2,0,0], [3,4,0,0], [5,6,0,0], [7,8,0,0]
], index=[2019,2020,2021,2022],
columns=pd.MultiIndex.from_product([['delhi','mumbai'],['avg_package','students']]))
branch_df2
```

```python
branch_df2.loc[2019]
```

**MultiIndex DataFrame in terms of both rows and columns:**
```python
branch_df3 = pd.DataFrame([
    [1,2,0,0],[3,4,0,0],[5,6,0,0],[7,8,0,0],
    [9,10,0,0],[11,12,0,0],[13,14,0,0],[15,16,0,0]
], index=multiindex,
columns=pd.MultiIndex.from_product([['delhi','mumbai'],['avg_package','students']]))
branch_df3
```

## Stacking and Unstacking

**Stack -> convert columns into rows:**
```python
branch_df3.stack()           # Converts inner level of index into rows
```

**Unstack -> convert rows into columns:**
```python
branch_df3.unstack()
```

### Working with MultiIndex DataFrames

```python
branch_df3.head()              # First few rows
branch_df3.shape              # Dimensions
branch_df3.info()             # Info
branch_df3.duplicated()       # Check duplicates
branch_df3.isnull()           # Check NaNs
```

**Extracting Rows:**
```python
branch_df3.loc[('cse',2020)]
branch_df3.loc[('cse',2019):('ece',2020):2]  # extract alternatively
branch_df3.iloc[0:5:2]
```

**Extracting Columns:**
```python
branch_df3['delhi']['students']
branch_df3.iloc[:,1:3]
branch_df3.iloc[[0,3],[0,3]]
```

**Sorting Index:**
```python
branch_df3.sort_index(ascending=False)
branch_df3.sort_index(ascending=[False,True])
branch_df3.sort_index(level=1,ascending=False)
```

**Transpose:**
```python
branch_df3.transpose()
```

**Swap Level:**
```python
branch_df3.swaplevel()               # row-wise
branch_df3.swaplevel(axis=1)         # column-wise
```

### Long Vs Wide Data

![Image description](https://drive.google.com/uc?export=view&id=1aWxH2yrWpjMRtAIh9PpLylRuIswE2s4-)

**Wide format** is where we have a single row for every data point with multiple columns to hold the values of various attributes.

**Long format** is where, for each data point we have as many rows as the number of attributes and each row contains the value of a particular attribute for a given data point.

### melt

**Convert wide to long format:**
```python
pd.DataFrame({'cse':[120]}).melt()
```

```python
pd.DataFrame({'cse':[120],'ece':[100],'mec':[50]}).melt(var_name='branch',value_name='students')
```

```python
pd.DataFrame({
    'branch':['cse','ece','mech'],
    '2020':[100,150,60],
    '2021':[120,130,80],
    '2022':[150,140,70]
}).melt(id_vars=['branch'], var_name='year', value_name='students')
```

## Pivot Table

The pivot table takes simple column-wise data as input, and groups the entries into a two-dimensional table that provides a multidimensional summarization of the data.

```python
import numpy as np
import pandas as pd
import seaborn as sns

df = sns.load_dataset('tips')
df.head()
```

**Average bill by gender:**
```python
df.groupby('sex')['total_bill'].mean()
```

**Average by gender and smoker status:**
```python
df.groupby(['sex','smoker'])['total_bill'].mean().unstack()
```

**Using pivot_table:**
```python
df.pivot_table(index='sex', columns='smoker', values='total_bill')
```

### aggfunc

> Additional parameter to apply any aggregate function:
> `std`, `sum`, `max`, `min`, etc. Default is `mean`.

```python
df.pivot_table(index='sex', columns='smoker', values='total_bill', aggfunc='std')
df.pivot_table(index='sex', columns='smoker', values='total_bill', aggfunc='max')
df.pivot_table(index='sex', columns='smoker', margins=True)
```

**Multidimensional pivot:**
```python
df.pivot_table(index=['sex','smoker'], columns=['day','time'],
aggfunc={'size':'mean','tip':'max','total_bill':'sum'})
```

**Plotting Graphs:**
```python
df = pd.read_csv('expense_data.csv')
df['Category'].value_counts().plot(kind='pie')
```

**Convert date column to datetime and extract month:**
```python
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month_name()
```

**Pivot Table Visualization:**
```python
df.pivot_table(index='Month', columns='Category', values='INR', fill_value=0).plot()
df.pivot_table(index='Month', columns='Income/Expense', values='INR', aggfunc='sum').plot()
```



---



## Conclusion
This repository currently focuses on Pandas **Series**, demonstrating various ways to create, manipulate, and analyze Series objects. In future updates, we will explore **DataFrames** and other essential Pandas functionalities.

Stay tuned for more updates! 🚀

