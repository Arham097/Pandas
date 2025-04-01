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

---



## Conclusion
This repository currently focuses on Pandas **Series**, demonstrating various ways to create, manipulate, and analyze Series objects. In future updates, we will explore **DataFrames** and other essential Pandas functionalities.

Stay tuned for more updates! 🚀

