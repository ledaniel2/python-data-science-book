# Chapter 8: Data Manipulation with Python

Effective data manipulation is crucial for deriving meaningful insights and preparing data for machine learning algorithms. As you progress in your data science journey, you will encounter increasingly complex datasets, making advanced data manipulation techniques indispensable.

We will explore advanced `pandas` techniques that will help you harness the full potential of this powerful library. You will learn to group, merge, and reshape data in ways that facilitate deeper analysis and enable you to uncover hidden relationships within the data. These techniques will allow you to combine and organize datasets with different structures, making it easier to work with diverse sources of data.

Additionally, you will learn how to apply custom functions to your data, enabling you to perform complex calculations and transformations. This skill will empower you to create highly tailored solutions for specific data processing needs, further enhancing your data manipulation capabilities.

Our learning goals for this chapter are:

 * Acquire advanced data manipulation skills using `pandas` for more sophisticated data processing and analysis.
 * Learn to group, merge, and reshape data to uncover hidden relationships and facilitate deeper analysis.
 * Gain proficiency in applying custom functions to data, enabling you to perform complex calculations and transformations.

## 8.1: Advanced `pandas` techniques

We will explore advanced data manipulation techniques in `pandas`, which will enable you to manage, transform, and analyze complex hierarchical data more effectively. We will discuss the concepts of multi-indexing, pivoting, and stacking/unstacking, as well as their practical applications in various data manipulation tasks. By understanding and mastering these techniques, you will be better equipped to handle data with multiple levels of hierarchy, restructure it to suit your needs, and uncover valuable insights hidden within your datasets. So, let's dive into these powerful data manipulation tools and learn how to harness their full potential in your data science journey.

### Multi-indexing

Multi-indexing, also known as hierarchical indexing, is a feature in the `pandas` library that allows you to work with higher-dimensional data using a lower-dimensional data structure like a DataFrame or Series. It enables you to organize and manipulate data with multiple levels of hierarchy or multiple keys in rows and columns, providing a way to represent complex data relationships.

In `pandas`, a MultiIndex object is created by combining two or more columns or index levels into a single index structure. This allows you to represent data with multiple levels of indexing or labeling on one or both axes (rows and columns) of a DataFrame or Series.

Some common uses of multi-indexing in `pandas` include:

 1. Grouping and aggregation: Multi-indexing can be used to group data based on multiple keys or categories and perform aggregation operations like `sum`, `mean`, or `count`.

 2. Reshaping data: You can use multi-indexing to pivot, stack, or unstack your data, which helps in changing the layout or structure of your data to suit your needs.

 3. Slicing and querying: With multi-indexing, you can perform advanced selection and filtering operations on your data based on multiple keys or levels of hierarchy.

 4. Merging and joining: Multi-indexing facilitates merging and joining operations on data with complex hierarchical structures.

Here's an example of creating a MultiIndex DataFrame:

```python
import pandas as pd

# Data
data = [
    ('A', 'X', 1),
    ('A', 'Y', 2),
    ('A', 'Z', 3),
    ('B', 'X', 4),
    ('B', 'Y', 5),
    ('B', 'Z', 6)
]

# Create a MultiIndex DataFrame
index = pd.MultiIndex.from_tuples([(row[0], row[1]) for row in data], names=['Letter', 'Coordinate'])
df = pd.DataFrame([row[2] for row in data], index=index, columns=['Value'])

print(df)
```

This will output:

```plaintext
                Value
Letter Coordinate      
A      X            1
       Y            2
       Z            3
B      X            4
       Y            5
       Z            6
```

You can access subsets of the data using the `.loc[]` method:

```python
# Access data for letter 'A'
print(df.loc['A'])
```

This will output:

```plaintext
Coordinate
X               1
Y               2
Z               3
```

```python
# Access data for letter 'A' and coordinate 'Y'
print(df.loc['A','Y'])
```

Output:

```plaintext
Value    2
Name: (A, Y), dtype: int64
```

### Pivoting

Pivoting is a data transformation technique in `pandas `that allows you to reshape or restructure a DataFrame by changing its layout. In the context of `pandas`, pivoting usually involves turning a long-format DataFrame (where data is spread across multiple rows) into a wide-format DataFrame (where data is spread across multiple columns). It is particularly useful when you have data with multiple levels of hierarchy and want to change the way it is presented.

The `pivot()` function in pandas is used to perform the pivoting operation on a DataFrame. It takes three main arguments:

 * `index`: The column to use as the new DataFrame's row index.
 * `columns`: The column to use as the new DataFrame's column labels.
 * `values`: The column(s) to use as the new DataFrame's data values.

Here's an example to demonstrate pivoting:

Consider the following long-format DataFrame:

```plaintext
   Category Year  Value
0      A    2021     10
1      A    2022     11
2      B    2021     20
3      B    2022     21
```

To pivot this DataFrame and make 'Category' the row index, 'Year' the columns, and 'Value' the data values, you can use the `pivot()` function:

```python
import pandas as pd

# Creating a long-format DataFrame
data = {
    'Category': ['A', 'A', 'B', 'B'],
    'Year': [2021, 2022, 2021, 2022],
    'Value': [10, 11, 20, 21]
}
long_df = pd.DataFrame(data)

# Pivoting the DataFrame
wide_df = long_df.pivot(index='Category', columns='Year', values='Value')

print(wide_df)
```

This will output the following wide-format DataFrame:

```plaintext
Year      2021  2022
Category            
A           10    11
B           20    21
```

In some cases, you may encounter duplicate entries in the combination of index and column values. In such cases, you will need to use the `pivot_table()` function, which allows you to define an aggregation function to handle duplicate entries, such as `mean`, `sum`, `min`, or `max`.

In summary, pivoting is a technique in `pandas` that allows you to transform the layout of a DataFrame by changing its row and column indices, making it easier to analyze and visualize hierarchical data.

### Stacking and Unstacking

Stacking and unstacking are operations in `pandas` that allow you to reshape the layout of a DataFrame or a Series by rearranging the data along the index levels. These operations help in changing the structure of your data to suit your analysis or visualization needs.

Stacking is the process of pivoting a DataFrame's columns into rows, resulting in a "taller" and narrower DataFrame or Series with a MultiIndex. The `stack()` function is used to perform this operation, and it moves the innermost column level to the innermost row level, creating a MultiIndex in the process. Stacking is useful when you want to convert a wide-format DataFrame (with multiple columns) into a long-format DataFrame (with fewer columns and more rows).

Here's an example of stacking:

```python
import pandas as pd

# Creating a wide-format DataFrame
data = {
    '2021': [10, 20],
    '2022': [11, 21]
}
wide_df = pd.DataFrame(data, index=['A', 'B'])

# Stacking the DataFrame
stacked_df = wide_df.stack()

print(stacked_df)
```

This will output the following long-format DataFrame:

```plaintext
A  2021    10
   2022    11
B  2021    20
   2022    21
dtype: int64
```

Unstacking is the inverse operation of stacking, where a DataFrame's or a Series' MultiIndex rows are pivoted into columns, resulting in a "wider" and shorter DataFrame. The `unstack()` function is used to perform this operation, and it moves the innermost row level to the innermost column level. Unstacking is useful when you want to convert a long-format DataFrame (with fewer columns and more rows) into a wide-format DataFrame (with multiple columns).

Here's an example of unstacking:

```python
import pandas as pd

# Creating a long-format DataFrame with a MultiIndex
index = pd.MultiIndex.from_tuples([('A', 2021), ('A', 2022), ('B', 2021), ('B', 2022)])
data = [10, 11, 20, 21]
long_df = pd.Series(data, index=index)

# Unstacking the DataFrame
unstacked_df = long_df.unstack()

print(unstacked_df)
```

This will output the following wide-format DataFrame:

```plaintext
   2021  2022
A    10    11
B    20    21
```

In summary, stacking and unstacking are techniques in `pandas` that allow you to reshape your data by rearranging the index levels. Stacking pivots columns into rows, creating a MultiIndex, while unstacking pivots MultiIndex rows into columns. These operations help you restructure your data for analysis, visualization, or other data manipulation tasks.

## 8.2: Grouping, merging, and melting data

In the world of data analysis, organizing, processing, and transforming data are crucial steps to gain valuable insights and derive useful information. This often involves a variety of techniques such as grouping, merging, and reshaping data to create a more suitable structure for further analysis and visualization. We will explore these essential techniques using `pandas`, which provides various functions and tools to help you manipulate and manage your data efficiently. By understanding and applying these techniques, you will be well-equipped to handle complex data manipulation tasks and streamline your data analysis workflow.

### Grouping Data

Grouping data is a common operation in data analysis, where you need to categorize or organize data based on certain criteria and then perform some aggregation or transformation operations on each group. In `pandas`, the `groupby()` function is used to group data in a DataFrame or Series based on specified column(s) or index level(s).

The `groupby()` function returns a special `GroupBy` object, which is a collection of groups where each group consists of the grouped data and a unique group label. You can perform various operations on this `GroupBy` object, such as aggregation, transformation, or filtering. Some common operations that can be applied to grouped data include:

 1. Aggregation: Calculate summary statistics like count, sum, mean, median, or standard deviation for each group. You can use the `agg()` or `aggregate()` functions to apply multiple aggregation functions at once.

 2. Transformation: Perform operations on each group's data while maintaining the original shape of the DataFrame. You can use the `transform()` function to apply a custom or built-in function to each group.

 3. Filtering: Filter groups based on certain conditions, such as the size of the group or the value of a specific statistic. You can use the `filter()` function to apply a custom function that returns a boolean value for each group.

Here's an example of grouping data using `pandas`:

```python
import pandas as pd

# Create a DataFrame
data = {
    'Category': ['A', 'A', 'B', 'B', 'A', 'B'],
    'Year': [2021, 2022, 2021, 2022, 2021, 2022],
    'Value': [10, 11, 20, 21, 12, 22]
}
df = pd.DataFrame(data)

# Group the data by 'Category'
grouped = df.groupby('Category')

# Calculate the mean value for each group
mean_values = grouped['Value'].mean()

print(mean_values)
```

This will output the following mean values for each category:

```plaintext
Category
A    11.0
B    21.0
Name: Value, dtype: float64
```

You can also group by multiple columns or index levels by passing a list of column names or index levels to the `groupby()` function.

In summary, grouping data is an essential technique in data analysis that allows you to categorize and organize your data based on specific criteria. The `pandas` library provides the `groupby()` function and various operations like aggregation, transformation, and filtering to work efficiently with grouped data in DataFrames and Series.

### Merging Data

Merging data is a process of combining two or more datasets based on a common key or set of keys, also known as joining in the context of relational databases. In `pandas`, the `merge()` function is used to merge DataFrames based on specified column(s) or index level(s). This operation is particularly useful when you need to bring together data from different sources or tables that share a common relationship.

The `merge()` function takes two DataFrames as its main arguments and offers several optional parameters to control the merging behavior. Some important parameters include:

 * `on`: The column(s) or index level(s) to use as the key(s) for the merge operation. If not specified, pandas will use the common columns between the two DataFrames as the key(s).
 * `left_on` and `right_on`: The column(s) or index level(s) from the left and right DataFrames to use as the key(s) if they have different names.
 * `left_index` and `right_index`: Boolean flags to indicate whether to use the index of the left and/or right DataFrames as the key(s) for the merge operation.
 * `how`: The type of merge to be performed, which can be one of the following:
     * `inner`: The default option, returns only the rows with matching keys in both DataFrames.
     * `outer`: Returns all rows from both DataFrames, filling missing values with NaN for non-matching keys.
     * `left`: Returns all rows from the left DataFrame and the matching rows from the right DataFrame, filling missing values with NaN for non-matching keys in the right DataFrame.
     * `right`: Returns all rows from the right DataFrame and the matching rows from the left DataFrame, filling missing values with NaN for non-matching keys in the left DataFrame.

Here's an example of merging data using `pandas`:

```python
import pandas as pd

# Create two DataFrames
data1 = {'Key': ['A', 'B', 'C'], 'Value1': [1, 2, 3]}
data2 = {'Key': ['B', 'C', 'D'], 'Value2': [4, 5, 6]}
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# Merge the DataFrames on the 'Key' column
merged_df = pd.merge(df1, df2, on='Key', how='inner')

print(merged_df)
```

This will output the following merged DataFrame:

```plaintext
  Key  Value1  Value2
0   B       2       4
1   C       3       5
```

In summary, merging data is a crucial operation in data analysis, allowing you to combine different datasets based on a common key or set of keys. The `pandas` library provides the `merge()` function and various options to control the merging behavior, making it easy to join DataFrames in different ways, similar to joins in relational databases.

### Melting Data

Melting data is the process of transforming a wide-format DataFrame (with multiple columns) into a long-format DataFrame (with fewer columns and more rows). It is also known as "unpivoting" or "reshaping" data. The `melt()` function in `pandas` is used for melting data by specifying identifier variables and value variables. Melting is particularly useful for converting data from a wide format, often used in spreadsheets, to a long format, which is more suitable for statistical analysis, machine learning, and visualization tasks.

The `melt()` function takes the following important parameters:

 * `id_vars`: A list of columns to use as identifier variables. These columns will be kept unchanged in the melted DataFrame.
 * `value_vars`: A list of columns to be melted into a single value column. If not specified, all columns not set as identifier variables will be considered as value variables.
 * `var_name`: A string to use as the name of the new column that will contain the variable names from the melted columns. If not specified, a default name will be used.
 * `value_name`: A string to use as the name of the new column that will contain the values from the melted columns. If not specified, a default name will be used.

Here's an example of melting data using `pandas`:

```python
import pandas as pd

# Create a wide-format DataFrame
data = {
    'ID': [1, 2, 3],
    'Category_A': [10, 11, 12],
    'Category_B': [20, 21, 22],
    'Category_C': [30, 31, 32]
}
wide_df = pd.DataFrame(data)

# Melt the DataFrame
long_df = pd.melt(wide_df, id_vars=['ID'], value_vars=['Category_A', 'Category_B', 'Category_C'],
                  var_name='Category', value_name='Value')

print(long_df)
```

This will output the following long-format DataFrame:

```plaintext
   ID    Category  Value
0   1  Category_A     10
1   2  Category_A     11
2   3  Category_A     12
3   1  Category_B     20
4   2  Category_B     21
5   3  Category_B     22
6   1  Category_C     30
7   2  Category_C     31
8   3  Category_C     32
```

In summary, melting data is a crucial operation in data analysis that allows you to reshape a wide-format DataFrame into a long-format DataFrame. This transformation makes your data more suitable for various statistical analysis, machine learning, and visualization tasks. The `pandas` library provides the `melt()` function with different parameters to control the melting process, making it easy to reshape your data as needed.

## 8.3: Applying Functions to Data

Data analysis often requires applying various functions to manipulate, transform, or aggregate the data for better understanding and interpretation. In `pandas`, apply functions are a set of methods that allow you to apply custom or built-in functions to the elements, rows, or columns of a DataFrame or Series. They are useful when you need to perform complex data manipulation, transformation, or aggregation tasks that are not covered by the standard `pandas` functions.

### Applying a function to a Series

The `apply()` function can be used to apply a custom or built-in function to each element in a Series. It takes a function as its argument and returns a new Series with the function applied to each element.

```python
import pandas as pd

# Create a Series
s = pd.Series([1, 2, 3, 4])

# Apply a custom function to each element
s_squared = s.apply(lambda x: x**2)

print(s_squared)
```

This will output:

```plaintext
0     1
1     4
2     9
3    16
dtype: int64
```

### Applying a function to a column or row

The `apply()` function is also used to apply a custom or built-in function to each column or row of a DataFrame along a specified `axis` parameter (0 for columns, 1 for rows). It takes a function as its argument and returns a new DataFrame or Series with the function applied to each column or row.

```python
import pandas as pd

# Create a DataFrame
data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)

# Apply a custom function to each column
col_sum = df.apply(lambda x: x.sum(), axis=0)

print(col_sum)
```

This will output:

```plaintext
A     6
B    15
dtype: int64
```

### Applying a function to elements

The `applymap()` function is used to apply a custom or built-in function to each element in a DataFrame. It takes a function as its argument and returns a new DataFrame with the function applied to each element.

```python
import pandas as pd

# Create a DataFrame
data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)

# Apply a custom function to each element
df_squared = df.applymap(lambda x: x**2)

print(df_squared)
```

This will output:

```plaintext
   A   B
0  1  16
1  4  25
2  9  36
```

### Aggregation of columns or rows

The `agg()` (or `aggregate()`) functions are used to apply one or more aggregation functions to the columns or rows of a DataFrame along a specified axis (0 for columns, 1 for rows). They can take a single function, a list of functions, or a dictionary of functions with column or row labels as keys.

```python
import pandas as pd

# Create a DataFrame
data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)

# Apply multiple aggregation functions to each column
result = df.agg(['sum', 'mean'], axis=0)

print(result)
```

This will output:

```plaintext
        A     B
sum   6.0  15.0
mean  2.0   5.0
```

In summary, apply functions are powerful tools in `pandas` that allow you to apply custom or built-in functions to your data for complex data manipulation, transformation, and aggregation tasks. They provide flexibility and adaptability when working with DataFrames and Series in various data analysis scenarios.
