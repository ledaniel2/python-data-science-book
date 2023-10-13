# Chapter 7: Feature Engineering and Selection

In the world of data science, the quality of input features has a significant impact on the performance of machine learning models. Properly selecting and engineering features can lead to more accurate and efficient models, ultimately providing better insights and predictions.

We will explore the art of creating new features from existing data, a technique that can reveal hidden patterns and relationships within the dataset. We will also discuss the importance of feature scaling and transformation to ensure that your data is prepared and compatible with various machine learning algorithms.

Feature selection is another essential aspect of model building, as it helps identify the most relevant features for a given problem. By incorporating only the most significant features, you can build models that are more interpretable, less prone to overfitting, and computationally efficient. We will introduce several feature selection methods, including filter, wrapper, and embedded techniques, that will help you choose the best features for your project.

Our learning goals for this chapter are:

 * Learn to create new features that can potentially improve model performance.
 * Understand the importance of feature scaling and transformation.
 * Gain proficiency in various feature selection methods to choose the most relevant features for your machine learning models.

## 7.1: Creating New Features

We will explore various techniques for creating new features, including mathematical transformations, binning, interaction features, aggregation features, feature extraction from text and dates, and feature engineering with domain knowledge. By understanding and applying these techniques, students will be better equipped to build robust and accurate models for data analysis.

### Mathematical Transformations

Mathematical transformations involve applying mathematical functions to existing features in order to create new ones. These transformations can help enhance relationships between variables, handle skewed data, and normalize features. Mathematical transformations can often reveal interesting patterns in the data that were not apparent before. Common mathematical transformations include:

 * Log transformation: Useful for reducing the impact of outliers and handling right-skewed data.
 * Square root and cube root transformations: Helpful for reducing the effects of outliers in moderately skewed data.
 * Exponential and power transformations: Can be used to handle left-skewed data.
 * Polynomial features: Create new features by raising existing features to different powers or combining them using arithmetic operations.

Here are some examples of performing mathematical transformations using NumPy and `pandas`:

```python
import numpy as np
import pandas as pd

# Example DataFrame
data = pd.DataFrame({'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

# Log transformation
data['log_value'] = np.log(data['value'])

# Square root transformation
data['sqrt_value'] = np.sqrt(data['value'])

# Cube root transformation
data['cbrt_value'] = np.cbrt(data['value'])

# Exponential transformation
data['exp_value'] = np.exp(data['value'])

# Power transformation
data['power_value'] = np.power(data['value'], 2)
```

### Binning

Binning is a technique used to convert continuous features into categorical ones by dividing the data into intervals or "bins." This can help simplify complex relationships between variables and improve model interpretability. There are several binning methods, such as equal-width binning, equal-frequency binning, and adaptive binning using clustering algorithms like k-means. Binning can help reduce the noise in the data and reveal underlying patterns. In `pandas`, you can use the `cut()` or `qcut()` functions to create bins.

For example, if we want to create age categories in a sample dataset, we can use the following code:

```python
import numpy as np
import pandas as pd

# Example DataFrame
data = pd.DataFrame({'age': [2, 9, 12, 17, 21, 22, 24, 30, 35, 41, 49, 50, 55, 70, 89]})

# Create age categories using custom bins
bins = [0, 18, 35, 60, 100]
labels = ['Child', 'Young Adult', 'Adult', 'Senior']
data['age_category'] = pd.cut(data['age'], bins=bins, labels=labels)
print(data)
```

This will output:

```plaintext
    age age_category
0     2        Child
1     9        Child
2    12        Child
3    17        Child
4    21  Young Adult
5    22  Young Adult
6    24  Young Adult
7    30  Young Adult
8    35  Young Adult
9    41        Adult
10   49        Adult
11   50        Adult
12   55        Adult
13   70       Senior
14   89       Senior
```

Alternatively, to create a specified number of bins either spanning identical age ranges ("equal-width binning") or with the same number of samples in each bin ("equal-frequency binning"), we can use:

```python
# Equal-width binning
data['value_bin_eq_width'] = pd.cut(data['age'], bins=3)

# Equal-frequency binning
data['value_bin_eq_freq'] = pd.qcut(data['age'], q=3)

print(data)
```

This will output:

```plaintext
    age age_category value_bin_eq_width value_bin_eq_freq
0     2        Child      (1.913, 31.0]   (1.999, 21.667]
1     9        Child      (1.913, 31.0]   (1.999, 21.667]
2    12        Child      (1.913, 31.0]   (1.999, 21.667]
3    17        Child      (1.913, 31.0]   (1.999, 21.667]
4    21  Young Adult      (1.913, 31.0]   (1.999, 21.667]
5    22  Young Adult      (1.913, 31.0]  (21.667, 43.667]
6    24  Young Adult      (1.913, 31.0]  (21.667, 43.667]
7    30  Young Adult      (1.913, 31.0]  (21.667, 43.667]
8    35  Young Adult       (31.0, 60.0]  (21.667, 43.667]
9    41        Adult       (31.0, 60.0]  (21.667, 43.667]
10   49        Adult       (31.0, 60.0]    (43.667, 89.0]
11   50        Adult       (31.0, 60.0]    (43.667, 89.0]
12   55        Adult       (31.0, 60.0]    (43.667, 89.0]
13   70       Senior       (60.0, 89.0]    (43.667, 89.0]
14   89       Senior       (60.0, 89.0]    (43.667, 89.0]
```

### Interaction Features

Interaction features capture the combined effect of two or more features on the target variable. These features can be created by multiplying, dividing, adding, or subtracting the original features. Interaction features can help reveal complex relationships and improve model performance.

Suppose we have two categories of numerical data, `A` and `B`. It is possible to create records of the interactions between the two using mathematical operators:

```python
import pandas as pd

# Example DataFrame
data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Multiplication
data['AB_mult'] = data['A'] * data['B']

# Division
data['AB_div'] = data['A'] / data['B']

# Addition
data['AB_add'] = data['A'] + data['B']

# Subtraction
data['AB_sub'] = data['A'] - data['B']

print(data)
```

When examining the output below, the interaction feature `AB_sub` appears to be of interest as it is identical for each row:

```plaintext
   A  B  AB_mult  AB_div  AB_add  AB_sub
0  1  4        4    0.25       5      -3
1  2  5       10    0.40       7      -3
2  3  6       18    0.50       9      -3
```

As another example, suppose we have a dataset with columns `height` in centimeters and `weight` in kilograms. We can create a new feature called `bmi` (Body Mass Index) using the formula *BMI = weight in kilograms / (height in meters squared)* as follows:

```python
data['bmi'] = data['weight'] / (data['height'] / 100) ** 2
```

If the units were in inches and pounds, the scaling of the formula could be adjusted to cater for this.

### Aggregation Features

Aggregation features are created by summarizing or grouping data from multiple records or columns based on certain conditions. They can provide valuable insights into the relationship between features and the target variable, especially in cases of hierarchical or time-series data. Common aggregation functions include `sum`, `average`, `minimum`, `maximum`, and `count`. These functions can help capture high-level patterns in the data that might not be visible otherwise.

Suppose we have two categories of numerical data, `A` and `B`. It is possible to create a new dataset with the results of various aggregation functions for each category. The `reset_index()` function causes the dataset to be indexed by `0`, `1` and so on, instead of using the category names:

```python
import pandas as pd

# Example DataFrame
data = pd.DataFrame({'category': ['A', 'A', 'B', 'B', 'A', 'B'],
                     'value': [1, 2, 3, 4, 5, 6]})

# Aggregation features
agg_features = data.groupby('category')['value'].agg(['sum', 'mean', 'min', 'max', 'count']).reset_index()
print(agg_features)
```

This will output:

```plaintext
  category  sum      mean  min  max  count
0        A    8  2.666667    1    5      3
1        B   13  4.333333    3    6      3
```

As another example, suppose we have a dataset with columns `price` and `category`. We can create a new feature representing the average price per category as follows:

```python
data['average_price_per_category'] = data.groupby('category')['price'].transform('mean')
```

### Feature Extraction from Text and Dates

Text and date features often require special preprocessing steps to extract valuable information. For text features, techniques like tokenization, stemming, and stopword removal can be applied to convert raw text into a structured format. Then, text vectorization methods like Bag-of-Words or TF-IDF can be used to create numerical features.

For date features, useful information can be extracted by creating new features such as day of the week, month, year, time since a particular event, or time until the next event. These derived features can help reveal patterns and trends in the data.

Here is an example of how to use text vectorization methods on a `pandas` dataframe containing phrases of text:

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Example text data
text_data = pd.DataFrame({'text': ['I love data science', 'Machine learning is amazing', 'Python is great']})

# Text vectorization - Bag-of-Words
bow_vectorizer = CountVectorizer()
text_bow = bow_vectorizer.fit_transform(text_data['text'])

# Text vectorization - TF-IDF
tfidf_vectorizer = TfidfVectorizer()
text_tfidf = tfidf_vectorizer.fit_transform(text_data['text'])
```

Often, datasets may contain text or date columns that can be used to create new features, for example by:

 * Extracting day, month, and year from a date column
 * Calculating the length of a text column
 * Counting the number of words or unique words in a text column

Here's an example of extracting features from a date column:

```python
import pandas as pd

# Example date data
date_data = pd.DataFrame({'date': pd.date_range(start='2020-01-01', periods=5, freq='D')})

# Extracting date features
date_data['day_of_week'] = date_data['date'].dt.dayofweek
date_data['month'] = date_data['date'].dt.month
date_data['year'] = date_data['date'].dt.year
```

And here's another example of extracting features from a text column:

```python
import pandas as pd

data = pd.DataFrame({'text_data': ['One man went to mow, went to mow a meadow', 'Whether the weather be fine, whether the weather be not']})
# Calculate the length of each text entry
data['text_length'] = data['text_data'].str.len()

# Count the number of words in each text entry
data['word_count'] = data['text_data'].str.split().str.len()

# Count the number of unique words in each text entry
data['unique_word_count'] = data['text_data'].apply(lambda x: len(set(x.lower().split())))
```

### Feature Engineering with Domain Knowledge

Domain knowledge plays a crucial role in feature engineering. By incorporating insights from subject matter experts, new features can be created that better capture the underlying patterns in the data. This can lead to improved model performance and more accurate predictions.

Suppose we have a dataset containing information about houses, and we want to create a new feature called `age` using the `year_built` column.

```python
# Example DataFrame
data = pd.DataFrame({'year_built': [1990, 2000, 2010, 2020]})

# Using domain knowledge to create a new feature
current_year = pd.Timestamp.now().year
data['age'] = current_year - data['year_built']
```

As a more in-depth study, suppose you're working with a dataset of real estate transactions. You might create new features based on your domain knowledge, such as:

 * Distance to the nearest public transportation station
 * Number of schools within a certain radius
 * Crime rate in the neighborhood

To create these features, you may need to combine data from different sources and apply spatial or statistical calculations.

Remember that feature engineering is more of an art than a science, and it often requires experimentation and iteration. Don't be afraid to try out different ideas and techniques to find the best set of features for your problem.

## 7.2: Feature scaling and transformation

Feature scaling and transformation are essential steps in the process of preparing data for machine learning and data science tasks. These techniques help to standardize and normalize the data, ensuring that the features have similar scales and distributions. This is particularly important because many machine learning algorithms, such as linear regression, support vector machines (SVM) and neural networks, are sensitive to the scale of the input features.

When features are on different scales, the model may give more importance to features with higher magnitudes, leading to suboptimal performance. Feature scaling and transformation help normalize the data, ensuring that each feature contributes equally to the model. Proper feature scaling and transformation can improve the performance of these algorithms and make the model more interpretable.

We will explore the importance of feature scaling and transformation, discuss different methods like Min-max scaling, Standard Scaling (Z-score Normalization), and Log Transformation, and provide guidance on selecting the appropriate method. We will also illustrate these concepts with Python code examples.

In summary, feature scaling and transformation are important for ensuring:

 * Uniformity: Ensuring that features have the same scale allows for better comparison and interpretation of feature importance.
 * Algorithm Performance: Some machine learning algorithms assume that the input features have a specific scale or distribution, and applying feature scaling and transformation can help meet these assumptions.
 * Convergence: Algorithms that use gradient descent for optimization can converge faster if the features are on the same scale.
 * Noise Reduction: Outliers and noise in the data can be minimized through scaling and transformation, which can improve the model's performance.
 
### Min-Max Scaling

Min-Max scaling, also known as normalization, scales the data to a specific range, typically [0, 1]. The formula for Min-Max scaling is: *scaled_value = (value - min) / (max - min)*

Here is an example using the `MinMaxScaler` class from the `sklearn.preprocessing` module:

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Example dataset
data = np.array([[100, 0.5],
                 [80, 0.4],
                 [120, 0.6],
                 [90, 0.55]])

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
print(scaled_data)
```

This will output:

```plaintext
[[0.5  0.5 ]
 [0.   0.  ]
 [1.   1.  ]
 [0.25 0.75]]
```

### Standard Scaling (Z-score Normalization)

Standard scaling, also known as Z-score normalization, transforms the features to have a mean of 0 and a standard deviation of 1. This is done by subtracting the mean of the feature and dividing it by the standard deviation of the feature: *scaled_value = (value - mean) / standard_deviation* or *x&lsquo; = (x - x&#772;) / &sigma;*

Here's an example using the `StandardScaler` class from the `sklearn.preprocessing` module:

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

# Sample data
data = np.array([[10, 200], [15, 180], [30, 210]])

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the data
scaled_data = scaler.fit_transform(data)
print(scaled_data)
```

This will output:

```plaintext
[[-0.98058068  0.26726124]
 [-0.39223227 -1.33630621]
 [ 1.37281295  1.06904497]]
```

### Log Transformation

Log transformation is a nonlinear transformation that can be applied to features with a skewed distribution or a wide range of values. This method helps to reduce the impact of outliers and can transform features with an exponential distribution into a more Gaussian-like distribution.

By applying a log transformation, we can make the data more symmetrical and easier for machine learning algorithms to process. Keep in mind that log transformation can only be applied to positive values.

Here's an example of how to apply log transformation to a dataset:

```python
import numpy as np

# Sample data
data = np.array([[10, 200], [15, 180], [30, 210]])

# Apply log transformation
log_transformed_data = np.log(data)
print(log_transformed_data)
```

This will output:

```plaintext
[[2.30258509 5.29831737]
 [2.7080502  5.19295685]
 [3.40119738 5.34710753]]
```

### Selecting the Appropriate Scaling and Transformation Method

Choosing the right scaling and transformation method depends on the dataset, the features, and the machine learning algorithm being used. Here are some guidelines to help you decide:

 1. Min-max scaling is useful when you want to preserve the original distribution of the data and transform it into a specific range.
 2. Standard scaling is suitable when the features have a Gaussian-like distribution and you want to standardize them to have zero mean and unit variance. This is particularly helpful when working with algorithms that assume input features are normally distributed.
 3. Log transformation is beneficial when dealing with features that have a skewed distribution, wide range of values, or exponential growth patterns. It can help reduce the impact of outliers and create a more symmetric distribution.

It's important to experiment with different scaling and transformation methods to determine which one works best for your specific dataset and problem. Additionally, keep in mind that some machine learning algorithms, such as decision trees and random forests, are less sensitive to feature scaling, and applying these techniques may not significantly impact their performance.

To further illustrate the process of selecting an appropriate method, let's consider an example:

Suppose we have a dataset with two features, `income` and `age`; `income` has a wide range of values and is heavily skewed, while `age` is normally distributed but on a different scale.

In this case, we could apply log transformation to the `income` feature to reduce the skewness and impact of outliers. For the `age` feature, we could use standard scaling to ensure it has zero mean and unit variance. This would result in a dataset with both features on a similar scale and more suitable for various machine learning algorithms.

Remember to always evaluate the performance of your model using different scaling and transformation techniques to identify the most suitable method for your problem. It's essential to apply the same scaling and transformation to both your training and test datasets to maintain consistency.

## 7.3: Feature selection methods

Feature selection is a crucial step in the machine learning pipeline, as it helps to identify the most relevant features for a specific problem. This not only reduces the complexity of the model but also improves its performance by removing irrelevant or redundant features. We will explore three main feature selection methods: Filter Methods, Wrapper Methods, and Embedded Methods.

First, let's load a sample dataset which we can apply feature selection methods to. The 'iris' dataset we met with `seaborn` is provided in the form of four independent variables (sepal and petal, length and width), and the dependent variable (species encoded as 0, 1, or 2):

```python
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

print('Independent Variable(s): X')
print(X[0:5])
print('Dependent Variable: y')
print(y)
```

This will output:

```plaintext
Independent Variable(s): X
[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]]
Dependent Variable: y
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
```

### Filter Methods

Filter methods are used to rank features based on their relevance to the target variable, without considering the interaction between features. These methods are typically fast and computationally efficient, but they may not always provide the best results. Common filter methods include:

 1. Variance Threshold: This method selects features with variance above a certain threshold, assuming that low-variance features contain less information.

```python
from sklearn.feature_selection import VarianceThreshold

# Instantiate the selector with a threshold
selector = VarianceThreshold(threshold=0.2)

# Fit and transform the dataset
X_high_variance = selector.fit_transform(X)
```

 2. Univariate Selection: This method selects the best features based on univariate statistical tests like chi-squared or ANOVA.

```python
from sklearn.feature_selection import SelectKBest, chi2

# Instantiate the selector, selecting the top 2 features
selector = SelectKBest(chi2, k=2)

# Fit and transform the dataset
X_best_features = selector.fit_transform(X, y)
```

### Wrapper Methods

Wrapper methods involve evaluating the performance of a specific model with different feature subsets. This approach is more computationally intensive but usually yields better results than filter methods. The most popular wrapper methods include:

 1. Recursive Feature Elimination (RFE): RFE involves fitting a model, ranking features based on their importance, and recursively eliminating the least important feature until the desired number of features is reached.

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Instantiate the model and the RFE selector
model = LogisticRegression(max_iter=1000)
selector = RFE(model, n_features_to_select=3)

# Fit and transform the dataset
X_selected_features = selector.fit_transform(X, y)
```

 2. Forward Selection: In this approach, features are added one at a time based on their contribution to the model's performance until no further improvements are observed.

```python
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

# Instantiate the model and the forward selector
model = LinearRegression()
selector = SequentialFeatureSelector(model, k_features=4, forward=True)

# Fit and transform the dataset
X_selected_features = selector.fit_transform(X, y)
```

### Embedded Methods

Embedded methods involve feature selection as part of the model training process. These methods usually provide a good balance between performance and computational efficiency. Some common embedded methods are:

 1. Lasso Regression: Lasso (Least Absolute Shrinkage and Selection Operator) is a linear regression model that uses L1 regularization to penalize high coefficients, resulting in feature selection.

```python
from sklearn.linear_model import Lasso

# Instantiate the Lasso model with an alpha value
model = Lasso(alpha=0.1)

# Fit the model
model.fit(X, y)

# Select features with non-zero coefficients
X_selected_features = X[:, model.coef_ != 0]
```

 2. Decision Trees: Decision tree-based models like Random Forest and Gradient Boosting automatically perform feature selection by splitting the data based on feature importance.

```python
from sklearn.ensemble import RandomForestClassifier

# Instantiate the Random Forest model
model = RandomForestClassifier()

# Fit the model
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_

# Set a threshold for feature importance
threshold = 0.1

# Select features with importance greater than the threshold
X_selected_features = X[:, importances > threshold]
```

In conclusion, feature selection methods play a vital role in improving the performance and interpretability of machine learning models. Filter methods are computationally efficient but may not provide the best results, while wrapper methods offer better performance at the cost of increased computation. Embedded methods strike a balance between performance and computational efficiency. Depending on the specific problem and dataset, you may choose to use one or a combination of these methods to optimize your machine learning pipeline.
