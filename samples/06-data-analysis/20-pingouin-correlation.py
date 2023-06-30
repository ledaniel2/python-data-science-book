import seaborn as sns
import pingouin as pg

# Load the Iris dataset
iris = sns.load_dataset('iris')

# Calculate partial correlation between sepal_length and
# sepal_width, controlling for petal_length and petal_width
partial_corr = pg.partial_corr(data=iris, x='sepal_length', 
    y='sepal_width', covar=['petal_length', 'petal_width'])

print(partial_corr)
