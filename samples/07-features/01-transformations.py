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

print(data)
