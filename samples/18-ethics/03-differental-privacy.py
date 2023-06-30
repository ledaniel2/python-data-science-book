import numpy as np
import pandas as pd

age_data = pd.DataFrame({'Age': [25, 30, 35, 20, 40, 45, 50]})

def laplace_mechanism(sensitive_value, epsilon):
    scale = 1 / epsilon
    noise = np.random.laplace(0, scale)
    return sensitive_value + noise

age_sum = age_data['Age'].sum()
epsilon = 0.1
private_age_sum = laplace_mechanism(age_sum, epsilon)
print('Age sum with differential privacy:', private_age_sum)
