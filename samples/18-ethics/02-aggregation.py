import numpy as np
import pandas as pd

age_data = pd.DataFrame({'Age': [25, 30, 35, 20, 40, 45, 50]})
age_data['Age_Range'] = pd.cut(age_data['Age'], bins=np.arange(10, 61, 10), right=False)
print(age_data.groupby('Age_Range').size())
