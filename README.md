## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```

<img width="312" height="320" alt="image" src="https://github.com/user-attachments/assets/c3852b41-e053-4c0c-892c-0143b8773d06" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

<img width="147" height="197" alt="image" src="https://github.com/user-attachments/assets/911143df-f274-46d8-b188-099eacf3f09b" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

<img width="345" height="331" alt="image" src="https://github.com/user-attachments/assets/620888c1-96ae-4222-978e-09a97f8763a9" />


```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```

<img width="435" height="326" alt="image" src="https://github.com/user-attachments/assets/cf4ff79e-ea4f-48cf-90ff-ebdde6d6682c" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
```
df2=pd.concat([df2,enc],axis=1)
df2
```

<img width="662" height="328" alt="image" src="https://github.com/user-attachments/assets/d1b0d8d8-2f18-4a51-a795-63a664e770a7" />

```
pd.get_dummies(df2,columns=["nom_0"])
```

<img width="466" height="317" alt="image" src="https://github.com/user-attachments/assets/8d0a0c74-2d47-47e4-abcf-8a81e94a3516" />


```
df=pd.read_csv("data.csv")
df
```

<img width="472" height="315" alt="image" src="https://github.com/user-attachments/assets/693e5ea0-79e6-45bb-b2bd-88a70afecdd2" />

```
nd=(df['Ord_2'])
df
```

<img width="560" height="328" alt="image" src="https://github.com/user-attachments/assets/7a65f4e6-f175-4532-be1f-20cd294b02c8" />

```
dfb=pd.concat([df,nd],axis=1)
dfb
```

<img width="732" height="387" alt="image" src="https://github.com/user-attachments/assets/0dd9110e-1b15-4ea0-932a-3e23e5fb61a4" />

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```

<img width="366" height="107" alt="image" src="https://github.com/user-attachments/assets/553d0eda-1b7c-46ee-813c-48def48cda9b" />

```
df.skew()
```

<img width="527" height="232" alt="image" src="https://github.com/user-attachments/assets/7dd0a609-01aa-46c3-bd89-acd85d3dbcc7" />

```
np.log(df["Highly Positive Skew"])
```

<img width="530" height="227" alt="image" src="https://github.com/user-attachments/assets/140e17f9-e8b7-4ce5-b4c2-bb778c4aff92" />

```
np.reciprocal(df["Moderate Positive Skew"])
```

<img width="611" height="243" alt="image" src="https://github.com/user-attachments/assets/e24d8223-7671-4d7e-b0bf-3fd0da2ee588" />

```
np.sqrt(df["Highly Positive Skew"])
```

<img width="627" height="245" alt="image" src="https://github.com/user-attachments/assets/6fd7fa7f-8f3e-4066-97c1-9d2ea664b5c5" />


```
np.square(df["Highly Positive Skew"])
```

<img width="992" height="395" alt="image" src="https://github.com/user-attachments/assets/3b79ec15-42a7-4f5b-b05f-553283e23ccd" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

<img width="443" height="122" alt="image" src="https://github.com/user-attachments/assets/c0d0746c-e15e-49f1-aad0-5263046d4384" />

```
df.skew()
```

<img width="523" height="146" alt="image" src="https://github.com/user-attachments/assets/2ac627eb-6172-4270-a1d4-30a66bd8a429" />


```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

<img width="1115" height="406" alt="image" src="https://github.com/user-attachments/assets/54b922d2-7753-4045-9823-e9411f146166" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

<img width="765" height="485" alt="image" src="https://github.com/user-attachments/assets/6fa06994-1620-4c0e-a3c7-781c22face47" />

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="751" height="482" alt="image" src="https://github.com/user-attachments/assets/399a97c9-fa23-4ce9-9848-6af335d3d321" />


```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

<img width="876" height="491" alt="image" src="https://github.com/user-attachments/assets/d12a413a-c5d2-4c4e-9f99-f393cb0c7c38" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="717" height="482" alt="image" src="https://github.com/user-attachments/assets/b2ac06c3-501d-429e-8ecd-d2aaad361d2a" />

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

<img width="786" height="387" alt="image" src="https://github.com/user-attachments/assets/5108af16-d485-419d-b335-fb77bca7360e" />

```
dt=pd.read_csv("Data_to_Transform.csv")
dt
```

<img width="730" height="482" alt="image" src="https://github.com/user-attachments/assets/106276e1-fd87-4071-9b96-f1d5b02edad8" />

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

<img width="730" height="482" alt="Screenshot 2025-09-29 111107" src="https://github.com/user-attachments/assets/8680b99c-72ec-4ba6-931b-d05f03f49200" />




# RESULT:
Thus,To read the given data and perform Feature Encoding and Transformation process and save the data to a file is completed.

       
