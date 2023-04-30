# In[1]:


# Import required libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the data
df = pd.read_csv('weatherAUS.csv')


# Data description
print(df.describe())

# To check null values in our dataset
print(df.isnull().sum())


# Separate categorical and discrete data
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
disc_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Plot categorical data
n_rows = (len(cat_cols) // 3) + (len(cat_cols) % 3 > 0) # Calculate number of subplot rows
plt.figure(figsize=(12, n_rows*4))
for i, col in enumerate(cat_cols):
    plt.subplot(n_rows, 3, i+1)
    sns.countplot(x=col, data=df)
    plt.title(col)
plt.tight_layout()
plt.show()

# Plot discrete data
n_rows = (len(disc_cols) // 3) + (len(disc_cols) % 3 > 0) # Calculate number of subplot rows
plt.figure(figsize=(12, n_rows*4))
for i, col in enumerate(disc_cols):
    plt.subplot(n_rows, 3, i+1)
    sns.histplot(x=col, data=df, kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()


# Drop columns that are not required
df = df.drop(['Date', 'Location', 'Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm'], axis=1)
df.info()
# Remove rows with missing values in the target variable
df = df.dropna(subset=['RainTomorrow'])

# Split the dataset into features and target variable
X = df.drop('RainTomorrow', axis=1)
y = df['RainTomorrow']



# In[2]:


X


# In[3]:


y


# In[4]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


X_train


# In[6]:


y_train


# In[7]:


# Separate the numerical and categorical features
numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train.select_dtypes(include=[np.object]).columns.tolist()

# Perform preprocessing for numerical features
imputer_num = SimpleImputer(strategy='mean')
scaler = StandardScaler()

X_train[numerical_features] = imputer_num.fit_transform(X_train[numerical_features])
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = imputer_num.transform(X_test[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])


# In[8]:


X_train


# In[9]:


y_train


# In[10]:


# One-hot encoding for categorical features
from sklearn.preprocessing import OneHotEncoder

# Identify categorical features

# Impute missing values in categorical features
from sklearn.impute import SimpleImputer

imputer_cat = SimpleImputer(strategy='most_frequent')

X_train[categorical_features] = imputer_cat.fit_transform(X_train[categorical_features])
X_test[categorical_features] = imputer_cat.transform(X_test[categorical_features])

# One-hot encoding
onehot = OneHotEncoder(handle_unknown='ignore')

X_train_onehot = onehot.fit_transform(X_train[categorical_features]).toarray()
X_test_onehot = onehot.transform(X_test[categorical_features]).toarray()

# Get the names of the encoded columns
encoded_feature_names = onehot.get_feature_names_out(categorical_features)

# Put transformed data back into DataFrames
X_train_onehot = pd.DataFrame(X_train_onehot, columns=encoded_feature_names)
X_test_onehot = pd.DataFrame(X_test_onehot, columns=encoded_feature_names)





# In[11]:


X_train_onehot


# In[12]:


X_train=X_train.reset_index()
X_test=X_test.reset_index()


# In[13]:


X_train


# In[14]:



# Concatenate the encoded features with the continuous features
X_train = pd.concat([X_train[numerical_features], X_train_onehot], axis=1)
X_test = pd.concat([X_test[numerical_features], X_test_onehot], axis=1)


# In[15]:


X_train


# In[16]:


y_train


# In[17]:


X_test


# In[18]:


y_test


# In[19]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
print('Logistic Regression')
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[20]:


from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[21]:

print('KNeighborsClassifier')
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[22]:

print('SVM')
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[23]:

print('DecisionTreeClassifier')
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[24]:

print('RandomForestClassifier')
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[ ]:




