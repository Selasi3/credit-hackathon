
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score


# In[2]:


#Load the dataset
data = pd.read_csv("./data/credit_risk.csv")


# In[3]:


# Display the first 10 rowa
data.head()


# In[4]:


#brief info of the dataset
data.info()


# In[5]:


#Statistics summary
data.describe()


# In[ ]:





# In[6]:


#Check for duplicate
data.duplicated().sum()


# In[7]:


#Check the percentage of null values in each column
null_percentage = data.isnull().sum() / len(data) * 100
null_percentage


# In[8]:


# Fill the NaN values in the 'Rate' and Emp_length column with the calculated mean
data['Rate'].fillna(data['Rate'].mean(), inplace=True)
data['Emp_length'].fillna(data['Emp_length'].mean(), inplace=True)



# In[9]:


#brief info of the dataset
data.info()


# In[10]:


categorical_columns = data.select_dtypes(include=['object']).columns
# Select categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Loop through each categorical column, display unique values count, and plot countplots
for col in categorical_columns:
    unique_values_count = data[col].nunique()
    print(f"Column: {col}, Unique Values: {unique_values_count}")
    print(data[col].value_counts(), data[col].value_counts(normalize=True))
    
    # Sort the categories in order for better visualization
    sorted_categories = data[col].value_counts().index
    
    # Plot count plot
    plt.figure(figsize=(5,5))
    sns.countplot(x=col, data=data, order=sorted_categories)
    plt.title(f'Count Plot of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)  # Rotate labels if necessary
    plt.tight_layout()
    plt.show()
    plt.show()


# In[11]:


numeric_columns = ['Age', 'Income', 'Emp_length', 'Amount', 'Rate', 'Percent_income', 'Cred_length']
plt.figure(figsize=(14, 12))

# Plot each numeric column
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(4, 2, i)  # Adjust based on the number of columns
    sns.histplot(data[col], kde=True)  # Histogram with KDE
    plt.title(f'Distribution of {col}')
    plt.tight_layout()

plt.show()


# In[12]:


# Set up the matplotlib figure
plt.figure(figsize=(14, 12))

# Plot each numeric column using boxplot
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(4, 2, i)
    sns.boxplot(x=data[col])
    plt.title(f'Boxplot of {col}')
    plt.tight_layout()

plt.show()


# In[13]:


#Handling ouliers
# data = data[data['Amount']<= 25000]
# data = data[data['Age']<= 60]
# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = data['Amount'].quantile(0.25)
Q3 = data['Amount'].quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Detect outliers
outliers = data[(data['Amount'] < lower_bound) | (data['Amount'] > upper_bound)]
print(f"Number of outliers in Amount: {outliers.shape[0]}")

# Remove outliers
data_cleaned = data[(data['Amount'] >= lower_bound) & (data['Amount'] <= upper_bound)]

data_cleaned = data_cleaned[(data_cleaned["Age"] < 80) & (data_cleaned['Emp_length'] < 80)]
print(f"Shape of the dataset after removing outliers: {data_cleaned.shape}")


# In[14]:


# Set up the matplotlib figure
plt.figure(figsize=(14, 12))

# Plot each numeric column using boxplot
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(4, 2, i)
    sns.boxplot(x=data_cleaned[col])
    plt.title(f'Boxplot of {col}')
    plt.tight_layout()

plt.show()


# In[15]:


data_cleaned.info()


# In[16]:


data_cleaned.describe()


# In[17]:


# Define function to categorize age
def categorize_age(age):
    if age < 25:
        return '18-24'
    elif 25 <= age < 35:
        return '25-34'
    elif 35 <= age < 45:
        return '35-44'
    elif 45 <= age < 55:
        return '45-54'
    elif 55 <= age < 65:
        return '55-64'
    elif 65 <= age < 80:
        return '65-79'
    else:
        return '80+'
    
# Define function to categorize amount
def categorize_amount(amount):
    if amount < 3000:
        return 'Very Low'
    elif 3000 <= amount < 6000:
        return 'Low'
    elif 6000 <= amount < 9000:
        return 'Medium'
    elif 9000 <= amount < 12000:
        return 'High'
    else:
        return 'Very High'

def categorize_emp_length(emp_length):
    if emp_length == 0:
        return 'No Experience'
    elif 0 < emp_length <= 2:
        return 'Junior'
    elif 2 < emp_length <= 5:
        return 'Mid-level'
    elif 5 < emp_length <= 10:
        return 'Senior'
    else:
        return 'Veteran'
    
def categorize_cred_length(cred_length):
    if cred_length < 2:
        return 'New'
    elif 2 <= cred_length < 5:
        return 'Short'
    elif 5 <= cred_length < 10:
        return 'Moderate'
    elif 10 <= cred_length < 20:
        return 'Long'
    else:
        return 'Very Long'
    
def categorize_rate(rate):
    if rate < 5:
        return 'Low Rate'
    elif 5 <= rate < 10:
        return 'Medium Rate'
    elif 10 <= rate < 20:
        return 'High Rate'
    else:
        return 'Very High Rate'
    
def categorize_percent_income(percent_income):
    if percent_income < 0.09:
        return 'Low Percent Income'
    elif 0.09 <= percent_income < 0.14:
        return 'Moderate Percent Income'
    elif 0.14 <= percent_income < 0.22:
        return 'High Percent Income'
    else:
        return 'Very High Percent Income'

# Apply the function to create a new column
data_cleaned['Percent_income_category'] = data_cleaned['Percent_income'].apply(categorize_percent_income)

# Apply the function to create a new column
data_cleaned['Rate_category'] = data_cleaned['Rate'].apply(categorize_rate)

# Apply the function to create a new column
data_cleaned['Cred_length_category'] = data_cleaned['Cred_length'].apply(categorize_cred_length)
    
data_cleaned['Age_category'] = data_cleaned['Age'].apply(categorize_age)
data_cleaned['Amount_category'] = data_cleaned['Amount'].apply(categorize_amount)
data_cleaned['Emp_length_category'] = data_cleaned['Emp_length'].apply(categorize_emp_length)


# In[18]:


data_cleaned


# In[19]:


categorical_columns = data_cleaned.select_dtypes(include=['object']).columns
# Select categorical columns
categorical_columns = data_cleaned.select_dtypes(include=['object']).columns

# Loop through each categorical column, display unique values count, and plot countplots
for col in categorical_columns:
    unique_values_count = data_cleaned[col].nunique()
    print(f"Column: {col}, Unique Values: {unique_values_count}")
    print(data_cleaned[col].value_counts(), data_cleaned[col].value_counts(normalize=True))
    
    # Sort the categories in order for better visualization
    sorted_categories = data_cleaned[col].value_counts().index
    
    # Plot count plot
    plt.figure(figsize=(5,5))
    sns.countplot(x=col, data=data_cleaned, order=sorted_categories)
    plt.title(f'Count Plot of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)  # Rotate labels if necessary
    plt.tight_layout()
    plt.show()
    plt.show()


# In[20]:


data_cleaned.columns


# In[21]:


# List of selected categorical columns to iterate through
selected_columns = ['Home', 'Intent', 'Status','Rate_category','Age_category', 'Amount_category', 'Emp_length_category', 'Cred_length_category', 'Percent_income_category']  # Add other relevant columns here

# Loop through each column and create count plots against 'default'
for col in selected_columns:
    percentages = data_cleaned.groupby('Default')[col].value_counts(normalize=True).unstack() * 100
    print(f"Percentage distribution for {col}:\n{percentages}\n")
    
    # Count plot for each selected column against 'default'
    sns.countplot(x=col, hue='Default', data=data_cleaned, palette='Set2')
    
    plt.title(f'Count Plot of {col} vs Default')
    plt.xlabel(col.replace('_', ' ').title())  # Formatting the column name for display
    plt.ylabel('Count')
    plt.xticks(rotation=45)  # Rotate labels if necessary
    plt.tight_layout()
    plt.show()


# In[ ]:





# In[22]:


import pandas as pd
from scipy.stats import chi2_contingency

# List of categorical columns to test against Default
categorical_columns = ['Home', 'Intent', 'Status', 'Rate_category','Age_category', 'Amount_category', 'Emp_length_category', 'Cred_length_category','Percent_income_category']

# Loop through each column and perform Chi-Square test
for col in categorical_columns:
    # Create a contingency table
    contingency_table = pd.crosstab(data_cleaned[col], data_cleaned['Default'])
    
    # Perform Chi-Square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    # Display the hypotheses
    print(f"Chi-Square Test for {col}:")
    print(f"Null Hypothesis (H0): There is no association between {col} and Default.")
    print(f"Alternative Hypothesis (H1): There is an association between {col} and Default.")
    
    # Display the Chi-Square statistic and p-value
    print(f"Chi2 Statistic: {chi2}")
    print(f"p-value: {p}")
    print(f"Degrees of Freedom: {dof}")
    print(f"Expected Frequencies:\n{expected}\n")
    
    # Interpret the p-value
    alpha = 0.05
    if p < alpha:
        print(f"Reject the null hypothesis: There is a significant association between {col} and Default.\n")
    else:
        print(f"Fail to reject the null hypothesis: No significant association between {col} and Default.\n")


# In[23]:


data_cleaned.info()


# In[ ]:





# In[24]:


data_sel_cols =data_cleaned[['Home', 'Status','Amount', 'Emp_length', "Rate",'Percent_income','Default']]


# In[25]:


data_sel_cols


# In[26]:


le_home = LabelEncoder()


# In[27]:


data_sel_cols['Home'] = le_home.fit_transform(data_sel_cols['Home'])
with open('label_encoder_home.pkl', 'wb') as file:
    pickle.dump(le_home, file)


# In[28]:


X = data_sel_cols.drop('Default', axis=1)
y = data_sel_cols['Default']


# In[29]:


X.head()


# In[30]:


y.head()


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[32]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[33]:


#Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)


# In[34]:


models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'Gradient Boosting': GradientBoostingClassifier()
}


# In[35]:


model_accuracy = {}
for model_name, model in models.items():
    # Cross-validation
    scores = cross_val_score(model, X_train_res, y_train_res, cv=5, scoring='accuracy')
    print(f"{model_name} - Cross-Validation Accuracy: {np.mean(scores):.4f}")

    # Fit model and make predictions
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_scaled)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} - Test Accuracy: {accuracy:.4f}")
    
    model_accuracy[model_name] = accuracy
best_model_name = max(model_accuracy, key=model_accuracy.get)
best_model = models[best_model_name]
print(f"\nBest Model: {best_model_name} with Test Accuracy: {model_accuracy[best_model_name]:.4f}")
with open(f'{best_model_name}_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

print(f"\nBest model saved as '{best_model_name}_model.pkl'")

