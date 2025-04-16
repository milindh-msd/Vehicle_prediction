# importing the pandas module for data frame
import pandas as pd

# load the data set into train variable.
train = pd.read_csv('vehicles.csv')

# display top 5 values of data set
train.head()

# Functions to extract features from timestamp
def get_dom(dt):
    return dt.day

def get_weekday(dt):
    return dt.weekday()

def get_hour(dt):
    return dt.hour

def get_year(dt):
    return dt.year

def get_month(dt):
    return dt.month

def get_dayofyear(dt):
    return dt.dayofyear

def get_weekofyear(dt):
    return dt.isocalendar().week  # Updated for newer pandas versions

# Convert 'DateTime' to datetime format and extract features
train['DateTime'] = pd.to_datetime(train['DateTime'])
train['date'] = train['DateTime'].map(get_dom)
train['weekday'] = train['DateTime'].map(get_weekday)
train['hour'] = train['DateTime'].map(get_hour)
train['month'] = train['DateTime'].map(get_month)
train['year'] = train['DateTime'].map(get_year)
train['dayofyear'] = train['DateTime'].map(get_dayofyear)
train['weekofyear'] = train['DateTime'].map(get_weekofyear)

# Remove the original DateTime column
train = train.drop(['DateTime'], axis=1)

# Separate features and target
train1 = train.drop(['Vehicles'], axis=1)
target = train['Vehicles']

print(train1.head())
print(target.head())

# Importing and training the RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

m1 = RandomForestRegressor()
m1.fit(train1, target)

# ✅ Fixed prediction with column names
test_input = pd.DataFrame([[11, 6, 0, 1, 2015, 11, 2]], 
                          columns=['date', 'weekday', 'hour', 'month', 'year', 'dayofyear', 'weekofyear'])

prediction = m1.predict(test_input)
print("Predicted Vehicles:", prediction[0])
