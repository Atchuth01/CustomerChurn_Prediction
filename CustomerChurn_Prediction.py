import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#read the csv file
df = pd.read_csv('Customer Churn.csv')

# One-Hot Encode categorical variables
df_encoded = pd.get_dummies(df, columns=['country', 'gender'])

# Now, df_encoded has numerical representations for Country and Gender
df = df_encoded

#split the data for input features(X) and target(y)
X = df.drop(['churn'], axis=1)
y = df['churn']

#Split the daa for training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2 , random_state=42)

#model defining and training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#testing and evultting the model
y_pred = model.predict(X_test)
print('Accuracy : ', accuracy_score(y_test, y_pred))
print("Classification Report: \n", classification_report(y_test, y_pred))

#testing on a separate csv file contaning customer information
test_df = pd.read_csv('test_data[churn].csv')

#for categorical value containing columns
test_df_encoded = pd.get_dummies(test_df,columns=['country', 'gender'])

test_df = test_df_encoded

# predicting the customer churn
y_pred = model.predict(test_df)

#getting customer ID's
customer_id = test_df['customer_id'][y_pred == 1]

#Printinf the customer ID's who are going to leave
print("Customers who are going to leave : ")
print(customer_id)
