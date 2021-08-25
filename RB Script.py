import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/Sam/Desktop/FFBData/fantasy-full2020.csv')
df2 = pd.read_csv('/Users/Sam/Desktop/FFBData/fantasy-full2019.csv')
df3 = pd.read_csv('/Users/Sam/Desktop/FFBData/fantasy-full2018.csv')

df['Year'] = 2020
df2['Year'] = 2019
df3['Year'] = 2018

df['Catch Percentage'] = df.Receptions/df.Targets
df.fillna(0, inplace=True)
df = df[df['Catch Percentage'] != 0]

df2['Catch Percentage'] = df2.Receptions/df2.Targets
df2.fillna(0, inplace=True)
df2 = df2[df2['Catch Percentage'] != 0]

df3['Catch Percentage'] = df3.Receptions/df3.Targets
df3.fillna(0, inplace=True)
df3 = df3[df3['Catch Percentage'] != 0]

df['Touches'] = df['Rushing Attempts']+df.Receptions
df2['Touches'] = df2['Rushing Attempts']+df2.Receptions
df3['Touches'] = df3['Rushing Attempts']+df3.Receptions

#scaler = StandardScaler()
#df2[['Targets', 'Y/R', 'Catch Percentage', 'Rushing Attempts', 'Rushing Y/A']] = scaler.fit_transform([['Targets', 'Y/R', 'Catch Percentage', 'Rushing Attempts', 'Rushing Y/A']])
#df[['Targets', 'Y/R', 'Catch Percentage', 'Rushing Attempts', 'Rushing Y/A']] = scaler.fit_transform([['Targets', 'Y/R', 'Catch Percentage', 'Rushing Attempts', 'Rushing Y/A']])

#df2.drop('Name', 'Team', 'Pos', 'ID', axis=1)
#df.drop('Name', 'Team', 'Pos', 'ID', axis=1)

#df2 = df2.astype('float')
#df = df.astype('float')

#print(df2.dtypes)
#print(df.dtypes)
#RB Analysis - Linear Regression
df = df[df['Pos'] == 'RB']
df2 = df2[df2['Pos'] == 'RB']
df3 = df3[df3['Pos'] == 'RB']
#2018 to 2020 Predictions
X = df3[['Rushing TD', 'Rushing Attempts', 'Targets']]
y = df3['PPR']

X_test = df[['Rushing TD', 'Rushing Attempts', 'Targets']]
y_test = df['PPR']

lm = LinearRegression()
model = lm.fit(X, y)

print('2018 to 2020 prediction', model.coef_)

df['predictions'] = lm.predict(X_test)
print('Prediction score 2018-2020', lm.score(X_test, y_test))

X = df2[['Rushing TD', 'Rushing Attempts', 'Targets']]
y = df2['PPR']

X_test = df[['Rushing TD', 'Rushing Attempts', 'Targets']]
y_test = df['PPR']

lm = LinearRegression()
model = lm.fit(X, y)

print('2019 to 2020 prediction', model.coef_)

df['predictions'] = lm.predict(X_test)
print('Prediction score 2019-2020', lm.score(X_test, y_test))

#Calculate average number of carries over past three years - need to figure out how to get for each player. Probably need to build a loop :(
df = pd.concat([df,df2,df3], keys=['2020', '2019', '2018'])

#aggregate the data needed
aggregation_functions = {'Rushing TD': 'sum', 'Targets': 'sum', 'Rushing Attempts': 'sum', 'Games Played': 'sum', 'Age': 'max', 'Games Started': 'sum'}
df_new = df.groupby(df['Name']).aggregate(aggregation_functions)

#Predict 2021 score based on Variables
df_new['nextyearpred'] = (((df_new['Rushing TD']/df_new['Games Played']) * 17) * 7.785) + (((df_new['Targets']/df_new['Games Played']) * 17) * 1.6875)+ (((df_new['Rushing Attempts']/df_new['Games Played']) * 17) * 0.3785)

#update ages for 2021 season
df_new['Age'] = df_new['Age'] + 1

#create age factor for best performing seasons
agefactor = [(24,1.1), (25,1.1), (26,1.0), (23,1.0), (27,0.9), (22,0.9), (28,.85), (29, .8), (30,.8), (21,.8), (31,.7), (32,.7), (33,0), (34, 0), (35, 0), (36, 0), (37, 0)]
df_new['AgeFactor'] = df_new['Age'].map(dict(agefactor))

#create starter factor for boosting output of new role
df_new['StartRatio'] = df_new['Games Started']/df_new['Games Played']

df_new['nextyearpredage'] = df_new['nextyearpred']*df_new['AgeFactor']

#print(df[['Name', 'Rushing Attempts']].groupby('Name').mean())

#print('Prediction to PPR correlation', df.predictions.corr(df.PPR))
#print('Yearly RushTD Corr', df['Rushing TD'].corr(df3['Rushing TD']))
#print('Total Rushes to TD correlation:', df['Rushing Attempts'].corr(df['Rushing TD']))
#print('Total Rushes to TD correlation:', df2['Rushing Attempts'].corr(df2['Rushing TD']))
#print('Total Rushes to TD correlation:', df3['Rushing Attempts'].corr(df3['Rushing TD']))

#df = df[df['PPR'] > 100]
#plt.scatter(df['Rushing TD'], df['Rushing Attempts'])
#plt.show()
#plt.close()

#df2 = df2[df2['PPR'] > 100]
#plt.scatter(df2['Rushing TD'], df2['Rushing Attempts'])
#plt.show()
#plt.close()

#df3 = df3[df3['PPR'] > 100]
#plt.scatter(df3['Rushing TD'], df3['Rushing Attempts'])
#plt.show()
#plt.close()

#print('Descriptives of annual Rushing TD for >100 PPR 2020', df['Rushing TD'].describe())
#print('Descriptives of annual Rushing TD for >100 PPR 2019', df2['Rushing TD'].describe())
#print('Descriptives of annual Rushing TD for >100 PPR 2018', df3['Rushing TD'].describe())

#df['Carries Per Game'] = df['Rushing Attempts']/df['Games Played']
#df2['Carries Per Game'] = df2['Rushing Attempts']/df2['Games Played']

#print(df['Carries Per Game'])
#print(df2['Carries Per Game'])
#Calculate the mean number of rushing attempts for top 36 rushers


#RB Analysis - Lasso Regression

#reg = Lasso(alpha=0.5)
#reg.fit(X, y)

#print('Lasso Regression: R^2 score on training set', reg.score(X,y)*100)
#print('Lasso Regression: R^2 score on test set', reg.score (X_test, y_test)*100)

df_new.to_csv(r'/Users/Sam/Desktop/y.csv')
