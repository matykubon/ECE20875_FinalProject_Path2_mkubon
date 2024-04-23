import pandas
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

#Process Data:
#extraction of each column from raw data file
dataset_2 = pandas.read_csv('NYC_Bicycle_Counts_2016.csv')
dataset_2['Brooklyn Bridge']      = pandas.to_numeric(dataset_2['Brooklyn Bridge'].replace(',','', regex=True))
dataset_2['Manhattan Bridge']     = pandas.to_numeric(dataset_2['Manhattan Bridge'].replace(',','', regex=True))
dataset_2['Queensboro Bridge']    = pandas.to_numeric(dataset_2['Queensboro Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
dataset_2['Total']  = pandas.to_numeric(dataset_2['Total'].replace(',','', regex=True))

#average temperature recorded on each day in NYC
dataset_2['temp_average'] = (dataset_2['Low Temp'] + dataset_2['High Temp']) / 2

"""
for reference --> Additional cols in .csv: 
dataset_2['Date']
dataset_2['Day']
dataset_2['High Temp']
dataset_2['Low Temp']
dataset_2['Precipitation']
"""




#Function Definitions: 
#creates a set of plots and simple calculations for preliminary analysis of biking data in NYC
def prelim_analysis():

    #preliminary analysis of raw data 
    plt.scatter(dataset_2['Date'], dataset_2['Total'], s=10)
    plt.xlabel('Date')
    plt.ylabel('Measured Bike Traffic')
    plt.title('Bike Traffic in NYC - Total')
    n = 15  #show every 15th date
    plt.xticks(dataset_2['Date'][::n], rotation=45)
    plt.show()


    plt.scatter(dataset_2['Date'], dataset_2['Brooklyn Bridge'] , label='Brooklyn Bridge', s=10)
    plt.scatter(dataset_2['Date'], dataset_2['Manhattan Bridge'] , label='Manhattan Bridge', s=10)
    plt.scatter(dataset_2['Date'], dataset_2['Queensboro Bridge'] , label='Queensboro Bridge', s=10)
    plt.scatter(dataset_2['Date'], dataset_2['Williamsburg Bridge'] , label='Williamsburg Bridge', s=10)
    plt.xlabel('Date')
    plt.ylabel('Measured Bike Traffic')
    plt.title('Bike Traffic in NYC - Four Bridges')
    n = 15 #show every 15th date
    plt.xticks(dataset_2['Date'][::n], rotation=45)
    plt.legend(loc='lower right') 
    plt.show()


    plt.scatter(dataset_2['Date'], dataset_2['Total'], s=10)
    plt.xlabel('Date')
    plt.ylabel('Average Temperature (°F)')
    plt.title('Average Temperature in NYC')
    n = 15  #show every 15th date
    plt.xticks(dataset_2['Date'][::n], rotation=45)
    plt.show()
    
    plt.scatter(dataset_2['Date'], dataset_2['Total'], s=10)
    plt.xlabel('Date')
    plt.ylabel('Raindrop height (inches)')
    plt.title('Precipitation in NYC')
    n = 15  #show every 15th date
    plt.xticks(dataset_2['Date'][::n], rotation=45)
    plt.show()
    
    
    #bar chart for the total count of bikers on different days of the week 
    plt.figure(figsize=(8, 6))
    plt.bar(dataset_2['Day'], dataset_2['Total'], color='skyblue')
    plt.title('NYC Total Biker Count on Days of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Bikers in NYC')
    plt.xticks(rotation=45) 
    plt.grid(axis='y', linestyle='--', alpha=0.7) 
    plt.tight_layout() 
    plt.show()
    
    #sub plots for the 4 bridges
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()
    colors = ['skyblue', 'salmon', 'gold', 'lightgreen']
    bridges = ['Brooklyn Bridge', 'Manhattan Bridge', 'Queensboro Bridge', 'Williamsburg Bridge']

    for i, (bridge, color) in enumerate(zip(bridges, colors)):
        axs[i].bar(dataset_2['Day'], dataset_2[bridge], color=color)
        axs[i].set_title(bridge)
        axs[i].set_xlabel('Day of the Week')
        axs[i].set_ylabel('Number of Bikers')
        axs[i].set_xticklabels(dataset_2['Day'], rotation=45)
        axs[i].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def bridge_traffic_analysis():
    r_WI = dataset_2['Total'].corr(dataset_2['Williamsburg Bridge'])
    r_MA = dataset_2['Total'].corr(dataset_2['Manhattan Bridge'])
    r_QU = dataset_2['Total'].corr(dataset_2['Queensboro Bridge'])
    r_BR = dataset_2['Total'].corr(dataset_2['Brooklyn Bridge'])
    
    mse_WI = mean_squared_error(dataset_2['Total'], dataset_2['Williamsburg Bridge'])
    mse_MA = mean_squared_error(dataset_2['Total'], dataset_2['Manhattan Bridge'])
    mse_QU = mean_squared_error(dataset_2['Total'], dataset_2['Queensboro Bridge'])
    mse_BR = mean_squared_error(dataset_2['Total'], dataset_2['Brooklyn Bridge'])
    
    rmse_WI = numpy.sqrt(mse_WI)
    rmse_MA = numpy.sqrt(mse_MA)
    rmse_QU = numpy.sqrt(mse_QU)
    rmse_BR = numpy.sqrt(mse_BR)
    
    print('Correlation Coefficient Data:')
    
    print('Williamsburg Bridge')
    print('r: ' + str(r_WI))
    print('r^2: ' + str(r_WI ** 2))
    print('MSE: ' + str(mse_WI))
    print('RMSE: ' + str(rmse_WI))
    
    print('\nManhattan Bridge')
    print('r: ' + str(r_MA))
    print('r^2: ' + str(r_MA ** 2))
    print('MSE: ' + str(mse_MA))
    print('RMSE: ' + str(rmse_MA))
    
    print('\nQueensboro Bridge')
    print('r: ' + str(r_QU))
    print('r^2: ' + str(r_QU ** 2))
    print('MSE: ' + str(mse_QU))
    print('RMSE: ' + str(rmse_QU))
    
    print('\nBrooklyn Bridge')
    print('r: ' + str(r_BR))
    print('r^2: ' + str(r_BR ** 2))
    print('MSE: ' + str(mse_BR))
    print('RMSE: ' + str(rmse_BR))

def weather_analysis():
    
    #--------------------------------------------------
    #train 1st prediction model 
    X = dataset_2[['temp_average']]
    y = dataset_2['Total']
    
    #split dataset 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    #--------------------------------------------------
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    r_coefficient = numpy.sqrt(r_squared)
    print('\nMSE:', mse)
    print('RMSE:', numpy.sqrt(mse))
    print('r:', r_coefficient)
    print('r^2:', r_squared)
    slope = model.coef_[0]
    intercept = model.intercept_
    print('Model: y = ' + str(slope) + 'x + ' + str(intercept))
    
    #plot the linear model and the raw data side by side
    plt.scatter(X_test, y_test, color='blue', label='raw data')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='linear model')
    plt.xlabel('Average Temperature (°F)')
    plt.ylabel('Number of Bikers')
    plt.title('NYC Temperature - Linear Regression Model')
    plt.legend()
    plt.show()

    #--------------------------------------------------
    #train 2nd percipitation model 
    X = dataset_2[['Precipitation']]
    y = dataset_2['Total']
    
    #split dataset 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #fit the model 
    model = LinearRegression()
    model.fit(X_train, y_train)
    #--------------------------------------------------
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    r_coefficient = numpy.sqrt(r_squared)
    print('\nMSE: ', mse)
    print('RMSE:', numpy.sqrt(mse))
    print('r:', r_coefficient)
    print('r^2:', r_squared)
    slope = model.coef_[0]
    intercept = model.intercept_
    print('Model: y = ' + str(slope) + 'x + ' + str(intercept))
    
    #plot the linear model and the raw data side by side
    plt.scatter(X_test, y_test, color='blue', label='raw data')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='linear model')
    plt.xlabel('Precipitation (inches)')
    plt.ylabel('Number of Bikers')
    plt.title('NYC Precipitation - Linear Regression Model')
    plt.legend()
    plt.show()

    
    #--------------------------------------------------
    #train 3rd percipitation model 
    X = dataset_2[['temp_average', 'Precipitation']]
    y = dataset_2['Total']
    
    #split dataset 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #fit the model 
    model = LinearRegression()
    model.fit(X_train, y_train)
    #--------------------------------------------------
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    r_coefficient = numpy.sqrt(r_squared)
    print('\nMSE:', mse)
    print('RMSE:', numpy.sqrt(mse))
    print('r:', r_coefficient)
    print('r^2:', r_squared)
    coefficients = model.coef_
    intercept = model.intercept_
    print('Model: y = {:.4f} * temp_average + {:.4f} * Precipitation + {:.4f}'.format(coefficients[0], coefficients[1], intercept))
    
def daily_analysis():
    #KNN model
    X = dataset_2[['Total']]        #feature
    y = dataset_2['Day']            #target
    
    #encode into numerical format
    #rn, the y data is strings - Monday, Sunday ... and that can't be used to compute MSE, RMSE ...
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    #cross-validation for each k
    #5-fold cross val. being used (cv)
    k_values = list(range(1, 100))
    cv_scores = []
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(model, X_train, y_train, cv=8, scoring='accuracy')  
        cv_scores.append(numpy.mean(scores))


    optimal_k = k_values[numpy.argmax(cv_scores)]
    print('Optimal k value:', optimal_k)
    
    # Plot cross-validation results
    plt.plot(k_values, cv_scores, marker='o', linestyle='-')
    plt.title('8-fold Cross-Validation results (range from 1-100)')
    plt.xlabel('Number of K Neighbors')
    plt.ylabel('Mean Accuracy')
    plt.show()

    #use optimal k-value
    model = KNeighborsClassifier(n_neighbors=optimal_k)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    predictions_decoded = label_encoder.inverse_transform(predictions)
    
    accuracy = accuracy_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = numpy.sqrt(mse)
    
    #results
    print('Model:', model)
    print('Accuracy Score:', accuracy)
    print('MSE:', mse)
    print('RMSE:', rmse)


#uncomment these functions to run different parts of the analysis <---
#prelim_analysis()
#bridge_traffic_analysis()
#weather_analysis()
daily_analysis()