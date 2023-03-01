import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy.stats import jarque_bera

df: DataFrame = pd.read_excel(r'DATA_Project_1.xlsx')
df['WEEKDAY'] = [i.day_of_week for i in df['DATE']]
assets_only = df[df.columns.difference(['DATE', 'WEEKDAY'])]
assets_byweek = df.loc[df['WEEKDAY'] == 4]
assets_only_byweek=assets_byweek[assets_byweek.columns.difference(['DATE', 'WEEKDAY'])]


# Compute daily and weekly simple returns, moving weekly or fridays only
def assets_returns(assets, period, compounded=True):
    if not compounded:
        asset_return = (assets/assets.shift(periods=period)) - 1
    elif compounded:
        asset_return = np.log(assets/assets.shift(periods=period))

    asset_return.drop(index=asset_return.index[0:abs(period)], axis=0, inplace=True)
    asset_return['DATE'] = df['DATE']
    asset_return.set_index('DATE', inplace=True)
    return asset_return

# def assets_returns_weekly(assets, period, compounded=True):


daily_simple = assets_returns(assets_only, 1, False)
weekly_simple = assets_returns(assets_only_byweek, 1, False)
daily_compounded = assets_returns(assets_only, 1, True)
weekly_compounded = assets_returns(assets_only_byweek, 1, True)


daily_diff = daily_simple - daily_compounded
weekly_diff = weekly_simple - weekly_compounded


# Plot here. Maybe with a plot showing the differences between simple & compounded too
# daily_compounded.plot()
# plt.show()
# daily_simple.plot()
# plt.show()
daily_diff.plot()
plt.show()


# Descriptive Statistics
print('Daily Compounded Returns:')
print(daily_compounded.describe())
print('Weekly Compounded Returns:')
print(weekly_compounded.describe())

#2A
# Extract S&P 500 daily returns
sp500_DCR = daily_compounded.iloc[:, 4]
# Get the largest and smallest Daily Compounded returns
largest_DCR = sp500_DCR.nlargest(5)
smallest_DCR = sp500_DCR.nsmallest(5)



#2C

#Daily Compounded Return

# Convert the dataframe into a numpy array
DCR = daily_compounded.to_numpy()

# Calculate the sample skewness and kurtosis of the data
skewness = np.mean((DCR - np.mean(DCR))**3) / np.mean((DCR - np.mean(DCR))**2)**(3/2)
kurtosis = np.mean((DCR - np.mean(DCR))**4) / np.mean((DCR - np.mean(DCR))**2)**2 - 3

# Perform the Jarque-Bera test
jbtest = jarque_bera(DCR)

# Extract the test statistic and p-value from the test result
test_statistic = jbtest[0]
p_value = jbtest[1]

# Print the sample skewness, kurtosis, test statistic and p-value
print('Sample skewness: {:.4f}'.format(skewness))
print('Sample kurtosis: {:.4f}'.format(kurtosis))
print('Jarque-Bera test statistic: {:.4f}'.format(test_statistic))
print('p-value: {:.4f}'.format(p_value))

# Check if the null hypothesis is rejected or not
alpha = 0.05
if p_value > alpha:
    print('The null hypothesis is not rejected at the {:.0%} significance level.'.format(alpha))
else:
    print('The null hypothesis is rejected at the {:.0%} significance level.'.format(alpha))
    
 #Weekly Compounded Return

# Convert the dataframe into a numpy array
WCR = weekly_compounded.to_numpy()

# Calculate the sample skewness and kurtosis of the data
skewness = np.mean((WCR - np.mean(WCR))**3) / np.mean((WCR - np.mean(WCR))**2)**(3/2)
kurtosis = np.mean((WCR - np.mean(WCR))**4) / np.mean((WCR - np.mean(WCR))**2)**2 - 3

# Perform the Jarque-Bera test
jbtest = jarque_bera(WCR)

# Extract the test statistic and p-value from the test result
test_statistic = jbtest[0]
p_value = jbtest[1]

# Print the sample skewness, kurtosis, test statistic and p-value
print('Sample skewness: {:.4f}'.format(skewness))
print('Sample kurtosis: {:.4f}'.format(kurtosis))
print('Jarque-Bera test statistic: {:.4f}'.format(test_statistic))
print('p-value: {:.4f}'.format(p_value))

# Check if the null hypothesis is rejected or not
alpha = 0.05
if p_value > alpha:
    print('The null hypothesis is not rejected at the {:.0%} significance level.'.format(alpha))
else:
    print('The null hypothesis is rejected at the {:.0%} significance level.'.format(alpha))









