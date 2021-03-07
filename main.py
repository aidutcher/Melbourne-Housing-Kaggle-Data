import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)

# Create the dataframe
df = pd.read_csv("C:/Users/aidut/Documents/Kaggle/melb_data.csv")
# Format the float numbers to 2 decimals
pd.set_option('display.float_format', '{:.2f}'.format)

# Total rows, columns
print('Total rows and columns: \n')
print(df.shape)
# (13580, 21)

# Checking out the data
print('\nDescription of the data: \n')
print(df.describe())
#               Rooms         Price  ...    Longtitude  Propertycount
# count  13580.000000  1.358000e+04  ...  13580.000000   13580.000000
# mean       2.937997  1.075684e+06  ...    144.995216    7454.417378
# std        0.955748  6.393107e+05  ...      0.103916    4378.581772
# min        1.000000  8.500000e+04  ...    144.431810     249.000000
# 25%        2.000000  6.500000e+05  ...    144.929600    4380.000000
# 50%        3.000000  9.030000e+05  ...    145.000100    6555.000000
# 75%        3.000000  1.330000e+06  ...    145.058305   10331.000000
# max       10.000000  9.000000e+06  ...    145.526350   21650.000000
#
# [8 rows x 13 columns]

# First five rows
print('\nFirst five rows: \n')
print(df.head())
#        Suburb           Address  ...             Regionname Propertycount
# 0  Abbotsford      85 Turner St  ...  Northern Metropolitan        4019.0
# 1  Abbotsford   25 Bloomburg St  ...  Northern Metropolitan        4019.0
# 2  Abbotsford      5 Charles St  ...  Northern Metropolitan        4019.0
# 3  Abbotsford  40 Federation La  ...  Northern Metropolitan        4019.0
# 4  Abbotsford       55a Park St  ...  Northern Metropolitan        4019.0
# [5 rows x 21 columns]

# Last five rows
print('\nLast five rows: \n')
print(df.tail())
#               Suburb        Address  ...                  Regionname Propertycount
# 13575  Wheelers Hill   12 Strada Cr  ...  South-Eastern Metropolitan        7392.0
# 13576   Williamstown  77 Merrett Dr  ...        Western Metropolitan        6380.0
# 13577   Williamstown    83 Power St  ...        Western Metropolitan        6380.0
# 13578   Williamstown   96 Verdon St  ...        Western Metropolitan        6380.0
# 13579     Yarraville     6 Agnes St  ...        Western Metropolitan        6543.0
#
# [5 rows x 21 columns]

# Data types
print('\nData types: \n')
print(df.dtypes)
# Suburb            object
# Address           object
# Rooms              int64
# Type              object
# Price            float64
# Method            object
# SellerG           object
# Date              object
# Distance         float64
# Postcode         float64
# Bedroom2         float64
# Bathroom         float64
# Car              float64
# Landsize         float64
# BuildingArea     float64
# YearBuilt        float64
# CouncilArea       object
# Lattitude        float64
# Longtitude       float64
# Regionname        object
# Propertycount    float64
# dtype: object

# Number of duplicate rows (zero, in this case)
duplicate_rows_df = df[df.duplicated()]
print('\nNumber of duplicate rows: ' + str(duplicate_rows_df.shape) + '\n')
# Number of duplicate rows: (0, 21)

# Find null values
# Lot of nulls in BuildingArea, YearBuilt, CouncilArea
print('\nNumber of rows with null values: \n')
print(df.isnull().sum())
# Suburb              0
# Address             0
# Rooms               0
# Type                0
# Price               0
# Method              0
# SellerG             0
# Date                0
# Distance            0
# Postcode            0
# Bedroom2            0
# Bathroom            0
# Car                62
# Landsize            0
# BuildingArea     6450
# YearBuilt        5375
# CouncilArea      1369
# Lattitude           0
# Longtitude          0
# Regionname          0
# Propertycount       0
# dtype: int64

# Drop rows with null values
# Not always recommended
# df = df.dropna()

# Impute values for null
from sklearn.impute import SimpleImputer
med_imputer = SimpleImputer(missing_values= np.NaN, strategy='median')
mean_imputer = SimpleImputer(missing_values= np.NaN, strategy='mean')
mode_imputer = SimpleImputer(missing_values= np.NaN, strategy='most_frequent')

df.BuildingArea = mean_imputer.fit_transform(df['BuildingArea'].values.reshape(13580,1))
df.YearBuilt = med_imputer.fit_transform(df['YearBuilt'].values.reshape(13580,1))
df.Car = mean_imputer.fit_transform(df['Car'].values.reshape(13580,1))
df.CouncilArea = mode_imputer.fit_transform(df['CouncilArea'].values.reshape(13580,1))

print('\nAfter imputing nulls: \n')
print(df.isnull().sum())
# Suburb           0
# Address          0
# Rooms            0
# Type             0
# Price            0
# Method           0
# SellerG          0
# Date             0
# Distance         0
# Postcode         0
# Bedroom2         0
# Bathroom         0
# Car              0
# Landsize         0
# BuildingArea     0
# YearBuilt        0
# CouncilArea      0
# Lattitude        0
# Longtitude       0
# Regionname       0
# Propertycount    0
# dtype: int64

# Plot variables to identify outliers
# Price, BuildingArea, YearBuilt

# We see a few outliers in Price
# sns.boxplot(x = df['Price'])
# plt.show()
#
# # Also a few in BuildingArea
# sns.boxplot(x = df['BuildingArea'])
# plt.show()
#
# # One in YearBuilt - 1200???
# sns.boxplot(x = df['YearBuilt'])
# plt.show()

# Determine inter-quartile range to identify outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print('\nInter-quartile ranges: \n')
print(IQR)

# Inter-quartile ranges:
#
# Rooms                1.00
# Price           680000.00
# Distance             6.90
# Postcode           104.00
# Bedroom2             1.00
# Bathroom             1.00
# Car                  1.00
# Landsize           474.00
# BuildingArea        29.97
# YearBuilt           15.00
# Lattitude            0.10
# Longtitude           0.13
# Propertycount     5951.00
# dtype: float64

# Remove outliers based on IQR
df_out = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
# Show plots to confirm

sns.boxplot(x = df_out['Price'])
plt.show()

sns.boxplot(x = df_out['BuildingArea'])
plt.show()

sns.boxplot(x = df_out['YearBuilt'])
plt.show()

