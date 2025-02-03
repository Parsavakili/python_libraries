import pandas as pd
print(pd.__version__)  # Check installed version

# Output
# 1.5.0 (or any installed version)

# Creating a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [24, 27, 22, 32],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
}

df = pd.DataFrame(data)

# Display the DataFrame
print("Original DataFrame:")
print(df)

# Output
#    Name  Age         City
# 0  Alice   24     New York
# 1    Bob   27  Los Angeles
# 2 Charlie   22      Chicago
# 3  David   32      Houston

# 1. head() - Returns the first n rows (default is 5)
print("\nFirst 2 rows:")
print(df.head(2))

# Output
#     Name  Age         City
# 0  Alice   24     New York
# 1    Bob   27  Los Angeles

# 2. tail() - Returns the last n rows (default is 5)
print("\nLast 2 rows:")
print(df.tail(2))

# Output
#       Name  Age      City
# 2  Charlie   22   Chicago
# 3    David   32   Houston

# 3. info() - Provides a summary of the DataFrame
print("\nDataFrame Info:")
df.info()

# Output
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 4 entries, 0 to 3
# Data columns (total 3 columns):
#  #   Column  Non-Null Count  Dtype 
# ---  ------  --------------  ----- 
#  0   Name    4 non-null      object
#  1   Age     4 non-null      int64 
#  2   City    4 non-null      object
# dtypes: int64(1), object(2)
# memory usage: 128.0+ bytes

# 4. describe() - Generates descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Output
#         Age
# count   4.0
# mean   26.25
# std     4.56
# min    22.0
# 25%    23.0
# 50%    25.5
# 75%    29.5
# max    32.0

# 5. shape - Returns the dimensions of the DataFrame
print("\nDataFrame Shape:")
print(df.shape)

# Output
# (4, 3)

# 6. columns - Returns the column labels
print("\nDataFrame Columns:")
print(df.columns)

# Output
# Index(['Name', 'Age', 'City'], dtype='object')

# 7. index - Returns the index (row labels)
print("\nDataFrame Index:")
print(df.index)

# Output
# RangeIndex(start=0, stop=4, step=1)

# 8. dtypes - Returns the data types of each column
print("\nData Types:")
print(df.dtypes)

# Output
# Name     object
# Age       int64
# City     object
# dtype: object

# 9. loc[] - Access a group of rows and columns by labels
print("\nAccessing rows using loc:")
print(df.loc[1:2])

# Output
#       Name  Age         City
# 1    Bob   27  Los Angeles
# 2 Charlie   22      Chicago

# 10. iloc[] - Access a group of rows and columns by integer position
print("\nAccessing rows using iloc:")
print(df.iloc[1:3])

# Output
#       Name  Age         City
# 1    Bob   27  Los Angeles
# 2 Charlie   22      Chicago

# 11. at[] - Access a single value for a row/column label pair
print("\nAccessing a single value using at:")
print(df.at[2, 'Name'])

# Output
# Charlie

# 12. iat[] - Access a single value for a row/column pair by integer position
print("\nAccessing a single value using iat:")
print(df.iat[2, 0])

# Output
# Charlie

# 13. set_index() - Set the DataFrame index using existing columns
df.set_index('Name', inplace=True)
print("\nDataFrame after setting 'Name' as index:")
print(df)

# Output
#         Age         City
# Name                    
# Alice   24     New York
# Bob     27  Los Angeles
# Charlie 22      Chicago
# David   32      Houston

# 14. reset_index() - Reset the index of the DataFrame
df.reset_index(inplace=True)
print("\nDataFrame after resetting index:")
print(df)

# Output
#       Name  Age         City
# 0    Alice   24     New York
# 1      Bob   27  Los Angeles
# 2  Charlie   22      Chicago
# 3    David   32      Houston

# 15. sort_values() - Sort by the values along either axis
print("\nDataFrame sorted by 'Age':")
print(df.sort_values(by='Age'))

# Output
#       Name  Age         City
# 2  Charlie   22      Chicago
# 0    Alice   24     New York
# 1      Bob   27  Los Angeles
# 3    David   32      Houston

# 16. drop() - Drop specified labels from rows or columns
df_dropped = df.drop(columns=['City'])
print("\nDataFrame after dropping 'City' column:")
print(df_dropped)

# Output
#       Name  Age
# 0    Alice   24
# 1      Bob   27
# 2  Charlie   22
# 3    David   32

# 17. rename() - Rename columns or index
df_renamed = df.rename(columns={'Name': 'Full Name'})
print("\nDataFrame after renaming 'Name' to 'Full Name':")
print(df_renamed)

# Output
#    Full Name  Age
# 0      Alice   24
# 1        Bob   27
# 2    Charlie   22
# 3      David   32

# 18. fillna() - Fill NA/NaN values using the specified method
df_with_nan = pd.DataFrame({'A': [1, 2, None], 'B': [5, None, None]})
df_filled = df_with_nan.fillna(0)
print("\nDataFrame after filling NaN values with 0:")
print(df_filled)

# Output
#      A    B
# 0  1.0  5.0
# 1  2.0  0.0
# 2  0.0  0.0

# 19. dropna() - Remove missing values
df_dropped_na = df_with_nan.dropna()
print("\nDataFrame after dropping rows with NaN values:")
print(df_dropped_na)

# Output
#      A    B
# 0  1.0  5.0
# 1  2.0  NaN

# 20. groupby() - Group data using a mapper or by a series of columns
grouped = df.groupby('City').mean(numeric_only=True)
print("\nGrouped DataFrame by 'City' (mean of numeric columns):")
print(grouped)

# Output
# Empty DataFrame
# Columns: [Age]
# Index: []

# 21. merge() - Merge DataFrame objects
df2 = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Eve'],
    'Salary': [70000, 80000, 90000]
})

merged_df = pd.merge(df, df2, on='Name', how='inner')
print("\nMerged DataFrame:")
print(merged_df)

# Output
#    Name  Age         City  Salary
# 0  Alice   24     New York   70000
# 1    Bob   27  Los Angeles   80000

# 22. concat() - Concatenate pandas objects along a particular axis
concatenated_df = pd.concat([df, df2], axis=1)
print("\nConcatenated DataFrame:")
print(concatenated_df)

# Output
#       Name  Age         City   Name  Salary
# 0    Alice   24     New York  Alice   70000
# 1      Bob   27  Los Angeles    Bob   80000
# 2  Charlie   22      Chicago  NaN    NaN
# 3    David   32      Houston  NaN    NaN

# 23. apply() - Apply a function along an axis of the DataFrame
df['Age_plus_5'] = df['Age'].apply(lambda x: x + 5)
print("\nDataFrame after applying a function to 'Age':")
print(df)

# Output
#       Name  Age         City  Age_plus_5
# 0    Alice   24     New York          29
# 1      Bob   27  Los Angeles          32
# 2  Charlie   22      Chicago          27
# 3    David   32      Houston          37

# 24. value_counts() - Return a Series containing counts of unique values
print("\nValue counts for 'City':")
print(df['City'].value_counts())

# Output
# New York       1
# Los Angeles    1
# Chicago        1
# Houston        1
# Name: City, dtype: int64

# 25. pivot_table() - Create a spreadsheet-style pivot table
pivot_table = df.pivot_table(index='City', values='Age', aggfunc='mean')
print("\nPivot Table:")
print(pivot_table)

# Output
#                Age
# City              
# Chicago       22.0
# Houston       32.0
# Los Angeles   27.0
# New York      24.0

# 26. to_csv() - Write DataFrame to a CSV file
df.to_csv('output.csv', index=False)
print("\nDataFrame written to 'output.csv'")

# Output
# DataFrame written to 'output.csv'

# 27. read_csv() - Read a CSV file into a DataFrame
df_from_csv = pd.read_csv('output.csv')
print("\nDataFrame read from 'output.csv':")
print(df_from_csv)

# Output
#       Name  Age         City  Age_plus_5
# 0    Alice   24     New York          29
# 1      Bob   27  Los Angeles          32
# 2  Charlie   22      Chicago          27
# 3    David   32      Houston          37

# 28. to_excel() - Write DataFrame to an Excel file
df.to_excel('output.xlsx', index=False)
print("\nDataFrame written to 'output.xlsx'")

# Output
# DataFrame written to 'output.xlsx'

# 29. read_excel() - Read an Excel file into a DataFrame
df_from_excel = pd.read_excel('output.xlsx')
print("\nDataFrame read from 'output.xlsx':")
print(df_from_excel)

# Output
#       Name  Age         City  Age_plus_5
# 0    Alice   24     New York          29
# 1      Bob   27  Los Angeles          32
# 2  Charlie   22      Chicago          27
# 3    David   32      Houston          37

# 30. unique() - Return unique values in a Series
print("\nUnique values in 'City':")
print(df['City'].unique())

# Output
# ['New York' 'Los Angeles' 'Chicago' 'Houston']

# 31. nunique() - Return the number of unique values in a Series
print("\nNumber of unique values in 'City':")
print(df['City'].nunique())

# Output
# 4

# 32. isna() - Detect missing values
print("\nDetecting missing values:")
print(df_with_nan.isna())

# Output
#        A      B
# 0  False  False
# 1  False   True
# 2   True   True

# 33. notna() - Detect non-missing values
print("\nDetecting non-missing values:")
print(df_with_nan.notna())

# Output
#        A     B
# 0   True   True
# 1   True  False
# 2  False  False

# 34. query() - Query the columns of a DataFrame with a boolean expression
print("\nDataFrame after querying 'Age > 25':")
print(df.query('Age > 25'))

# Output
#     Name  Age         City  Age_plus_5
# 1   Bob   27  Los Angeles          32
# 3  David   32      Houston          37

# 35. sample() - Return a random sample of items from an axis
print("\nRandom sample of 2 rows:")
print(df.sample(2))

# Output
#       Name  Age         City  Age_plus_5
# 1      Bob   27  Los Angeles          32
# 2  Charlie   22      Chicago          27

# 36. corr() - Compute pairwise correlation of columns
print("\nCorrelation matrix (only numeric columns):")
print(df[['Age', 'Age_plus_5']].corr())

# Output
#               Age  Age_plus_5
# Age           1.0         1.0
# Age_plus_5    1.0         1.0

# 37. cov() - Compute pairwise covariance of columns
print("\nCovariance matrix (only numeric columns):")
print(df[['Age', 'Age_plus_5']].cov())

# Output
#                Age  Age_plus_5
# Age           18.25        18.25
# Age_plus_5    18.25        18.25

# 38. duplicated() - Return boolean Series denoting duplicate rows
print("\nDuplicated rows:")
print(df.duplicated())

# Output
# 0    False
# 1    False
# 2    False
# 3    False
# dtype: bool

# 39. drop_duplicates() - Remove duplicate rows
df_no_duplicates = df.drop_duplicates()
print("\nDataFrame after dropping duplicates:")
print(df_no_duplicates)

# Output
#       Name  Age         City  Age_plus_5
# 0    Alice   24     New York          29
# 1      Bob   27  Los Angeles          32
# 2  Charlie   22      Chicago          27
# 3    David   32      Houston          37

# 40. replace() - Replace values given in to_replace with value
df_replaced = df.replace({'City': {'New York': 'NYC'}})
print("\nDataFrame after replacing 'New York' with 'NYC':")
print(df_replaced)

# Output
#       Name  Age         City  Age_plus_5
# 0    Alice   24          NYC          29
# 1      Bob   27  Los Angeles          32
# 2  Charlie   22      Chicago          27
# 3    David   32      Houston          37

# 41. astype() - Cast a pandas object to a specified dtype
df['Age'] = df['Age'].astype('float')
print("\nDataFrame after changing 'Age' to float:")
print(df)

# Output
#       Name   Age         City  Age_plus_5
# 0    Alice  24.0     New York          29
# 1      Bob  27.0  Los Angeles          32
# 2  Charlie  22.0      Chicago          27
# 3    David  32.0      Houston          37

# 42. to_numpy() - Convert the DataFrame to a NumPy array
numpy_array = df.to_numpy()
print("\nDataFrame converted to NumPy array:")
print(numpy_array)

# Output
# [['Alice' 24.0 'New York' 29]
#  ['Bob' 27.0 'Los Angeles' 32]
#  ['Charlie' 22.0 'Chicago' 27]
#  ['David' 32.0 'Houston' 37]]

# 43. to_dict() - Convert the DataFrame to a dictionary
dict_data = df.to_dict()
print("\nDataFrame converted to dictionary:")
print(dict_data)

# Output
# {'Name': {'0': 'Alice', '1': 'Bob', '2': 'Charlie', '3': 'David'},
#  'Age': {'0': 24.0, '1': 27.0, '2': 22.0, '3': 32.0},
#  'City': {'0': 'New York', '1': 'Los Angeles', '2': 'Chicago', '3': 'Houston'},
#  'Age_plus_5': {'0': 29, '1': 32, '2': 27, '3': 37}}

# 44. to_json() - Convert the DataFrame to a JSON string
json_data = df.to_json()
print("\nDataFrame converted to JSON:")
print(json_data)

# Output
# {"Name":{"0":"Alice","1":"Bob","2":"Charlie","3":"David"},"Age":{"0":24.0,"1":27.0,"2":22.0,"3":32.0},"City":{"0":"New York","1":"Los Angeles","2":"Chicago","3":"Houston"},"Age_plus_5":{"0":29,"1":32,"2":27,"3":37}}

# 45. to_html() - Render a DataFrame as an HTML table
html_data = df.to_html()
print("\nDataFrame converted to HTML:")
print(html_data)

# Output
# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th>Name</th>
#       <th>Age</th>
#       <th>City</th>
#       <th>Age_plus_5</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>0</th>
#       <td>Alice</td>
#       <td>24.0</td>
#       <td>New York</td>
#       <td>29</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>Bob</td>
#       <td>27.0</td>
#       <td>Los Angeles</td>
#       <td>32</td>
#     </tr>
#     <tr>
#       <th>2</th>
#       <td>Charlie</td>
#       <td>22.0</td>
#       <td>Chicago</td>
#       <td>27</td>
#     </tr>
#     <tr>
#       <th>3</th>
#       <td>David</td>
#       <td>32.0</td>
#       <td>Houston</td>
#       <td>37</td>
#     </tr>
#   </tbody>
# </table>
