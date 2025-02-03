import pandas as pd

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

# 1. head() - Returns the first n rows (default is 5)
print("\nFirst 2 rows:")
print(df.head(2))

# 2. tail() - Returns the last n rows (default is 5)
print("\nLast 2 rows:")
print(df.tail(2))

# 3. info() - Provides a summary of the DataFrame
print("\nDataFrame Info:")
print(df.info())

# 4. describe() - Generates descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# 5. shape - Returns the dimensions of the DataFrame
print("\nDataFrame Shape:")
print(df.shape)

# 6. columns - Returns the column labels
print("\nDataFrame Columns:")
print(df.columns)

# 7. index - Returns the index (row labels)
print("\nDataFrame Index:")
print(df.index)

# 8. dtypes - Returns the data types of each column
print("\nData Types:")
print(df.dtypes)

# 9. loc[] - Access a group of rows and columns by labels
print("\nAccessing rows using loc:")
print(df.loc[1:2])

# 10. iloc[] - Access a group of rows and columns by integer position
print("\nAccessing rows using iloc:")
print(df.iloc[1:3])

# 11. at[] - Access a single value for a row/column label pair
print("\nAccessing a single value using at:")
print(df.at[2, 'Name'])

# 12. iat[] - Access a single value for a row/column pair by integer position
print("\nAccessing a single value using iat:")
print(df.iat[2, 0])

# 13. set_index() - Set the DataFrame index using existing columns
df.set_index('Name', inplace=True)
print("\nDataFrame after setting 'Name' as index:")
print(df)

# 14. reset_index() - Reset the index of the DataFrame
df.reset_index(inplace=True)
print("\nDataFrame after resetting index:")
print(df)

# 15. sort_values() - Sort by the values along either axis
print("\nDataFrame sorted by 'Age':")
print(df.sort_values(by='Age'))

# 16. drop() - Drop specified labels from rows or columns
df_dropped = df.drop(columns=['City'])
print("\nDataFrame after dropping 'City' column:")
print(df_dropped)

# 17. rename() - Rename columns or index
df_renamed = df.rename(columns={'Name': 'Full Name'})
print("\nDataFrame after renaming 'Name' to 'Full Name':")
print(df_renamed)

# 18. fillna() - Fill NA/NaN values using the specified method
df_with_nan = pd.DataFrame({'A': [1, 2, None], 'B': [5, None, None]})
df_filled = df_with_nan.fillna(0)
print("\nDataFrame after filling NaN values with 0:")
print(df_filled)

# 19. dropna() - Remove missing values
df_dropped_na = df_with_nan.dropna()
print("\nDataFrame after dropping rows with NaN values:")
print(df_dropped_na)

# 20. groupby() - Group data using a mapper or by a series of columns
grouped = df.groupby('City').mean()
print("\nGrouped DataFrame by 'City' and mean of 'Age':")
print(grouped)

# 21. merge() - Merge DataFrame objects
df2 = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Eve'],
    'Salary': [70000, 80000, 90000]
})

merged_df = pd.merge(df, df2, on='Name', how='inner')
print("\nMerged DataFrame:")
print(merged_df)

# 22. concat() - Concatenate pandas objects along a particular axis
concatenated_df = pd.concat([df, df2], axis=1)
print("\nConcatenated DataFrame:")
print(concatenated_df)

# 23. apply() - Apply a function along an axis of the DataFrame
df['Age_plus_5'] = df['Age'].apply(lambda x: x + 5)
print("\nDataFrame after applying a function to 'Age':")
print(df)

# 24. value_counts() - Return a Series containing counts of unique values
print("\nValue counts for 'City':")
print(df['City'].value_counts())

# 25. pivot_table() - Create a spreadsheet-style pivot table
pivot_table = df.pivot_table(index='City', values='Age', aggfunc='mean')
print("\nPivot Table:")
print(pivot_table)

# 26. to_csv() - Write DataFrame to a CSV file
df.to_csv('output.csv', index=False)
print("\nDataFrame written to 'output.csv'")

# 27. read_csv() - Read a CSV file into a DataFrame
df_from_csv = pd.read_csv('output.csv')
print("\nDataFrame read from 'output.csv':")
print(df_from_csv)

# 28. to_excel() - Write DataFrame to an Excel file
df.to_excel('output.xlsx', index=False)
print("\nDataFrame written to 'output.xlsx'")

# 29. read_excel() - Read an Excel file into a DataFrame
df_from_excel = pd.read_excel('output.xlsx')
print("\nDataFrame read from 'output.xlsx':")
print(df_from_excel)

# 30. unique() - Return unique values in a Series
print("\nUnique values in 'City':")
print(df['City'].unique())

# 31. nunique() - Return the number of unique values in a Series
print("\nNumber of unique values in 'City':")
print(df['City'].nunique())

# 32. isna() - Detect missing values
print("\nDetecting missing values:")
print(df_with_nan.isna())

# 33. notna() - Detect non-missing values
print("\nDetecting non-missing values:")
print(df_with_nan.notna())

# 34. query() - Query the columns of a DataFrame with a boolean expression
print("\nDataFrame after querying 'Age > 25':")
print(df.query('Age > 25'))

# 35. sample() - Return a random sample of items from an axis
print("\nRandom sample of 2 rows:")
print(df.sample(2))

# 36. corr() - Compute pairwise correlation of columns
print("\nCorrelation matrix:")
print(df.corr())

# 37. cov() - Compute pairwise covariance of columns
print("\nCovariance matrix:")
print(df.cov())

# 38. duplicated() - Return boolean Series denoting duplicate rows
print("\nDuplicated rows:")
print(df.duplicated())

# 39. drop_duplicates() - Remove duplicate rows
df_no_duplicates = df.drop_duplicates()
print("\nDataFrame after dropping duplicates:")
print(df_no_duplicates)

# 40. replace() - Replace values given in to_replace with value
df_replaced = df.replace({'City': {'New York': 'NYC'}})
print("\nDataFrame after replacing 'New York' with 'NYC':")
print(df_replaced)

# 41. astype() - Cast a pandas object to a specified dtype
df['Age'] = df['Age'].astype('float')
print("\nDataFrame after changing 'Age' to float:")
print(df)

# 42. to_numpy() - Convert the DataFrame to a NumPy array
numpy_array = df.to_numpy()
print("\nDataFrame converted to NumPy array:")
print(numpy_array)

# 43. to_dict() - Convert the DataFrame to a dictionary
dict_data = df.to_dict()
print("\nDataFrame converted to dictionary:")
print(dict_data)

# 44. to_json() - Convert the DataFrame to a JSON string
json_data = df.to_json()
print("\nDataFrame converted to JSON:")
print(json_data)

# 45. to_html() - Render a DataFrame as an HTML table
html_data = df.to_html()
print("\nDataFrame converted to HTML:")
print(html_data)

# 46. to_string() - Render a DataFrame to a console-friendly tabular output
string_data = df.to_string()
print("\nDataFrame converted to string:")
print(string_data)

# 47. to_clipboard() - Copy the object to the system clipboard
df.to_clipboard()
print("\nDataFrame copied to clipboard.")

# 48. to_latex() - Render a DataFrame to a LaTeX tabular environment
latex_data = df.to_latex()
print("\nDataFrame converted to LaTeX:")
print(latex_data)

# 49. to_markdown() - Print DataFrame in Markdown-friendly format
markdown_data = df.to_markdown()
print("\nDataFrame converted to Markdown:")
print(markdown_data)

# 50. to_records() - Convert DataFrame to a NumPy record array
records = df.to_records()
print("\nDataFrame converted to records:")
print(records)