# import_data.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def process_data(file_path = 'C:\\Users\\axt43242\\Downloads\\irp-workbooks-progress\\Extracted_Prod_Data.xlsx'):
    # Read the Excel file
    prd_data_df = pd.read_excel(file_path, index_col=None)

    prd_data_df.drop(columns='Unnamed: 0', inplace=True)

    prd_df = prd_data_df.groupby(["Well_Completion_Name", "BOEM_FIELD", "LEASE"]).filter(lambda x: len(x) > 60)

    # Filter the DataFrame using a lambda function to calculate the percentage of zero flow rate days
    prd_df = prd_df.groupby(["Well_Completion_Name", "BOEM_FIELD", "LEASE"]).filter(
        lambda x: (x['Daily_Rates_Oil'].eq(0).sum() / len(x)) < 0.4
    )

    prd_df.reset_index(inplace=True)

    return prd_df

if __name__ == "__main__":
    file_path = 'C:\\Users\\axt43242\\Downloads\\irp-workbooks-progress\\Extracted_Prod_Data.xlsx'
    processed_df = process_data(file_path)
    print(processed_df.head())
