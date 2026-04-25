import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.optimize import curve_fit, OptimizeWarning
from sklearn.preprocessing import LabelEncoder
import os
import glob
import torch
from ipywidgets import widgets, Output
from IPython.display import display
from data_import import process_data
import matplotlib
from IPython.core.display import display, HTML
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=OptimizeWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

display(HTML("<script src='https://cdn.jsdelivr.net/npm/@widgetti/styler'></script>"))

matplotlib.use('nbagg')

# Constants
DATA_PATH = 'C:\\Users\\axt43242\\Downloads\\irp-workbooks-progress\\Extracted_Prod_Data.xlsx'
FOLDER_PATH = 'C:\\Users\\axt43242\\Downloads\\Sands'
COMP_FILE_PATH = 'C:\\Users\\axt43242\\Downloads\\irp-workbooks-progress\\Completion_Data.xlsx'

def main(selected_function_name="Exponential Decline"):
    print(f"Main function started with {selected_function_name}")
    prd_df, trend_and_resid, well_att_gg_grouped = load_and_process_data(selected_function_name)
    print("Data loaded and processed.")
    create_interactive_plot(prd_df, trend_and_resid, well_att_gg_grouped, selected_function_name)
    print("Interactive plot created.")

def load_and_process_data(selected_function_name):
    print("[DEBUG] load_and_process_data started")
    prd_df = process_data(DATA_PATH)
    print("[DEBUG] Data processed")
    decomposition_results = decompose_all_series(prd_df)
    print("[DEBUG] Decomposition results obtained")
    trend_and_resid = process_decomposition_results(decomposition_results)
    print("[DEBUG] Trend and residuals processed")

    convert_xls_to_xlsx(FOLDER_PATH)
    print("[DEBUG] XLS files converted to XLSX")
    combined_gg_df = merge_files(FOLDER_PATH)
    print("[DEBUG] Files merged")
    well_att_gg_grouped = process_combined_data(combined_gg_df)
    print("[DEBUG] Combined data processed")

    selected_function, initial_params, param_names = select_decline_function(selected_function_name)
    print(f"[DEBUG] Selected function: {selected_function.__name__}")
    params_df = fit_decline_curves(prd_df, selected_function, initial_params, param_names)
    print("[DEBUG] Decline curves fitted")
    return prd_df, trend_and_resid, well_att_gg_grouped

def create_interactive_plot(prd_df, trend_and_resid, well_att_gg_grouped, selected_function_name):
    print("[DEBUG] Creating interactive plot...")
    output = Output()

    def update_plot(selected_function):
        with output:
            try:
                output.clear_output(wait=True)
                print(f"Attempting to select function: {selected_function}")
                function, initial_params, param_names = select_decline_function(selected_function)
                print(f"Selected function: {function}, Initial params: {initial_params}, Param names: {param_names}")
                
                # Recreate params_df with the selected function's parameters
                params_df = fit_decline_curves(prd_df, function, initial_params, param_names)
                print(f"Params DataFrame: {params_df.head()}")
                if params_df.empty:
                    print(f"No parameters were fitted for the selected function: {selected_function}")
                    return

                updated_merged_df = merge_all_data(trend_and_resid, well_att_gg_grouped, params_df)
                print(f"Updated merged DataFrame: {updated_merged_df.head()}")

                print("Training and fitting started.")
                predicted_params_dict = train_and_fit(prd_df, updated_merged_df, function, initial_params, param_names)
                print("Training and fitting completed.")
                plot_final_fit(prd_df, function, predicted_params_dict, param_names, params_df)
                print("Plotting completed.")
            except Exception as e:
                print(f"An error occurred in updating plot: {e}")

    dropdown = widgets.Dropdown(
        options=["Exponential Decline", "Hyperbolic Decline", "Harmonic Decline", "Stretched Exponential Decline", "Ilk Power Law Decline", "Duong's Decline"],
        value=selected_function_name,
        description='Function:'
    )

    def on_dropdown_change(change):
        new_function = change['new']
        print(f'Selected function: {new_function}')
        update_plot(new_function)

    dropdown.observe(on_dropdown_change, names='value')

    display(dropdown)
    display(output)

    update_plot(dropdown.value)

def merge_all_data(trend_and_resid, well_att_gg_grouped, params_df):
    merged_df = pd.merge(trend_and_resid, well_att_gg_grouped, on=['Well_Completion_Name', 'BOEM_FIELD', 'LEASE'], how='inner')
    merged_df = pd.merge(merged_df, params_df, on=['Well_Completion_Name', 'BOEM_FIELD', 'LEASE'], how='inner')
    merged_df.set_index('Well_Completion_Name', inplace=True)
    merged_df.dropna(inplace=True)
    print(f"Shape of the merged dataframe (Input) is {merged_df.shape[0]}")
    return merged_df

def select_decline_function(selected_function_name="Exponential Decline"):
    functions = {
        "Exponential Decline": (exponential_decline, ['b'], [1e-2]),
        "Hyperbolic Decline": (hyperbolic_decline, ['b', 'd'], [1e-2, 0.5]),
        "Harmonic Decline": (harmonic_decline, ['b'], [1e-2]),
        "Stretched Exponential Decline": (se_decline, ['tau', 'n'], [5, 0.1]),
        "Ilk Power Law Decline": (ilk_power_law, ['a', 'b', 'n'], [1e-2, 1e-2, 1e-1]),
        "Duong's Decline": (duong, ['a', 'm'], [1e-2, 1e-2])
    }
    selected_function, param_names, initial_params = functions.get(selected_function_name, functions["Exponential Decline"])
    return selected_function, initial_params, param_names

def train_and_fit(prd_df, merged_df, selected_function, initial_params, param_names):
    X, y = prepare_data_for_tabnet(merged_df, ['BOEM_FIELD', 'LEASE', 'PLAY'], param_names)
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split_indices(X, y, merged_df.index, test_size=0.2)

    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    print("Training and evaluating TabNet model...")
    tabnet_model = train_tabnet(X_train, y_train, X_test, y_test)
    evaluate_model(tabnet_model, X_test, y_test)

    predicted_params_dict = {}
    y_pred = tabnet_model.predict(X_test)
    for i, completion in enumerate(test_idx):
        try:
            if isinstance(y_pred[i], np.ndarray):
                predicted_params_dict[completion] = y_pred[i].tolist()
            elif isinstance(y_pred[i], float):
                predicted_params_dict[completion] = [y_pred[i]]
            else:
                predicted_params_dict[completion] = y_pred[i]
        except IndexError as e:
            print(f"IndexError: {e} for test index {i} and completion {completion}")
            print(f"Predictions shape: {y_pred.shape}")
            raise e
    
    return predicted_params_dict

def train_test_split_indices(X, y, indices, **kwargs):
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(X, y, indices, **kwargs)
    return X_train, X_test, y_train, y_test, train_idx, test_idx

def plot_final_fit(prd_df, selected_function, predicted_params_dict, param_names, params_df):
    print("Starting plot_final_fit function...")
    print(f"Predicted params dict: {predicted_params_dict}")
    grouped_comps = [comp for comp in prd_df['Well_Completion_Name'].unique() if comp in predicted_params_dict]
    print(f"Grouped completions: {grouped_comps}")

    output = widgets.Output()

    def plot_and_fit(group_name):
        with output:
            output.clear_output(wait=True)
            print(f"Plotting for group: {group_name}")
            group_data = prd_df[prd_df['Well_Completion_Name'] == group_name]
            print(f"Group data for {group_name}: {group_data}")

            if group_data.empty:
                print(f"No data available for group: {group_name}")
                return

            if 'Daily_Rates_Oil' not in group_data.columns or 'Days_Elapsed' not in group_data.columns:
                print(f"Required columns are missing in the data for group: {group_name}")
                return

            q0 = group_data['Daily_Rates_Oil'].max()
            max_q0_index = group_data['Daily_Rates_Oil'].idxmax()
            x_fit = group_data.loc[max_q0_index:, 'Days_Elapsed'].values - group_data.loc[max_q0_index, 'Days_Elapsed']
            y_fit = group_data.loc[max_q0_index:, 'Daily_Rates_Oil'].values

            plt.figure(figsize=(8, 6))
            plt.plot(x_fit, y_fit, label='Daily Rates Oil', color='thistle', marker='o')

            try:
                if group_name not in predicted_params_dict:
                    raise KeyError(f"Key '{group_name}' not found in predicted_params_dict. Available keys are: {list(predicted_params_dict.keys())}")

                predicted_params = predicted_params_dict[group_name]
                if not isinstance(predicted_params, (list, np.ndarray)):
                    raise TypeError(f"Expected list or array for parameters, got {type(predicted_params)}")

                num_params = len(param_names)  # Use the correct number of params
                predicted_params = predicted_params[:num_params]
                y_fit_curve = selected_function(x_fit, *predicted_params, q0)
                plt.plot(x_fit, y_fit_curve, label=f'{selected_function.__name__} Fit (Regressed)', linestyle='--')
                print(f"{selected_function.__name__} predicted parameters: {predicted_params}")

            except KeyError as e:
                print(f"KeyError: {e}")
            except TypeError as e:
                print(f"TypeError: {e}")
            except RuntimeError as e:
                print(f"Fit could not be completed for {selected_function.__name__}: {e}")

            try:
                actual_params = params_df[params_df['Well_Completion_Name'] == group_name].iloc[0][param_names].values
                y_actual_curve = selected_function(x_fit, *actual_params, q0)
                plt.plot(x_fit, y_actual_curve, label=f'{selected_function.__name__} Fit (Actual)', linestyle=':')
                print(f"{selected_function.__name__} actual parameters: {actual_params}")
            except Exception as e:
                print(f"An error occurred in plotting actual curve: {e}")

            plt.title(f'Decline Curve Fitting for Completion {group_name}')
            plt.xlabel('Time (Days Elapsed)')
            plt.ylabel('Oil Rates in BOEPD')
            plt.legend()
            plt.tight_layout()
            plt.show()

    print("Setting up dropdown...")
    dropdown = widgets.Dropdown(
        options=grouped_comps,
        value=grouped_comps[0],
        description='Completion:',
    )

    def on_dropdown_change(change):
        plot_and_fit(change['new'])

    dropdown.observe(on_dropdown_change, names='value')

    print("Displaying dropdown and output...")
    display(dropdown)
    display(output)
    print("Interact setup completed.")

# def decompose_series(series, model, period):
#     if len(series) >= 2 * period:
#         return seasonal_decompose(series, model=model, period=period)
#     else:
#         print(f"Not enough data to decompose series. Needs at least {2 * period} observations, got {len(series)}.")
#         return None
    
def decompose_series(series, model='additive', period=12):
    if len(series) >= 2 * period:
        decomposition = seasonal_decompose(series, model=model, period=period)
        trend = decomposition.trend.dropna()
        resid = decomposition.resid.dropna()  # Calculate residuals and drop NaN values

        # Align trend and residuals by index
        aligned_index = trend.index.intersection(resid.index)
        trend = trend.loc[aligned_index]
        resid = resid.loc[aligned_index]

        return list(zip(trend.index, trend.values, resid.values))  # Return list of tuples (timestamp, trend, resid)
    else:
        print(f"Not enough data to decompose series. Needs at least {2 * period} observations, got {len(series)}.")
        return []



# def decompose_all_series(prd_df):
#     grouped = prd_df.groupby(["Well_Completion_Name", "BOEM_FIELD", "LEASE"])
#     decomposition_results = {}
#     for name, group in grouped:
#         decomposition_results[name] = {}
#         if group['Daily_Rates_Oil'].nunique() > 1:
#             series = group.set_index('Production_Date')['Daily_Rates_Oil']
#             decomposition_results[name]['Daily_Rates_Oil'] = decompose_series(series, 'additive', period=12)
#     return decomposition_results

def decompose_all_series(prd_df):
    grouped = prd_df.groupby(["Well_Completion_Name", "BOEM_FIELD", "LEASE"])
    decomposition_results = {}

    for name, group in grouped:
        decomposition_results[name] = {}
        if group['Daily_Rates_Oil'].nunique() > 1:
            series = group.set_index('Production_Date')['Daily_Rates_Oil']
            trend_with_timestamps_and_resid = decompose_series(series, 'additive', period=12)
            if trend_with_timestamps_and_resid:
                timestamps, trends, resids = zip(*trend_with_timestamps_and_resid)
                decomposition_results[name]['Timestamps'] = list(timestamps)
                decomposition_results[name]['Trend'] = list(trends)
                decomposition_results[name]['Resid'] = list(resids)  # Add residuals to results

    return decomposition_results

# def process_decomposition_results(decomposition_results):
#     trend_data = []
#     resid_data = []
#     for name, components in decomposition_results.items():
#         Well_Completion_Name, BOEM_FIELD, LEASE = name
#         row_trend = {
#             'Well_Completion_Name': Well_Completion_Name,
#             'BOEM_FIELD': BOEM_FIELD,
#             'LEASE': LEASE
#         }
#         row_resid = row_trend.copy()
#         if 'Daily_Rates_Oil' in components and components['Daily_Rates_Oil'] is not None:
#             row_trend['Oil_Trend'] = components['Daily_Rates_Oil'].trend.dropna().values
#             row_resid['Oil_Resid'] = components['Daily_Rates_Oil'].resid.dropna().values
#         trend_data.append(row_trend)
#         resid_data.append(row_resid)

#     trend_df = pd.DataFrame(trend_data)
#     resid_df = pd.DataFrame(resid_data)

#     trend_and_resid = pd.merge(trend_df, resid_df, on=['Well_Completion_Name', 'BOEM_FIELD', 'LEASE'], how='inner')
#     trend_and_resid = calculate_and_merge_stats(trend_and_resid)
#     return trend_and_resid

def process_decomposition_results(decomposition_results):
    trend_data = []
    resid_data = []

    for name, components in decomposition_results.items():
        Well_Completion_Name, BOEM_FIELD, LEASE = name
        
        # Check if the decomposition includes trend, residuals, and timestamps
        if 'Trend' in components and 'Resid' in components and 'Timestamps' in components:
            for timestamp, trend_value in zip(components['Timestamps'], components['Trend']):
                row_trend = {
                    'Well_Completion_Name': Well_Completion_Name,
                    'BOEM_FIELD': BOEM_FIELD,
                    'LEASE': LEASE,
                    'Production_Date': timestamp,
                    'Oil_Trend': trend_value
                }
                trend_data.append(row_trend)
                
            for timestamp, resid_value in zip(components['Timestamps'], components['Resid']):
                row_resid = {
                    'Well_Completion_Name': Well_Completion_Name,
                    'BOEM_FIELD': BOEM_FIELD,
                    'LEASE': LEASE,
                    'Production_Date': timestamp,
                    'Oil_Resid': resid_value
                }
                resid_data.append(row_resid)

    # Convert lists to DataFrames
    trend_df = pd.DataFrame(trend_data)
    resid_df = pd.DataFrame(resid_data)

    # Merge trend and residual data on the common columns
    trend_and_resid = pd.merge(trend_df, resid_df, on=['Well_Completion_Name', 'BOEM_FIELD', 'LEASE', 'Production_Date'], how='inner')

    # Calculate and merge additional statistics (if needed)
    trend_and_resid = calculate_and_merge_stats(trend_and_resid)

    return trend_and_resid



def calculate_stats(column):
    return column.apply(lambda x: pd.Series({
        'std_dev': np.std(x),
        'auto_corr': pd.Series(x).autocorr()
    }))

def calculate_and_merge_stats(df):
    oil_trend_stats = calculate_stats(df['Oil_Trend'])
    oil_trend_stats.columns = ['Oil_Trend_' + col for col in oil_trend_stats.columns]

    oil_resid_stats = calculate_stats(df['Oil_Resid'])
    oil_resid_stats.columns = ['Oil_Resid_' + col for col in oil_resid_stats.columns]

    df = pd.concat([df, oil_trend_stats, oil_resid_stats], axis=1)
    df.drop(columns=['Oil_Trend', 'Oil_Resid'], inplace=True)
    return df

def exponential_decline(t, b, q0):
    return q0 * np.exp(-b * t)

def harmonic_decline(t, b, q0):
    return q0 / ((1 + (b * t)))

def hyperbolic_decline(t, b, d, q0):
    base = 1 + (b * d * t)
    return q0 / (np.maximum(base, 1e-3)**(1/np.maximum(d, 1e-3)))

def se_decline(t, tau, n, q0 ):
    return q0*np.exp(-((t/np.maximum(tau, 1e-3))**n))

def ilk_power_law(t, a, b, n, q0):
    with np.errstate(over='ignore'):
        return q0 * np.exp(-((a * t) + (b * (t ** n))))

def duong(t, a, m, q0):
    with np.errstate(over='ignore', divide='ignore'):
        return q0 * (t ** (-m)) * np.exp((a / np.maximum((1 - m), 1e-3)) * ((t ** (1 - m)) - 1))

# def fit_decline_curves(trend_and_resid, selected_function, initial_params, param_names):
#     print("[DEBUG] fit_decline_curves started")
#     unique_comps = trend_and_resid['Well_Completion_Name'].unique()
#     results = []

#     for comp in unique_comps:
#         group = trend_and_resid[trend_and_resid['Well_Completion_Name'] == comp]
#         q0 = group['Oil_Trend'].max()
#         max_q0_index = group['Oil_Trend'].idxmax()
#         x_fit = group.loc[max_q0_index:, 'Days_Elapsed'].values - group.loc[max_q0_index, 'Days_Elapsed']
#         y_fit = group.loc[max_q0_index:, 'Daily_Rates_Oil'].values

#         if len(x_fit) <= len(initial_params):
#             print(f"[DEBUG] Not enough data points to fit {comp}. Skipping.")
#             continue

#         try:
#             popt, pcov = curve_fit(lambda t, *params: selected_function(t, *params, q0), x_fit, y_fit, p0=initial_params, maxfev=100000)
#             result = {
#                 'BOEM_FIELD': group['BOEM_FIELD'].iloc[0],
#                 'LEASE': group['LEASE'].iloc[0],
#                 'Well_Completion_Name': comp,
#                 'q0': q0
#             }
#             for i, param in enumerate(param_names):
#                 result[param] = popt[i]
#             results.append(result)
#         except RuntimeError as e:
#             if "Optimal parameters not found" in str(e):
#                 print(f"[DEBUG] Optimal parameters not found for {comp} within maxfev. Skipping this completion.")

#     print("[DEBUG] fit_decline_curves completed")
#     return pd.DataFrame(results)

def fit_decline_curves(trend_and_resid, selected_function, initial_params, param_names):
    print("[DEBUG] fit_decline_curves started")
    unique_comps = trend_and_resid['Well_Completion_Name'].unique()
    results = []

    for comp in unique_comps:
        group = trend_and_resid[trend_and_resid['Well_Completion_Name'] == comp]
        q0 = group['Oil_Trend'].max()
        max_q0_index = group['Oil_Trend'].idxmax()
        x_fit = group.loc[max_q0_index:, 'Days_Elapsed'].values - group.loc[max_q0_index, 'Days_Elapsed']
        y_fit = group.loc[max_q0_index:, 'Oil_Trend'].values  # Use Oil_Trend here

        if len(x_fit) <= len(initial_params):
            print(f"[DEBUG] Not enough data points to fit {comp}. Skipping.")
            continue

        try:
            popt, pcov = curve_fit(lambda t, *params: selected_function(t, *params, q0), x_fit, y_fit, p0=initial_params, maxfev=100000)
            result = {
                'BOEM_FIELD': group['BOEM_FIELD'].iloc[0],
                'LEASE': group['LEASE'].iloc[0],
                'Well_Completion_Name': comp,
                'q0': q0
            }
            for i, param in enumerate(param_names):
                result[param] = popt[i]
            results.append(result)
        except RuntimeError as e:
            if "Optimal parameters not found" in str(e):
                print(f"[DEBUG] Optimal parameters not found for {comp} within maxfev. Skipping this completion.")

    print("[DEBUG] fit_decline_curves completed")
    return pd.DataFrame(results)


def clean_data(df):
    for column in df.columns:
        df[column] = df[column].astype(str).apply(lambda x: ''.join([i if i.isprintable() else '' for i in x]))
    return df

def convert_xls_to_xlsx(folder_path):
    xls_files = glob.glob(os.path.join(folder_path, '*.xls'))
    for xls_file in xls_files:
        xlsx_file = xls_file.replace('.xls', '.xlsx')
        if not os.path.exists(xlsx_file):
            df = pd.read_excel(xls_file, engine='xlrd')
            df = clean_data(df)
            df.to_excel(xlsx_file, index=False, engine='openpyxl')
        else:
            print(f"{xlsx_file} already exists. Skipping conversion.")
    print("Conversion complete.")
    
def merge_files(folder_path):
    file_pattern1 = os.path.join(folder_path, '[0-9][0-9][0-9][0-9]sands*.xlsx')
    file_pattern2 = os.path.join(folder_path, '[0-9][0-9][0-9][0-9] Atlas Update*.xlsx')
    file_list = glob.glob(file_pattern1) + glob.glob(file_pattern2)
    if not file_list:
        print("No files matching the pattern were found.")
        return pd.DataFrame()
    print(f"Files found: {file_list}")

    df_list = []
    for file in file_list:
        try:
            df = pd.read_excel(file)
            df.rename(columns={
            'DISCOIL': 'ORIGINAL OIL',
            'DISCGAS': 'ORIGINAL GAS',
            'DISCBOE': 'ORIGINAL BOE',
            'Oil Reserves':'OIL RESERVES',
            'Gas Reserves':'GAS RESERVES',
            'BOE Reserves':'BOE RESERVES',
            'WELLAPI': 'WELL_API',
            'SD_YEAR': 'SDYEAR',
            'SD_DATE': 'SDDATE',
            'Pi': 'PI',
            'Original Oil': 'ORIGINAL OIL',
            'Original Gas': 'ORIGINAL GAS',
            'Original BOE': 'ORIGINAL BOE',
            'FSTRU':'FSTRUC',
            'RES_TYP':'RESTYP',
            'RES_TYPE':'RESTYP',        
            'Cum Oil': 'CUM OIL',
            'Cum Gas': 'CUM GAS',
            'Cum BOE': 'CUM BOE',
            'GOR ':'GOR',
            'Oil Reserves':'OIL RESERVES',
            'Gas Reserves':'GAS RESERVES',
            'BOE Reserves':'BOE RESERVES',
            'MMS_FIELD':'BOEM_FIELD',
            'WELLAPI_':'WELL_API',
            'BOEMRE_FIELD':'BOEM_FIELD',
            'TRCNT':'TCNT',
            'CUMOIL':'CUM OIL',
            'CUMGAS':'CUM GAS',
            'CUMBOE':'CUM BOE',
            'PERMEABILI':'PERMEABILITY',
            'PLAY_NAME':'PLAY'
            }, inplace=True)
            df.drop_duplicates(inplace=True)
            df_list.append(df)
        except Exception as e:
            print(f'Error reading {file}: {e}')
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.drop_duplicates(inplace=True)
    combined_df.drop(columns=[
        'SN_FORMSAND','SAND_NAME', 'ASSESSED','SDDATE', 'SDYEAR', 'SDDATEH','SDYEARH', 'SDYEAREH', 'FCLASS','FDDATE', 'FDYEAR','FTRAP2', 'EIAID', 'POOL_NAME','PLAY_TYPE','WDEP','OIL RESERVES', 'GAS RESERVES',
       'BOE RESERVES', 'CUM OIL', 'CUM GAS', 'CUM BOE', 'P_RECOIL', 'P_RECGAS', 'P_RECBOE', 'P_REMOIL', 'P_REMGAS', 'P_REMBOE', 'P_CUMCOIL', 'P_CUMGAS', 'P_CUMBOE', 'P_REMOIL', 'P_REMGAS', 'P_REMBOE','TVOL', 'OTHK', 'OAREA', 'OVOL', 'P_J', 
       'J_RECOIL', 'J_RECGAS', 'J_RECBOE','GTHK', 'GAREA', 'GVOL', 'YIELD', 'PROP', 'RECO_AF', 'RECG_AF', 'OIP', 'GIP', 'ORF', 'ORECO', 'ORECG', 'ORP', 'GRF', 'GRECO', 'GRECG', 'GRP', 'BHCOMP', 'SPGR', 'API', 'RSI', 'ORIGINAL BOE', 
       'LEASE', 'AREA_CODE', 'BLOCK_NUMBER', 'P_U','U_RECOIL', 'U_RECGAS', 'U_RECBOE', 'OLD_SAND_NAME', 'OLD_PLAY_NUM', 'OLD_PLAY_NAME', 'OLD_POOL_NAME', 'OLD_CHRONOZONE', 'OLD_PLAY_TYPE', 'P_CUMOIL', 'WELL', 'OPER_RES', 'OPER_NAME', 'PIC', 'SDTG', 'FSTAT', 'PLAY_NUM', 'NCNT', 'UCNT', 'SCNT'
    ], inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    return combined_df

def process_combined_data(combined_gg_df):
    comp_df = pd.read_excel(COMP_FILE_PATH, index_col=0)
    comp_df.WELL_API = comp_df.WELL_API.astype('Int64')
    combined_gg_df.WELL_API = combined_gg_df.WELL_API.astype('Int64')
    well_att_gg = pd.merge(comp_df, combined_gg_df, on=['SAND', 'PLAY', 'WELL_API', 'BOEM_FIELD'], how='inner')
    key_columns = ['Unique_Well_Name', 'BOEM_FIELD', 'WELL_API', 'PIC', 'PLAY', 'SAND']
    numerical_columns = well_att_gg.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = well_att_gg.select_dtypes(exclude=['number']).columns.difference(key_columns + numerical_columns).tolist()
    agg_dict = {col: 'max' for col in numerical_columns}
    agg_dict.update({col: 'first' for col in categorical_columns})
    well_att_gg_grouped = well_att_gg.groupby(key_columns, as_index=False).agg(agg_dict)
    final_columns_order = key_columns + numerical_columns + categorical_columns
    well_att_gg_grouped = well_att_gg_grouped[final_columns_order]
    well_att_gg_grouped.drop(columns=['Unique_Well_Name', 'WELL_API', 'PIC', 'SAND',
       'WELL_API', 'ORIGINAL OIL', 'ORIGINAL GAS', 'SS', 'TAREA',
       'TI', 'SDPG', 'GOR', 'BGI','TCNT', 'LAT', 'LONG', 'CHRONOZONE', 'DRIVE', 'FSTRUC', 'FTRAP1',
      'PLAREA', 'RESTYP', 'SD_TYPE'], inplace=True)
    return well_att_gg_grouped

# def prepare_data_for_tabnet(df, cat_cols, target_params):
#     for col in cat_cols:
#         le = LabelEncoder()
#         df[col] = le.fit_transform(df[col].astype(str))

#     df.dropna(subset=target_params, inplace=True)

#     X = df.drop(columns=target_params).astype('float32')
#     y = df[target_params].astype('float32')

#     return X, y

def prepare_data_for_tabnet(df, cat_cols, target_params):
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    df.dropna(subset=target_params, inplace=True)

    # Ensure we're using Oil_Trend as the target
    X = df.drop(columns=target_params).astype('float32')
    y = df[target_params].astype('float32')

    return X, y


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    print(f"Mean Squared Error on Test Data: {mse}")

# Define your TabNet model and training process
def train_tabnet(X_train, y_train, X_valid, y_valid):
    tabnet = TabNetRegressor()
    tabnet.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_name=['valid'],
        eval_metric=['rmse'],
        max_epochs=100,
        patience=10,
        batch_size=8,
        virtual_batch_size=8,
        num_workers=0,
        drop_last=False
    )
    return tabnet

if __name__ == "__main__":
    main()