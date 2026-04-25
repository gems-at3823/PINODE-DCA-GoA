import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import ruptures as rpt
from ipywidgets import interact, widgets, VBox, HBox
import IPython.display as display
from sklearn.metrics import r2_score, mean_squared_error
from IPython.display import clear_output

def main():
    file_path = 'C:\\Users\\axt43242\\Downloads\\irp-workbooks-progress\\Extracted_Prod_Data.xlsx'
    try:
        prd_df = load_data(file_path)
        cln_prd_df = filter_data(prd_df)
        create_widgets(cln_prd_df)
    except Exception as e:
        print(f"Error occurred: {e}")


def load_data(file_path):
    try:
        prd_data_df = pd.read_excel(file_path, index_col=None)
        prd_data_df.drop(columns='Unnamed: 0', inplace=True)
        prd_df = prd_data_df.groupby(["Well_Completion_Name", "BOEM_FIELD", "LEASE"]).filter(lambda x: len(x) > 24)
        prd_df = prd_df.groupby(["Well_Completion_Name", "BOEM_FIELD", "LEASE"]).filter(
            lambda x: (x['Daily_Rates_Oil'].eq(0).sum() / len(x)) < 0.4
        )
        prd_df.reset_index(inplace=True)
        return prd_df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        raise
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def filter_data(prd_df):
    cln_prd_df = prd_df.groupby('Well_Completion_Name').filter(
        lambda x: (x['Days_Elapsed'].max() >= 365))
    return cln_prd_df


def exponential_decline(t, q0, b):
    return q0 * np.exp(-b * t)


def harmonic_decline(t, q0, b):
    return q0 / (1 + (b * t))


def hyperbolic_decline(t, q0, b, d):
    base = 1 + (b * d * t)
    return q0 / (np.maximum(base, 1e-5) ** (1 / d))


def se_decline(t, q0, tau, n):
    return q0 * np.exp(-((t / tau) ** n))


def ilk_power_law(t, q0, a, b, n):
    return q0 * np.exp(-(a * t) - (b * (t ** n)))


def duong(t, q0, a, m):
    # Ensure t does not contain NaNs or Infs
    t = np.nan_to_num(t, nan=1e-5, posinf=1e5, neginf=1e-5)
    
    # Prevent division by zero or negative values inside the exp function
    a = np.max([1e-3, a])
    m = np.max([1e-3, m])

    # Compute the Duong function
    try:
        result = q0 * (t ** (-m)) * np.exp((a / (1 - m)) * ((t ** (1 - m)) - 1))
    except FloatingPointError:
        result = np.nan_to_num(q0 * (t ** (-m)) * np.exp((a / (1 - m)) * ((t ** (1 - m)) - 1)), nan=0, posinf=0, neginf=0)
        
    return result


def plot_and_fit(group_name, change_point, cln_prd_df, models_selected):

    clear_output(wait=True)
    plt.close('all')


    group_data = cln_prd_df[cln_prd_df['Well_Completion_Name'] == group_name]
    production_values = group_data['Daily_Rates_Oil'].values

    cp_detector = rpt.Pelt(model='rbf', jump=3, min_size=4)
    cp_detected = cp_detector.fit(production_values).predict(pen=3)
    cp_detected = [cp + 1 for cp in cp_detected]

    plt.figure(figsize=(8, 6))
    plt.plot(group_data['Days_Elapsed'], production_values, label='Daily Rates Oil', color='thistle', marker='o')

    for cp in cp_detected[:-1]:  # Skip the last point because it's the end of the data
        plt.axvline(x=group_data['Days_Elapsed'].iloc[cp], color='lightsteelblue', linestyle='--', label='Change Point' if cp == cp_detected[0] else "")

    if change_point < len(cp_detected) - 1:  # Ensure valid change point
        start_idx = cp_detected[change_point - 1]
        x_fit = group_data['Days_Elapsed'].iloc[start_idx:].values - group_data['Days_Elapsed'].iloc[start_idx]
        y_fit = production_values[start_idx:]

        q0_initial = y_fit[0]

        # Fit models based on selection
        time_fitted = np.linspace(x_fit.min(), x_fit.max(), 200)
        time_fitted_full = time_fitted + group_data['Days_Elapsed'].iloc[start_idx]

        if 'Exponential' in models_selected:
            popt_exp, _ = curve_fit(lambda t, b: exponential_decline(t, q0_initial, b), x_fit, y_fit, p0=[0.0001], maxfev=150000)
            production_fitted_exp = exponential_decline(time_fitted, q0_initial, *popt_exp)
            plt.plot(time_fitted_full, production_fitted_exp, label='Exponential Decline', linestyle='-', color='gold')
            r2_exp = r2_score(y_fit, exponential_decline(x_fit, q0_initial, *popt_exp))
            mse_exp = mean_squared_error(y_fit, exponential_decline(x_fit, q0_initial, *popt_exp))
            print(f'Exponential Decline: R² = {r2_exp:.4f}, MSE = {mse_exp:.4f}')

        if 'Hyperbolic' in models_selected:
            popt_hyper, _ = curve_fit(lambda t, b, d: hyperbolic_decline(t, q0_initial, b, d), x_fit, y_fit, p0=[0.00001, 0.5], maxfev=150000)
            production_fitted_hyper = hyperbolic_decline(time_fitted, q0_initial, *popt_hyper)
            plt.plot(time_fitted_full, production_fitted_hyper, label='Hyperbolic Decline', linestyle='--', color='darkorange')
            r2_hyper = r2_score(y_fit, hyperbolic_decline(x_fit, q0_initial, *popt_hyper))
            mse_hyper = mean_squared_error(y_fit, hyperbolic_decline(x_fit, q0_initial, *popt_hyper))
            print(f'Hyperbolic Decline: R² = {r2_hyper:.4f}, MSE = {mse_hyper:.4f}')

        if 'Harmonic' in models_selected:
            popt_harm, _ = curve_fit(lambda t, b: harmonic_decline(t, q0_initial, b), x_fit, y_fit, p0=[0.00001], maxfev=150000)
            production_fitted_harm = harmonic_decline(time_fitted, q0_initial, *popt_harm)
            plt.plot(time_fitted_full, production_fitted_harm, label='Harmonic Decline', linestyle='-.', color='limegreen')
            r2_harm = r2_score(y_fit, harmonic_decline(x_fit, q0_initial, *popt_harm))
            mse_harm = mean_squared_error(y_fit, harmonic_decline(x_fit, q0_initial, *popt_harm))
            print(f'Harmonic Decline: R² = {r2_harm:.4f}, MSE = {mse_harm:.4f}')

        if 'Stretched Exponential' in models_selected:
            popt_sed, _ = curve_fit(lambda t, tau, n: se_decline(t, q0_initial, tau, n), x_fit, y_fit, p0=[7, 0.1], maxfev=150000)
            production_fitted_sed = se_decline(time_fitted, q0_initial, *popt_sed)
            plt.plot(time_fitted_full, production_fitted_sed, label='Stretched Exponential Decline', linestyle=':', color='violet')
            r2_sed = r2_score(y_fit, se_decline(x_fit, q0_initial, *popt_sed))
            mse_sed = mean_squared_error(y_fit, se_decline(x_fit, q0_initial, *popt_sed))
            print(f'Stretched Exponential Decline: R² = {r2_sed:.4f}, MSE = {mse_sed:.4f}')


        if 'Ilk Power Law' in models_selected:
            popt_power, _ = curve_fit(lambda t, a, b, n: ilk_power_law(t, q0_initial, a, b, n), x_fit, y_fit, p0=[0.001, 0.001, 0.1], maxfev=150000)
            production_fitted_power = ilk_power_law(time_fitted, q0_initial, *popt_power)
            plt.plot(time_fitted_full, production_fitted_power, label='Ilk Power Law Decline', linestyle='solid', color='brown')
            r2_power = r2_score(y_fit, ilk_power_law(x_fit, q0_initial, *popt_power))
            mse_power = mean_squared_error(y_fit, ilk_power_law(x_fit, q0_initial, *popt_power))
            print(f'Ilk Power Law Decline: R² = {r2_power:.4f}, MSE = {mse_power:.4f}')

        if 'Duong' in models_selected:
            popt_duong, _ = curve_fit(lambda t, a, m: duong(t, q0_initial, a, m), x_fit, y_fit, p0=[0.001, 0.1], maxfev=150000)
            production_fitted_duong = duong(time_fitted, q0_initial, *popt_duong)
            plt.plot(time_fitted_full, production_fitted_duong, label='Duong Decline', linestyle='solid', color='grey')
            r2_duong = r2_score(y_fit, duong(x_fit, q0_initial, *popt_duong))
            mse_duong = mean_squared_error(y_fit, duong(x_fit, q0_initial, *popt_duong))
            print(f'Duong Decline: R² = {r2_duong:.4f}, MSE = {mse_duong:.4f}')

        plt.title(f"Change Point Detection and Arps's Decline Curve Fitting for Completion {group_name}")
        plt.xlabel('Time (Days Elapsed)')
        plt.ylabel('Oil Rates in BOEPD')
        plt.legend()
        plt.tight_layout()
        plt.show()


def create_widgets(cln_prd_df):
    grouped_comps = cln_prd_df['Well_Completion_Name'].unique()

    dropdown = widgets.Dropdown(
        options=grouped_comps,
        value=grouped_comps[0],
        description='Completion:',
    )

    change_point_slider = widgets.IntSlider(
        value=1,
        min=1,
        max=20,  # This will be updated dynamically
        step=1,
        description='Change Point:',
        disabled=False,
    )

    model_options = ['Exponential', 'Hyperbolic', 'Harmonic', 'Stretched Exponential', 'Ilk Power Law', 'Duong']
    model_selection = widgets.SelectMultiple(
        options=model_options,
        value=model_options,  # Default to all selected
        description='Models',
        disabled=False
    )

    def update_change_point_slider(group_name):
        group_data = cln_prd_df[cln_prd_df['Well_Completion_Name'] == group_name]
        production_values = group_data['Daily_Rates_Oil'].values
        cp_detector = rpt.Pelt(model='rbf', jump=7, min_size=5)
        cp_detected = cp_detector.fit(production_values).predict(pen=3)
        change_point_slider.max = len(cp_detected) - 1  # Update slider max value dynamically

    widgets.interactive(update_change_point_slider, group_name=dropdown)

    interact(
        plot_and_fit,
        group_name=dropdown,
        change_point=change_point_slider,
        cln_prd_df=widgets.fixed(cln_prd_df),
        models_selected=model_selection
    )

    display(VBox([dropdown, change_point_slider, model_selection]))


if __name__ == "__main__":
    main()
