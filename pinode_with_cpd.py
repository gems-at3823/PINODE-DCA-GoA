import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from ipywidgets import widgets, interact, fixed
from IPython.display import display
from data_import import process_data  # Replace with actual data loading function
import ruptures as rpt
from sklearn.metrics import r2_score, mean_squared_error
from scipy.integrate import simpson

class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.b = nn.Parameter(torch.tensor(1e-2, dtype=torch.float32))  # Initial guess for b
        self.d = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))  # Initial guess for d

    def forward(self, t, q):
        dqdt = -self.b * q ** (self.d + 1)
        return dqdt


class NeuralODE(nn.Module):
    def __init__(self, ode_func, t):
        super(NeuralODE, self).__init__()
        self.ode_func = ode_func
        self.t = t

    def forward(self, y0, extra_steps=12):
        t_max = self.t[-1].item()
        dt = (self.t[1] - self.t[0]).item()
        t_extra = torch.linspace(t_max + dt, t_max + extra_steps * dt, extra_steps)
        t_extended = torch.cat((self.t, t_extra), dim=0)
        y = odeint(self.ode_func, y0, t_extended, rtol=1e-4, atol=1e-4)
        return y, t_extended


def preprocess_data(group):
    group = group.copy()
    group.dropna(inplace=True)
    # Remove duplicate Days_Elapsed, keeping only the row with the maximum Daily_Rates_Oil for each day
    group = group.loc[group.groupby('Days_Elapsed')['Daily_Rates_Oil'].idxmax()].reset_index(drop=True)
    return group


def get_completion_data(prd_df):
    completions = {comp: prd_df[prd_df['Well_Completion_Name'] == comp] 
                   for comp in prd_df['Well_Completion_Name'].unique()}
    return completions


def detect_change_points(signal):
    algo = rpt.Pelt(model='rbf', jump=3, min_size=4).fit(signal)
    change_points = algo.predict(pen=10)
    return change_points #[:-1]  # Exclude the last point as it is the end of the data


# def train_neural_ode(t, q, cumulative_production, num_epochs=1000, learning_rate=0.01):
#     t = torch.tensor(t, dtype=torch.float32)
#     q = torch.tensor(q, dtype=torch.float32)
#     q0 = q.max()

#     ode_func = ODEFunc()
#     neural_ode = NeuralODE(ode_func, t)
#     optimizer = torch.optim.Adam(neural_ode.parameters(), lr=learning_rate)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
#     loss_fn = nn.MSELoss()

#     for epoch in range(num_epochs):
#         optimizer.zero_grad()

#         try:
#             pred_q, t_extended = neural_ode(q0)  # Get both predictions and extended time
#             pred_q = pred_q[:len(t)]  # Ensure pred_q matches the length of t
#         except Exception as e:
#             print(f"Exception at epoch {epoch}: {e}")
#             continue

#         # Data fitting loss
#         data_loss = loss_fn(pred_q.squeeze(), q.squeeze())

#         # Physics-Informed Loss: Cumulative production constraint (only for observed data)
#         predicted_cumulative = simpson(y=pred_q.squeeze().detach().numpy(), x=t.detach().numpy())  # Numerical integration to get cumulative production
#         predicted_cumulative_tensor = torch.tensor(predicted_cumulative, dtype=torch.float32)  # Convert to tensor
#         cumulative_production_tensor = torch.tensor(cumulative_production, dtype=torch.float32)  # Convert to tensor
#         cumulative_loss = torch.abs(predicted_cumulative_tensor - cumulative_production_tensor)

#         # Second Derivative Penalty (encourages flattening out of decline)
#         second_derivative = pred_q[2:] - 2 * pred_q[1:-1] + pred_q[:-2]
#         late_time_behavior_loss = torch.mean(torch.relu(-second_derivative))

#         # Flattening Out Decline (ensures production rate does not decline too rapidly at the end)
#         late_time_rate_change = pred_q[-1] - pred_q[-2]
#         late_time_flattening_loss = torch.relu(-late_time_rate_change)

#         # Combine Losses
#         total_loss = (data_loss + cumulative_loss 
#                      + 0.1 * late_time_behavior_loss 
#                      + 0.1 * late_time_flattening_loss)    

#         # Regularization: Encourage small values of `b` and reasonable `d`
#         reg_lambda = 1e-3
#         total_loss += reg_lambda * (torch.norm(ode_func.b) + torch.norm(ode_func.d))

#         # Backpropagation and optimization
#         total_loss.backward()
#         optimizer.step()
#         scheduler.step(epoch)

#         with torch.no_grad():
#             ode_func.b.clamp_(min=0.0, max=100.0)
#             ode_func.d.clamp_(min=0.0, max=1.0)

#         if epoch % 100 == 0:
#             print(f'Epoch {epoch}, Loss: {total_loss.item()}, Cumulative Loss: {cumulative_loss.item()}, b: {neural_ode.ode_func.b.item()}, d: {neural_ode.ode_func.d.item()}')

#     return neural_ode

#Model for highest flow rate after the change point
# def train_neural_ode(t, q, cumulative_production, num_epochs=1000, learning_rate=0.01):
#     # Convert time and production rate data to tensors
#     t = torch.tensor(t, dtype=torch.float32)
#     q = torch.tensor(q, dtype=torch.float32)
#     q0 = q.max()  # Start from the peak flow rate after the change point
#     ode_func = ODEFunc()
#     neural_ode = NeuralODE(ode_func, t)
#     optimizer = torch.optim.Adam(neural_ode.parameters(), lr=learning_rate)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
#     loss_fn = nn.MSELoss()

#     for epoch in range(num_epochs):
#         optimizer.zero_grad()

#         try:
#             pred_q, t_extended = neural_ode(q0)  # Get both predictions and extended time
#             pred_q = pred_q[:len(t)]  # Ensure pred_q matches the length of t
#         except Exception as e:
#             print(f"Exception at epoch {epoch}: {e}")
#             continue

#         # Data fitting loss
#         data_loss = loss_fn(pred_q.squeeze(), q.squeeze())

#         # Physics-Informed Loss: Cumulative production constraint (only for observed data)
#         predicted_cumulative = simpson(y=pred_q.squeeze().detach().numpy(), x=t.detach().numpy())  # Numerical integration to get cumulative production
#         predicted_cumulative_tensor = torch.tensor(predicted_cumulative, dtype=torch.float32)  # Convert to tensor
#         cumulative_production_tensor = torch.tensor(cumulative_production, dtype=torch.float32)  # Convert to tensor
#         cumulative_loss = torch.abs(predicted_cumulative_tensor - cumulative_production_tensor)

#         # Second Derivative Penalty (encourages flattening out of decline)
#         second_derivative = pred_q[2:] - 2 * pred_q[1:-1] + pred_q[:-2]
#         late_time_behavior_loss = torch.mean(torch.relu(-second_derivative))

#         # Flattening Out Decline (ensures production rate does not decline too rapidly at the end)
#         late_time_rate_change = pred_q[-1] - pred_q[-2]
#         late_time_flattening_loss = torch.relu(-late_time_rate_change)

#         # Combine Losses
#         total_loss = (data_loss + cumulative_loss 
#                      + 0.1 * late_time_behavior_loss 
#                      + 0.1 * late_time_flattening_loss)    

#         # Regularization: Encourage small values of `b` and reasonable `d`
#         reg_lambda = 1e-3
#         total_loss += reg_lambda * (torch.norm(ode_func.b) + torch.norm(ode_func.d))

#         # Backpropagation and optimization
#         total_loss.backward()
#         optimizer.step()
#         scheduler.step(epoch)

#         with torch.no_grad():
#             ode_func.b.clamp_(min=0.0, max=100.0)
#             ode_func.d.clamp_(min=0.0, max=1.0)

#         if epoch % 100 == 0:
#             print(f'Epoch {epoch}, Loss: {total_loss.item()}, Cumulative Loss: {cumulative_loss.item()}, b: {neural_ode.ode_func.b.item()}, d: {neural_ode.ode_func.d.item()}')

#     return neural_ode

# Training over 75 Percentile
# def train_neural_ode(t, q, cumulative_production, num_epochs=1000, learning_rate=0.01):
#     # Convert time and production rate data to tensors
#     t = torch.tensor(t, dtype=torch.float32)
#     q = torch.tensor(q, dtype=torch.float32)
    
#     # Set q0 to the 75th percentile of the flow rates after the change point
#     q0 = torch.tensor(np.percentile(q.numpy(), 75), dtype=torch.float32)

#     ode_func = ODEFunc()
#     neural_ode = NeuralODE(ode_func, t)
#     optimizer = torch.optim.Adam(neural_ode.parameters(), lr=learning_rate)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
#     loss_fn = nn.MSELoss()

#     for epoch in range(num_epochs):
#         optimizer.zero_grad()

#         try:
#             pred_q, t_extended = neural_ode(q0)  # Get both predictions and extended time
#             pred_q = pred_q[:len(t)]  # Ensure pred_q matches the length of t
#         except Exception as e:
#             print(f"Exception at epoch {epoch}: {e}")
#             continue

#         # Data fitting loss
#         data_loss = loss_fn(pred_q.squeeze(), q.squeeze())

#         # Physics-Informed Loss: Cumulative production constraint (only for observed data)
#         predicted_cumulative = simpson(y=pred_q.squeeze().detach().numpy(), x=t.detach().numpy())  # Numerical integration to get cumulative production
#         predicted_cumulative_tensor = torch.tensor(predicted_cumulative, dtype=torch.float32)  # Convert to tensor
#         cumulative_production_tensor = torch.tensor(cumulative_production, dtype=torch.float32)  # Convert to tensor
#         cumulative_loss = torch.abs(predicted_cumulative_tensor - cumulative_production_tensor)

#         # Second Derivative Penalty (encourages flattening out of decline)
#         second_derivative = pred_q[2:] - 2 * pred_q[1:-1] + pred_q[:-2]
#         late_time_behavior_loss = torch.mean(torch.relu(-second_derivative))

#         # Flattening Out Decline (ensures production rate does not decline too rapidly at the end)
#         late_time_rate_change = pred_q[-1] - pred_q[-2]
#         late_time_flattening_loss = torch.relu(-late_time_rate_change)

#         # Combine Losses
#         total_loss = (data_loss + cumulative_loss 
#                      + 0.1 * late_time_behavior_loss 
#                      + 0.1 * late_time_flattening_loss)    

#         # Regularization: Encourage small values of `b` and reasonable `d`
#         reg_lambda = 1e-3
#         total_loss += reg_lambda * (torch.norm(ode_func.b) + torch.norm(ode_func.d))

#         # Backpropagation and optimization
#         total_loss.backward()
#         optimizer.step()
#         scheduler.step(epoch)

#         with torch.no_grad():
#             ode_func.b.clamp_(min=0.0, max=100.0)
#             ode_func.d.clamp_(min=0.0, max=1.0)

#         if epoch % 100 == 0:
#             print(f'Epoch {epoch}, Loss: {total_loss.item()}, Cumulative Loss: {cumulative_loss.item()}, b: {neural_ode.ode_func.b.item()}, d: {neural_ode.ode_func.d.item()}')

#     return neural_ode

def smooth_data(q_data, window_size=5):
    """Apply a moving average to smooth the data."""
    smoothed_q = pd.Series(q_data.flatten()).rolling(window=window_size, min_periods=1).mean().values
    return smoothed_q

#Training for smoothed window approach
def train_neural_ode(t, q, cumulative_production, num_epochs=1000, learning_rate=0.01):
    # Convert time and production rate data to tensors
    t = torch.tensor(t, dtype=torch.float32)
    q = torch.tensor(q, dtype=torch.float32)

    # Smooth the production data
    smoothed_q = smooth_data(q.numpy())
    
    # Set q0 based on the smoothed data
    q0 = torch.tensor(smoothed_q.max(), dtype=torch.float32)  # Example: using the maximum of the smoothed data

    ode_func = ODEFunc()
    neural_ode = NeuralODE(ode_func, t)
    optimizer = torch.optim.Adam(neural_ode.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        try:
            pred_q, t_extended = neural_ode(q0)  # Get both predictions and extended time
            pred_q = pred_q[:len(t)]  # Ensure pred_q matches the length of t
        except Exception as e:
            print(f"Exception at epoch {epoch}: {e}")
            continue

        # Data fitting loss
        data_loss = loss_fn(pred_q.squeeze(), q.squeeze())

        # Physics-Informed Loss: Cumulative production constraint (only for observed data)
        predicted_cumulative = simpson(y=pred_q.squeeze().detach().numpy(), x=t.detach().numpy())  # Numerical integration to get cumulative production
        predicted_cumulative_tensor = torch.tensor(predicted_cumulative, dtype=torch.float32)  # Convert to tensor
        cumulative_production_tensor = torch.tensor(cumulative_production, dtype=torch.float32)  # Convert to tensor
        cumulative_loss = torch.abs(predicted_cumulative_tensor - cumulative_production_tensor)

        # Second Derivative Penalty (encourages flattening out of decline)
        second_derivative = pred_q[2:] - 2 * pred_q[1:-1] + pred_q[:-2]
        late_time_behavior_loss = torch.mean(torch.relu(-second_derivative))

        # Flattening Out Decline (ensures production rate does not decline too rapidly at the end)
        late_time_rate_change = pred_q[-1] - pred_q[-2]
        late_time_flattening_loss = torch.relu(-late_time_rate_change)

        # Combine Losses
        total_loss = (data_loss + cumulative_loss 
                     + 0.1 * late_time_behavior_loss 
                     + 0.1 * late_time_flattening_loss)    

        # Regularization: Encourage small values of `b` and reasonable `d`
        reg_lambda = 1e-3
        total_loss += reg_lambda * (torch.norm(ode_func.b) + torch.norm(ode_func.d))

        # Backpropagation and optimization
        total_loss.backward()
        optimizer.step()
        scheduler.step(epoch)

        with torch.no_grad():
            ode_func.b.clamp_(min=0.0, max=100.0)
            ode_func.d.clamp_(min=0.0, max=1.0)

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss.item()}, Cumulative Loss: {cumulative_loss.item()}, b: {neural_ode.ode_func.b.item()}, d: {neural_ode.ode_func.d.item()}')

    return neural_ode


def plot_results(neural_ode, t_data, q_data, scaler_q, scaler_t, t_max_idx, days_elapsed, group_before, group_after, change_point_idx, change_points, completion_name):
    print(f"\n[DEBUG] Plotting results with change_point_idx: {change_point_idx}")

    # Determine where the curve should start
    if change_point_idx is not None:
        start_idx = change_points[change_point_idx]  # Use the selected change point
    else:
        start_idx = t_max_idx

    print(f"[DEBUG] start_idx: {start_idx}")

    # Ensure start_idx is within the bounds of days_elapsed
    if start_idx >= len(days_elapsed):
        print(f"[DEBUG] start_idx {start_idx} is out of bounds. Adjusting to the last index.")
        start_idx = len(days_elapsed) - 1

    print(f"[DEBUG] Corresponding day at start_idx: {days_elapsed[start_idx]}")

    # Get the original flow rate before scaling
    q0_value_unscaled = group_after['Daily_Rates_Oil'].iloc[0]
    
    # Manually scale this value using the scaler
    q0_value_scaled = scaler_q.transform([[q0_value_unscaled]])[0][0]

    # Use this scaled value as the initial condition
    q0 = torch.tensor([q0_value_scaled], dtype=torch.float32)

    print(f"[DEBUG] Unscaled q0_value: {q0_value_unscaled}")
    print(f"[DEBUG] Scaled q0_value: {q0_value_scaled}")

    # Generate predictions using the Neural ODE model
    with torch.no_grad():
        q_pred, t_extended = neural_ode(q0, extra_steps=12)
        q_pred = q_pred.numpy()

    # Inverse transform the predictions back to original scale
    q_pred = scaler_q.inverse_transform(q_pred.reshape(-1, 1)).flatten()

    # Calculate R² score and MSE
    y_true = group_after['Daily_Rates_Oil'].values  # Actual values
    y_pred = q_pred[:len(y_true)]  # Predicted values, ensuring the lengths match

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    print(f"R² score: {r2}")
    print(f"MSE: {mse}")

    # Convert t_extended to days, ensuring it matches q_pred length
    t_extended_days = scaler_t.inverse_transform(t_extended.numpy().reshape(-1, 1)).flatten()

    # Ensure that t_extended_days and q_pred have the same length
    assert len(t_extended_days) == len(q_pred), f"Length mismatch: t_extended_days={len(t_extended_days)}, q_pred={len(q_pred)}"

    # Plot the results
    plt.figure(figsize=(8, 6))

    # Plot data points before and after the selected change point with different colors
    plt.scatter(group_before['Days_Elapsed'], group_before['Daily_Rates_Oil'], color='thistle', label='Before Change Point', marker='o')
    plt.scatter(group_after['Days_Elapsed'], group_after['Daily_Rates_Oil'], color='olivedrab', label='After Change Point', marker='o')

    # Highlight all change points
    for cp_idx in change_points[:-1]:  # Skip the last point because it's the end of the data
        plt.axvline(x=days_elapsed[cp_idx], color='lightsteelblue', linestyle='--', label='Change Point' if cp_idx == change_points[0] else "")

    # Plot the predicted curve starting from the selected change point
    plt.plot(t_extended_days, q_pred, label='Predicted', linestyle='-', color='gold')

    plt.title(f"Change Point Detection and Neural ODE Fit for {completion_name}")
    plt.xlabel('Time (Days Elapsed)')
    plt.ylabel('Production Rate in BOEPD')
    plt.legend()
    plt.tight_layout()
    plt.show()

# def update_plot(completion_data, completion, change_point_idx):
#     group = preprocess_data(completion_data[completion])

#     # Detect change points
#     change_points = detect_change_points(group['Daily_Rates_Oil'].values)
#     print(f"Detected change points: {change_points}")

#     # Default to first change point instead of q_max
#     start_idx = change_points[change_point_idx]  # Start from the selected change point

#     # Slice the data from the selected change point
#     t_data = group['Days_Elapsed'].iloc[start_idx:].values.reshape(-1, 1).astype(np.float32)
#     q_data = group['Daily_Rates_Oil'].iloc[start_idx:].values.reshape(-1, 1).astype(np.float32)

#     if len(t_data) == 0 or len(q_data) == 0:
#         print("No data points available.")
#         return

#     scaler_q, scaler_t = MinMaxScaler(), MinMaxScaler()
#     q_data = scaler_q.fit_transform(q_data)
#     t_data = scaler_t.fit_transform(t_data)

#     # Calculate the actual cumulative production up to the last available point
#     cumulative_production = simpson(y=group['Daily_Rates_Oil'].iloc[start_idx:], x=group['Days_Elapsed'].iloc[start_idx:])

#     neural_ode = train_neural_ode(t_data.flatten(), q_data, cumulative_production)

#     # Call plot_results with correct arguments
#     plot_results(
#         neural_ode, 
#         t_data.flatten(), 
#         q_data, 
#         scaler_q, 
#         scaler_t, 
#         start_idx, 
#         group['Days_Elapsed'].values, 
#         group.iloc[:start_idx], 
#         group.iloc[start_idx:], 
#         change_point_idx,  # Use the selected change point index
#         change_points, 
#         completion
#     )

# Update Plot for Q_Max
# def update_plot(completion_data, completion, change_point_idx):
#     group = preprocess_data(completion_data[completion])

#     # Detect change points
#     change_points = detect_change_points(group['Daily_Rates_Oil'].values)
#     print(f"Detected change points: {change_points}")

#     # Start from the selected change point
#     start_idx = change_points[change_point_idx]

#     # Slice the data from the selected change point to the end
#     group_after_change = group.iloc[start_idx:]

#     # Identify the time corresponding to the highest flow rate after the change point
#     t_max_idx = group_after_change['Daily_Rates_Oil'].idxmax()
    
#     # Slice the data starting from the highest flow rate after the change point
#     t_data = group['Days_Elapsed'].iloc[t_max_idx:].values.reshape(-1, 1).astype(np.float32)
#     q_data = group['Daily_Rates_Oil'].iloc[t_max_idx:].values.reshape(-1, 1).astype(np.float32)

#     if len(t_data) == 0 or len(q_data) == 0:
#         print("No data points available.")
#         return

#     scaler_q, scaler_t = MinMaxScaler(), MinMaxScaler()
#     q_data = scaler_q.fit_transform(q_data)
#     t_data = scaler_t.fit_transform(t_data)

#     # Calculate the cumulative production from the max flow rate point to the last available point
#     cumulative_production = simpson(y=group['Daily_Rates_Oil'].iloc[t_max_idx:], x=group['Days_Elapsed'].iloc[t_max_idx:])

#     # Train the Neural ODE model on this subset of data
#     neural_ode = train_neural_ode(t_data.flatten(), q_data, cumulative_production)

#     # Plot the results starting from the highest flow rate after the change point
#     plot_results(
#         neural_ode, 
#         t_data.flatten(), 
#         q_data, 
#         scaler_q, 
#         scaler_t, 
#         t_max_idx, 
#         group['Days_Elapsed'].values, 
#         group.iloc[:t_max_idx], 
#         group.iloc[t_max_idx:], 
#         change_point_idx,  # Use the selected change point index
#         change_points, 
#         completion
#     )

def update_plot(completion_data, completion, change_point_idx):
    group = preprocess_data(completion_data[completion])

    # Detect change points
    change_points = detect_change_points(group['Daily_Rates_Oil'].values)
    print(f"Detected change points: {change_points}")

    # Start from the selected change point
    start_idx = change_points[change_point_idx]

    # Slice the data from the selected change point to the end
    group_after_change = group.iloc[start_idx:]

    # Identify the time corresponding to the highest flow rate after the change point
    t_max_idx = group_after_change['Daily_Rates_Oil'].idxmax()

    # Slice the data starting from the highest flow rate after the change point
    t_data = group['Days_Elapsed'].iloc[t_max_idx:].values.reshape(-1, 1).astype(np.float32)
    q_data = group['Daily_Rates_Oil'].iloc[t_max_idx:].values.reshape(-1, 1).astype(np.float32)

    if len(t_data) == 0 or len(q_data) == 0:
        print("No data points available.")
        return

    scaler_q, scaler_t = MinMaxScaler(), MinMaxScaler()
    q_data = scaler_q.fit_transform(q_data)
    t_data = scaler_t.fit_transform(t_data)

    # Calculate the cumulative production from the max flow rate point to the last available point
    cumulative_production = simpson(y=group['Daily_Rates_Oil'].iloc[t_max_idx:], x=group['Days_Elapsed'].iloc[t_max_idx:])

    # Train the Neural ODE model on this subset of data
    neural_ode = train_neural_ode(t_data.flatten(), q_data, cumulative_production)

    # Plot the results starting from the highest flow rate after the change point
    plot_results(
        neural_ode, 
        t_data.flatten(), 
        q_data, 
        scaler_q, 
        scaler_t, 
        t_max_idx, 
        group['Days_Elapsed'].values, 
        group.iloc[:t_max_idx], 
        group.iloc[t_max_idx:], 
        change_point_idx,  # Use the selected change point index
        change_points, 
        completion
    )
    
def main():
    file_path = 'C:\\Users\\axt43242\\Downloads\\irp-files-clean\\Extracted_Prod_Data.xlsx'
    try:
        prd_df = process_data(file_path)
        completion_data = get_completion_data(prd_df)

        completion_dropdown = widgets.Dropdown(
            options=completion_data.keys(),
            description='Completion:',
        )

        initial_completion = next(iter(completion_data.keys()))
        group = preprocess_data(completion_data[initial_completion])

        # Detect initial change points
        change_points = detect_change_points(group['Daily_Rates_Oil'].values)
        max_index = len(change_points) - 1

        change_point_slider = widgets.IntSlider(
            min=0,
            max=max_index,
            step=1,
            value=0,
            description='Change Point:',
            continuous_update=False,
        )

        def update_ui(completion, change_point_idx):
            update_plot(completion_data, completion, change_point_idx)

        interact(update_ui, completion=completion_dropdown, change_point_idx=change_point_slider)
        display(completion_dropdown)
        display(change_point_slider)

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
