import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error

def exp_model(x, a, b):
    return a * np.exp(b * x)


# load data
df = pd.read_parquet("/share/nas2_3/jfont/ILP-Radio-Array/metadata_combined.parquet", engine="fastparquet")
df_node = df[df["node_num"] == 17]

# compute the mean and std of the optimisation time per antenna number
stats = df_node.groupby("subarray_number")["optimisation_time"].agg(["median", "mean", "std"]).reset_index()
stats["median"] = stats["median"] / 3600
stats["std"] = stats["std"] / 3600
xdata = stats["subarray_number"].values
ydata = stats["median"].values
sigma = stats["std"].values
sigma[sigma == 0] = np.min(sigma[sigma > 0]) * 0.1
sigma = np.nan_to_num(sigma, nan=np.nanmin(sigma[sigma > 0])*0.1)

# fit exponential to data
popt, pcov = curve_fit(exp_model, xdata, ydata, sigma=sigma, absolute_sigma=True, p0=(0.02, 0.28), maxfev=100000)
a_fit, b_fit = popt

x_fit = np.linspace(min(xdata), max(xdata), 500)
y_fit = exp_model(x_fit, a_fit, b_fit)
y_pred = exp_model(xdata, a_fit, b_fit)

# calculate stats
chi2 = np.sum(((ydata - y_pred) / sigma) ** 2)
dof = len(ydata) - len(popt)
reduced_chi2 = chi2 / dof

rmse = np.sqrt(mean_squared_error(ydata, y_pred))
mae = mean_absolute_error(ydata, y_pred)

n = len(ydata)
p = len(popt)
residual_sum_squares = np.sum((ydata - y_pred) ** 2)
sigma2 = residual_sum_squares / n
log_likelihood = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
aic = 2 * p - 2 * log_likelihood
bic = p * np.log(n) - 2 * log_likelihood

# plot data
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Linear scale
axs[0].errorbar(
    stats["subarray_number"], stats["median"], yerr=stats["std"], fmt="o-", ecolor="gray",
    capsize=4, markersize=5, label="Optimisation Time"
)
axs[0].plot(x_fit, y_fit, 'r-', label=f'Fit: y = {a_fit:.2f} * exp({b_fit:.4f} x)')
axs[0].set_xlabel("Number of Subarrays")
axs[0].set_ylabel("Optimisation Time (hour)")
axs[0].set_title("Linear Scale")
axs[0].grid(True, linestyle="--", alpha=0.6)
axs[0].legend()

# Logarithmic scale
axs[1].errorbar(
    stats["subarray_number"], stats["median"], yerr=stats["std"], fmt="o-", ecolor="gray",
    capsize=4, markersize=5, label="Optimisation Time"
)
axs[1].plot(x_fit, y_fit, 'r-', label=f'Fit: y = {a_fit:.2f} * exp({b_fit:.4f} x)')
axs[1].set_yscale("log")
axs[1].set_xlabel("Number of Subarrays")
axs[1].set_title("Log Scale")
axs[1].grid(True, linestyle="--", alpha=0.6)
axs[1].legend()

plt.suptitle("Mean Optimisation Time vs Subarray Number (Node Number = 2)")
plt.savefig("plots/node_num_opt_time_sub2_side_by_side.png")

print(f"Fitted parameters: a = {a_fit:.6g}, b = {b_fit:.6g}")
print(f"Chi-squared: {chi2:.4f}")
print(f"Reduced Chi-squared: {reduced_chi2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"AIC: {aic:.4f}")
print(f"BIC: {bic:.4f}")
print(exp_model(8, a_fit, b_fit))
