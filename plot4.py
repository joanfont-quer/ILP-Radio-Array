import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt
from graph_loader import ska_mid_full_graph

# Load your data
df = pd.read_parquet("/share/nas2_3/jfont/ILP-Radio-Array/metadata_files/metadata_heuristic_1.parquet", engine="fastparquet")
df_sub2 = df[df["subarray_number"] == 2]
# df_node82 = df_sub2[df_sub2["node_num"] >= 83]

stats = df_sub2.groupby("node_num")["optimisation_time"].agg(["median", "mean", "std"]).reset_index()
xdata = stats["node_num"].values
ydata = stats["mean"].values
sigma = stats["std"].values
sigma[sigma == 0] = np.min(sigma[sigma > 0]) * 0.1
sigma = np.nan_to_num(sigma, nan=np.nanmin(sigma[sigma > 0]) * 0.1)

# Degree of polynomial to fit
poly_degree = 4

# Fit polynomial
poly_coeffs = np.polyfit(xdata, ydata, deg=poly_degree, w=1/sigma)
poly_model = np.poly1d(poly_coeffs)

# Predictions
x_fit = np.linspace(min(xdata), max(xdata), 500)
y_fit = poly_model(x_fit)
y_pred = poly_model(xdata)

# Stats
chi2 = np.sum(((ydata - y_pred) / sigma) ** 2)
dof = len(ydata) - (poly_degree + 1)
reduced_chi2 = chi2 / dof
rmse = np.sqrt(mean_squared_error(ydata, y_pred))
mae = mean_absolute_error(ydata, y_pred)
n = len(ydata)
p = poly_degree + 1
residual_sum_squares = np.sum((ydata - y_pred) ** 2)
sigma2 = residual_sum_squares / n
log_likelihood = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
aic = 2 * p - 2 * log_likelihood
bic = p * np.log(n) - 2 * log_likelihood

# Plot
plt.errorbar(stats["node_num"], stats["mean"], yerr=stats["std"], fmt="o-", ecolor="gray",
                capsize=4, markersize=5, label="Optimisation Time")
plt.plot(x_fit, y_fit, 'r-', label=f'Poly fit (deg={poly_degree})')
plt.xlabel("Number of Nodes")
plt.ylabel("Optimisation Time (sec)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.savefig("plots/node_num_opt_time_heuristic_polyfit.png")

# Output stats
print(f"Polynomial coefficients: {poly_coeffs}")
print(f"Chi-squared: {chi2:.4f}")
print(f"Reduced Chi-squared: {reduced_chi2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"AIC: {aic:.4f}")
print(f"BIC: {bic:.4f}")

graph, _, _ = ska_mid_full_graph()
x_pred = len(list(graph.nodes()))
y_pred_final = poly_model(x_pred)

sigma_resid = np.sqrt(residual_sum_squares / dof)

print(f"Predicted optimisation time for {x_pred} nodes = {y_pred_final:.2e} ± {sigma_resid:.2e} sec")
