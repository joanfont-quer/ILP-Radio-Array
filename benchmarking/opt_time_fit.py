import os
import numpy as np
from graph_loader import ska_mid_full_graph
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error


def exp_model(x, a, b):
    return a * np.exp(b * x)


def exp_jacobian(x, popt):
    a, b = popt
    return np.array([np.exp(b * x), a * x * np.exp(b * x)])


def poly(x, a, b):
    return a * x ** b


def poly_jacobian(x, popt):
    a, b = popt
    return np.array([x ** b, a * x ** b * np.log(x)])


class OptimisationBenchmark:
    def __init__(self, config, model, jacobian=None, model_p0=None, x_param="node_num"):
        self.config = config
        self.model = model
        self.df = config.metadata()
        self.jacobian = jacobian
        self.model_p0 = model_p0
        self.stats = None
        self.popt = None
        self.pcov = None
        self.x_param = x_param

        if self.x_param not in self.df.columns:
            raise ValueError(
                f"x_param='{self.x_param}' not found in metadata columns. "
                f"Available columns are: {list(self.df.columns)}"
            )

    def compute_stats(self):
        stats = self.df.groupby(self.x_param)["optimisation_time"].agg(["median", "mean", "std"]).reset_index()
        stats["mean"] /= 3600
        stats["std"] /= 3600

        sigma = stats["std"].values
        sigma[sigma == 0] = np.min(sigma[sigma > 0]) * 0.1
        sigma = np.nan_to_num(sigma, nan=np.nanmin(sigma[sigma > 0]) * 0.1)

        self.stats = stats
        self.sigma = sigma
        return stats

    def fit_model(self):
        xdata = self.stats[self.x_param].values
        ydata = self.stats["mean"].values

        popt, pcov = curve_fit(
            self.model,
            xdata,
            ydata,
            sigma=self.sigma,
            absolute_sigma=True,
            p0=self.model_p0,
            maxfev=100000
        )

        self.popt, self.pcov = popt, pcov
        return popt, pcov

    def evaluate_model(self):
        xdata = self.stats[self.x_param].values
        ydata = self.stats["mean"].values

        y_pred = self.model(xdata, *self.popt)

        chi2 = np.sum(((ydata - y_pred) / self.sigma) ** 2)
        dof = len(ydata) - len(self.popt)
        reduced_chi2 = chi2 / dof

        rmse = np.sqrt(mean_squared_error(ydata, y_pred))
        mae = mean_absolute_error(ydata, y_pred)

        n = len(ydata)
        p = len(self.popt)
        residual_sum_squares = np.sum((ydata - y_pred) ** 2)
        sigma2 = residual_sum_squares / n
        log_likelihood = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
        aic = 2 * p - 2 * log_likelihood
        bic = p * np.log(n) - 2 * log_likelihood

        results = dict(
            chi2=chi2, reduced_chi2=reduced_chi2, rmse=rmse, mae=mae, aic=aic, bic=bic
        )

        print(f"Fit metrics:")
        for k, v in results.items():
            print(f"    {k:>12s}: {v:.4f}")
        return results

    def plot_results(self):
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

        xdata = self.stats[self.x_param].values

        x_fit = np.linspace(min(xdata), max(xdata), 500)
        y_fit = self.model(x_fit, *self.popt)

        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        # Linear scale
        axs[0].errorbar(
            self.stats[self.x_param], self.stats["mean"], yerr=self.stats["std"], fmt="o-", ecolor="gray",
            capsize=4, markersize=5, label="Optimisation Time"
        )
        param_str = ", ".join(f"{p:.2g}" for p in self.popt)
        axs[0].plot(x_fit, y_fit, 'r-', label=f'{self.model.__name__}({param_str})')
        axs[0].set_xlabel(self.x_param)
        axs[0].set_ylabel("Optimisation Time (hours)")
        axs[0].grid(True, linestyle="--", alpha=0.6)
        axs[0].legend()

        # Logarithmic scale
        axs[1].errorbar(
            self.stats[self.x_param], self.stats["mean"], yerr=self.stats["std"], fmt="o-", ecolor="gray",
            capsize=4, markersize=5, label="Optimisation Time"
        )
        axs[1].plot(x_fit, y_fit, 'r-', label=f'{self.model.__name__}({param_str})')
        axs[1].set_yscale("log")
        axs[1].set_xlabel(self.x_param)
        axs[1].grid(True, linestyle="--", alpha=0.6)
        axs[1].legend()

        plot_path = os.path.join(self.config.OUTPUT_DIR, f"{self.config.NAME}_fit.png")
        plt.savefig(plot_path)

    def predict(self):
        graph, _, _ = ska_mid_full_graph()
        x_pred = len(list(graph.nodes()))
        y_pred = self.model(x_pred, *self.popt)

        if self.jacobian is not None:
            J = self.jacobian(x_pred, self.popt)
            var_f = J @ self.pcov @ J.T
            sigma_f = np.sqrt(var_f)
        else:
            sigma_f = np.nan

        print(f"Predicted time for {x_pred} {self.x_param} = {y_pred:.2e} ± {sigma_f:.2e} hours")
        return y_pred, sigma_f

    def run(self, plot=True, predict=True):
        self.compute_stats()
        self.fit_model()
        self.evaluate_model()

        if plot:
            self.plot_results()

        if predict:
            self.predict()
