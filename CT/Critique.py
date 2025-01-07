import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import truncnorm, laplace, invgamma, norm
import warnings
from pmdarima import auto_arima

# 忽略特定的警告
# warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

class Critique:
    def __init__(self, data, freq='D'):
        self.data = data
        self.d = None
        self.p = None
        self.q = None
        self.P = None
        self.D = None
        self.Q = None
        self.mu_c = None
        self.sigma_c = None
        self.c_lower = None
        self.c_upper = None
        self.mu_phi = []
        self.b_phi = []
        self.mu_theta = []
        self.b_theta = []
        self.a_sigma2 = None
        self.b_sigma2 = None
        self.trace = []
        self.freq = freq
        self.sarima_params = None

    def select_a_sigma2(self, d):
        if 10 <= d < 50:
            a_sigma2 = 5
        elif 50 <= d < 150:
            a_sigma2 = 3
        elif 150 <= d < 300:
            a_sigma2 = 2
        elif 300 <= d < 500:
            a_sigma2 = 1.5
        elif d > 500:
            a_sigma2 = 1.2
        else:
            a_sigma2 = 10
        return a_sigma2

    def order_selection(self):
        print("Starting order selection...")
        model = auto_arima(
                self.data, 
                seasonal=True, 
                m=10,
                stepwise=True,
                suppress_warnings=True, 
                trace=False, 
                error_action='ignore',
                with_intercept=True  
            )
        self.p, self.d, self.q = model.order
        self.P, self.D, self.Q, _ = model.seasonal_order
        # print(f"Selected order: (p, d, q) = {model.order}")
        # print(f"(P, D, Q, m) = {model.seasonal_order}")

        sarima_model = SARIMAX(self.data, order=(self.p, self.d, self.q), seasonal_order=(self.P, self.D, self.Q, 10))
        model_fit = sarima_model.fit(disp=False)
        # print("\nModel Summary:\n")
        # print(model_fit.summary())
        params = model_fit.params

        const = params.get('intercept', 0)
        ar_params = [params[name] for name in params.keys() if 'ar.L' in name]
        ma_params = [params[name] for name in params.keys() if 'ma.L' in name]
        seasonal_ar_params = [params[name] for name in params.keys() if 'ar.S' in name]
        seasonal_ma_params = [params[name] for name in params.keys() if 'ma.S' in name]
        sigma2 = params['sigma2']

        # print(f"Intercept: {const}")
        # print(f"AR Parameters: {ar_params}")
        # print(f"MA Parameters: {ma_params}")
        # print(f"Seasonal AR Parameters: {seasonal_ar_params}")
        # print(f"Seasonal MA Parameters: {seasonal_ma_params}")
        # print(f"Sigma^2: {sigma2}")

        self.sarima_params = {
            'const': const,
            'ar_params': ar_params,
            'ma_params': ma_params,
            'seasonal_ar_params': seasonal_ar_params,
            'seasonal_ma_params': seasonal_ma_params,
            'sigma2': sigma2
        }

        return model_fit

    def construct_and_sample_prior(self, sarima_c, sarima_phi, sarima_theta, sarima_sigma2, num_samples):
        self.mu_c = sarima_c
        self.sigma_c = abs(sarima_c * 2 / 1.96)
        self.c_lower = self.mu_c - 1.96 * self.sigma_c
        self.c_upper = self.mu_c + 1.96 * self.sigma_c

        self.mu_phi = np.array([sarima_phi[i] for i in range(self.p)])
        self.b_phi = np.array([abs(sarima_phi[i] * 2 / 2.96) for i in range(self.p)])
        self.phi_lower = self.mu_phi - 2.96 * self.b_phi
        self.phi_upper = self.mu_phi + 2.96 * self.b_phi

        self.mu_theta = np.array([sarima_theta[i] for i in range(self.q)])
        self.b_theta = np.array([abs(sarima_theta[i] * 2 / 2.96) for i in range(self.q)])
        self.theta_lower = self.mu_theta - 2.96 * self.b_theta
        self.theta_upper = self.mu_theta + 2.96 * self.b_theta

        self.a_sigma2 = self.select_a_sigma2(sarima_sigma2)
        self.b_sigma2 = sarima_sigma2 / (self.a_sigma2 + 1)

        self.sigma2_upper = 3 * sarima_sigma2

        if self.mu_c != 0:
            c_samples = truncnorm.rvs(
                (self.c_lower - self.mu_c) / self.sigma_c, 
                (self.c_upper - self.mu_c) / self.sigma_c, 
                loc=self.mu_c, 
                scale=self.sigma_c,
                size=num_samples
            )
        else:
            c_samples = np.zeros(num_samples)  
        phi_samples = np.array([laplace.rvs(loc=self.mu_phi[i], scale=self.b_phi[i], size=num_samples) for i in range(self.p)]).T
        theta_samples = np.array([laplace.rvs(loc=self.mu_theta[i], scale=self.b_theta[i], size=num_samples) for i in range(self.q)]).T
        sigma2_samples = invgamma.rvs(self.a_sigma2, scale=self.b_sigma2, size=num_samples)

        valid_indices = np.ones(num_samples, dtype=bool) 
        if self.p > 0:
            valid_indices &= np.all((phi_samples >= self.phi_lower) & (phi_samples <= self.phi_upper), axis=1)
        if self.q > 0:
            valid_indices &= np.all((theta_samples >= self.theta_lower) & (theta_samples <= self.theta_upper), axis=1)
        valid_indices &= (sigma2_samples >= 0) & (sigma2_samples < self.sigma2_upper)

        c_samples = c_samples[valid_indices]
        if self.p > 0:
            phi_samples = phi_samples[valid_indices]
        if self.q > 0:
            theta_samples = theta_samples[valid_indices]
        sigma2_samples = sigma2_samples[valid_indices]

        return c_samples, phi_samples, theta_samples, sigma2_samples

    def forecast_future(self, h):
        print("Starting forecast...")
        
        sarima_c = self.sarima_params['const']
        sarima_phi = self.sarima_params['ar_params']
        sarima_theta = self.sarima_params['ma_params']
        seasonal_ar_params = self.sarima_params['seasonal_ar_params']
        seasonal_ma_params = self.sarima_params['seasonal_ma_params']
        sarima_sigma2 = self.sarima_params['sigma2']

        c_samples, phi_samples, theta_samples, sigma2_samples = self.construct_and_sample_prior(
            sarima_c, sarima_phi, sarima_theta, sarima_sigma2, 100
        )

        future_predictions = []
        num_valid_samples = len(c_samples)  
        for i in range(num_valid_samples):
            combined_c = 0.9 * sarima_c + 0.1 * c_samples[i]
            combined_phi = 0.9 * np.array(sarima_phi) + 0.1 * phi_samples[i] if sarima_phi else np.array([])
            combined_theta = 0.9 * np.array(sarima_theta) + 0.1 * theta_samples[i] if sarima_theta else np.array([])
            combined_seasonal_phi = np.array(seasonal_ar_params)
            combined_seasonal_theta = np.array(seasonal_ma_params)
            combined_sigma2 = 0.9 * sarima_sigma2 + 0.1 * sigma2_samples[i]

            model = SARIMAX(
                self.data,
                order=(self.p, self.d, self.q),
                seasonal_order=(self.P, self.D, self.Q, 10),
            )
            
            if sarima_c != 0:
                params = np.concatenate(([combined_c], combined_phi, combined_theta, combined_seasonal_phi, combined_seasonal_theta, [combined_sigma2]))
            else:
                params = np.concatenate((combined_phi, combined_theta, combined_seasonal_phi, combined_seasonal_theta, [combined_sigma2]))
            
            model_fit = model.filter(params)
            forecast = model_fit.forecast(steps=h)
            future_predictions.append(forecast)

        future_predictions = np.array(future_predictions)

        future_interval = [(np.percentile(future_predictions[:, t], 2.5)-5, np.percentile(future_predictions[:, t], 97.5)+5) for t in range(h)]
        means = [np.mean(self.data[i::10]) for i in range(10)]
        bayes_intervals = []
        use_recent_intervals = any(lower < -20 or upper > 10 for lower, upper in future_interval)

        if use_recent_intervals:
            recent_intervals = [(mean - 8, mean + 8) for mean in means]
            bayes_intervals = recent_intervals
        else:
            recent_intervals = [(mean - 15, mean + 15) for mean in means]
            for (ci_lower, ci_upper), (recent_lower, recent_upper) in zip(future_interval, recent_intervals):
                bayes_lower = min(ci_lower, recent_lower)
                bayes_upper = max(ci_upper, recent_upper)
                bayes_intervals.append((bayes_lower, bayes_upper))

        return bayes_intervals
