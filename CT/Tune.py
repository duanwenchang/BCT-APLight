import numpy as np
from scipy.stats import gaussian_kde, norm

class Tune:
    def __init__(self, history_Q, current_Q):
        self.history_Q = history_Q 
        self.current_Q = current_Q 
        self.min_posterior_risks = []

    def kde_prior(self, Q_values):
       kde = gaussian_kde(Q_values, bw_method='silverman')
       return kde  

    def likelihood_history(self, Q, Q_hist, phase_idx):
        kde = gaussian_kde(Q_hist[phase_idx])
        h = kde.factor
        likelihood = 1.0

        for q_i in Q_hist[phase_idx]:
            gaussian_value = (1 / np.sqrt(2 * np.pi * h**2)) * np.exp(-((Q - q_i)**2) / (2 * h**2))
            likelihood *= gaussian_value
        return likelihood

    def likelihood_current(self, Q, Q_current, sigma_history_idx, k, phase_idx):
        sigma_current_idx = sigma_history_idx / k  
        return (1 / np.sqrt(2 * np.pi * sigma_current_idx**2)) * np.exp(-((Q - Q_current[phase_idx])**2) / (2 * sigma_current_idx**2))

    def posterior_distribution(self, Q, Q_current, Q_hist, k, phase_idx):
        sigma_history = np.std(Q_hist[phase_idx])
        prior_kde = self.kde_prior(Q_hist[phase_idx])
        prior_distribution = prior_kde(Q)  

        likelihood_current = self.likelihood_current(Q, Q_current, sigma_history, k, phase_idx)
        likelihood_history = self.likelihood_history(Q, Q_hist, phase_idx)

        posterior = likelihood_current * likelihood_history * prior_distribution
        return posterior

    def expected_posterior_risk(self, Q_current, Q_hist, phase_idx, k):
        risk = 0.0
        Q_phase = list(np.linspace(min(Q_hist[phase_idx]), max(Q_hist[phase_idx]), 1000))
        for i in range(len(Q_phase) - 1):
            Q_i = Q_phase[i]
            posterior_i = self.posterior_distribution(Q_i, Q_current, Q_hist, k, phase_idx)  
            delta_Q = Q_phase[i + 1] - Q_phase[i]
            loss = (Q_i - Q_current[phase_idx])**2
            risk += loss * posterior_i * delta_Q
        return risk
    

    def tune_decision(self):
        Q_hist = self.history_Q
        Q_current = self.current_Q
        risks = []
        for phase_idx in range(len(Q_hist)):
            k = (len(Q_hist[phase_idx]))**(1/2)
            risk = self.expected_posterior_risk(Q_current, Q_hist, phase_idx, k)
            risks.append(risk)
        min_risk_index = np.argmin(risks) 
        return min_risk_index 

