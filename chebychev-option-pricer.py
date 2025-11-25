import numpy as np
from scipy.stats import norm

class ChebyshevInterpolator:
    """
    Handles the mathematical machinery of Chebyshev approximations.
    """
    def __init__(self, degree, lower_bound, upper_bound):
        self.degree = degree
        self.a = lower_bound
        self.b = upper_bound
        
        # Pre-compute Chebyshev nodes (roots of T_n) in the canonical domain [-1, 1]
        # and map them to the physical domain [a, b]
        k = np.arange(1, degree + 1)
        self.nodes_canonical = np.cos((2 * k - 1) * np.pi / (2 * degree))
        self.nodes_physical = self._to_physical(self.nodes_canonical)

    def _to_physical(self, x_canonical):
        """Maps [-1, 1] -> [a, b]"""
        return 0.5 * (self.a + self.b) + 0.5 * (self.b - self.a) * x_canonical

    def _to_canonical(self, x_physical):
        """Maps [a, b] -> [-1, 1]"""
        # Clip to ensure stability if query points are slightly out of bounds
        x_clipped = np.clip(x_physical, self.a, self.b)
        return (2 * x_clipped - (self.a + self.b)) / (self.b - self.a)

    def evaluate_polynomials(self, x_canonical):
        """
        Evaluates Chebyshev polynomials T_0(x) ... T_n(x) at a given point.
        Returns array of shape (degree,).
        """
        T = np.zeros(self.degree)
        T[0] = 1.0
        if self.degree > 1:
            T[1] = x_canonical
        for i in range(2, self.degree):
            T[i] = 2 * x_canonical * T[i-1] - T[i-2]
        return T

    def fit_coefficients(self, y_values):
        """
        Computes Chebyshev coefficients c_j given function values at the nodes.
        Formula: c_j = (2/N) * sum(y_k * T_j(x_k))
        """
        coeffs = np.zeros(self.degree)
        for j in range(self.degree):
            # T_j evaluated at all nodes
            T_j_nodes = np.cos(j * np.arccos(self.nodes_canonical))
            if j == 0:
                coeffs[j] = (1 / self.degree) * np.sum(y_values)
            else:
                coeffs[j] = (2 / self.degree) * np.sum(y_values * T_j_nodes)
        return coeffs

    def interpolate(self, x_query, coeffs):
        """
        Approximates f(x_query) using the computed coefficients.
        """
        x_can = self._to_canonical(x_query)
        # Evaluate T_j(x) recursively
        val = 0.0
        T_prev2 = 1.0
        T_prev1 = x_can
        
        val += coeffs[0] * T_prev2
        if self.degree > 1:
            val += coeffs[1] * T_prev1
            
        for j in range(2, self.degree):
            T_curr = 2 * x_can * T_prev1 - T_prev2
            val += coeffs[j] * T_curr
            T_prev2 = T_prev1
            T_prev1 = T_curr
            
        return val


class QuantumChebyshevPricer:
    """
    Prices American Options using Backward Induction + Chebyshev Interpolation.
    Designed to mimic the structure of Quantum Algorithms (Miyamoto et al.).
    """
    def __init__(self, S0, K, T, r, sigma, poly_degree=20, time_steps=50):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.dt = T / time_steps
        self.steps = time_steps
        
        # Domain truncation for the asset price (usually +/- 4 standard deviations)
        # This is critical for fixed-domain approximations
        std_dev = sigma * np.sqrt(T)
        self.S_min = max(0.01, S0 * np.exp((r - 0.5*sigma**2)*T - 4*std_dev))
        self.S_max = S0 * np.exp((r - 0.5*sigma**2)*T + 4*std_dev)
        
        self.interpolator = ChebyshevInterpolator(poly_degree, self.S_min, self.S_max)

    def payoff(self, S):
        """Put Option Payoff"""
        return np.maximum(self.K - S, 0.0)

    def quantum_amplitude_estimation_mock(self, S_start, next_step_coeffs):
        """
        MOCK FUNCTION: Simulates the Quantum Amplitude Estimation step.
        
        In a real Quantum Computer:
        1. We construct a circuit U that loads the transition probability P(S_t+1 | S_t).
        2. We construct a circuit V that computes the value function using the 'next_step_coeffs'.
        3. We use QAE to estimate the integral E[Value(S_t+1)].
        
        Here: We compute the integral using accurate Gaussian Quadrature or exact formula 
        to simulate the result of a perfect quantum estimation.
        """
        # Classical simulation of the expectation:
        # We need E[ V_{t+1}(S_{t+1}) | S_t = S_start ]
        
        # 1. Determine distribution of S_{t+1} given S_start
        # S_{t+1} = S_start * exp( (r - 0.5*sigma^2)dt + sigma*sqrt(dt)*Z )
        # We approximate the expectation by sampling (Monte Carlo) or simple integration.
        # For speed and precision in this demo, we use a mini Monte Carlo of 1000 paths
        # (simulating the quantum measurement samples)
        
        Z = np.random.normal(0, 1, 5000)
        S_next = S_start * np.exp((self.r - 0.5 * self.sigma**2) * self.dt + 
                                  self.sigma * np.sqrt(self.dt) * Z)
        
        # 2. Evaluate the Value Function at t+1 using our Chebyshev approximation
        # (This avoids storing the whole tree/grid)
        V_next = self.interpolator.interpolate(S_next, next_step_coeffs)
        
        # Discount back to t
        continuation_value = np.exp(-self.r * self.dt) * np.mean(V_next)
        
        return continuation_value

    def price(self):
        print(f"Starting Pricing on Domain [{self.S_min:.2f}, {self.S_max:.2f}]")
        
        # 1. Initialize Value Function at Maturity T
        # Calculate payoff at all Chebyshev nodes
        nodes = self.interpolator.nodes_physical
        values_at_nodes = self.payoff(nodes)
        
        # Get initial coefficients for V(S, T)
        coeffs = self.interpolator.fit_coefficients(values_at_nodes)
        
        # 2. Backward Induction
        for t_idx in range(self.steps - 1, -1, -1):
            current_values_at_nodes = np.zeros_like(nodes)
            
            # For each Chebyshev node (asset price state), calculate Option Value
            for i, S_node in enumerate(nodes):
                
                # A. Immediate Exercise Value
                exercise_val = self.payoff(S_node)
                
                # B. Continuation Value (The Quantum Step)
                # We use the coefficients from the *previous* step (t+1) to estimate expectation
                cont_val = self.quantum_amplitude_estimation_mock(S_node, coeffs)
                
                # C. Maximize (American condition)
                current_values_at_nodes[i] = np.max([exercise_val, cont_val])
            
            # D. Update coefficients for this time step t
            coeffs = self.interpolator.fit_coefficients(current_values_at_nodes)
            
            if t_idx % 10 == 0:
                print(f"Step {t_idx}/{self.steps} complete. Approx Value at Spot: {self.interpolator.interpolate(self.S0, coeffs):.4f}")

        # 3. Final Interpolation at S0
        result = self.interpolator.interpolate(self.S0, coeffs)
        return result

# --- Main Execution ---
if __name__ == "__main__":
    # Parameters for American Put Option
    S0 = 100.0    # Spot Price
    K = 100.0     # Strike Price
    T = 1.0       # Time to Maturity (1 year)
    r = 0.05      # Risk-free rate
    sigma = 0.2   # Volatility
    
    # Run Pricer
    pricer = QuantumChebyshevPricer(S0, K, T, r, sigma, poly_degree=16, time_steps=50)
    price = pricer.price()
    
    print("\n" + "="*40)
    print(f"Final American Option Price: ${price:.4f}")
    print("="*40)