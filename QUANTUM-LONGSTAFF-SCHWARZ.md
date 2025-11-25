# Quantum Longstaff-Schwarz Method for Pricing American Options
This implementation uses **Qiskit** to demonstrate the core "atomic unit" of the Quantum Longstaff-Schwartz method: **using Quantum Amplitude Estimation (QAE) to estimate the continuation value.**

### The Logic

In classical Longstaff-Schwartz (LSM), you run thousands of Monte Carlo paths and use regression to find the continuation value ($E[\text{Payoff}_{t+1} | S_t]$).
In **Quantum LSM**, we replace the classical Monte Carlo average with **Quantum Amplitude Estimation**.

This code demonstrates a **hybrid approach**:

1.  **Classical Outer Loop:** Handles the time-stepping and the American exercise decision logic.
2.  **Quantum Inner Loop:** Uses QAE to calculate the expected value (Continuation Value) at a specific node, providing a quadratic speedup over classical sampling.

### Prerequisites

You will need to install Qiskit and Qiskit Finance:

```bash
pip install qiskit qiskit-finance qiskit-aer
```

### Python/Qiskit Implementation

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit_finance.circuit.library import LogNormalDistribution, EuropeanCallPricing
from qiskit.primitives import Sampler

def get_quantum_continuation_value(S_current, strike, time_to_maturity, volatility, risk_free_rate):
    """
    Uses Quantum Amplitude Estimation to estimate the expected payoff (continuation value).
    In a full QLSM, this replaces the regression step for specific grid points.
    
    For this demo, we model the transition from t to t+1 as a 
    European Option pricing problem (1-step lookahead).
    """
    
    # 1. Define Model Parameters for the next time step
    num_uncertainty_qubits = 3  # Higher = more precision in price distribution
    
    # Low and High bounds for the asset price distribution at t+1
    # (In a real scenario, these depend on S_current and volatility)
    low = S_current * np.exp(-3 * volatility * np.sqrt(time_to_maturity))
    high = S_current * np.exp(3 * volatility * np.sqrt(time_to_maturity))
    
    # 2. Construct the Quantum Circuit for the Asset Price Distribution
    # We use a Log-Normal distribution to model the asset price movement
    mu = (risk_free_rate - 0.5 * volatility**2) * time_to_maturity + np.log(S_current)
    sigma = volatility * np.sqrt(time_to_maturity)
    
    uncertainty_model = LogNormalDistribution(
        num_uncertainty_qubits, 
        mu=mu, 
        sigma=sigma**2, 
        bounds=(low, high)
    )

    # 3. Define the Payoff Function (The "Continuation Value")
    # In a full recursive QLSM, this would be the interpolated value function V(t+1).
    # Here, we use a standard Call payoff as a proxy for demonstration.
    european_call = EuropeanCallPricing(
        num_state_qubits=num_uncertainty_qubits,
        strike_price=strike,
        rescaling_factor=0.25,
        bounds=(low, high),
        uncertainty_model=uncertainty_model
    )

    # 4. Setup the Estimation Problem for QAE
    # This prepares the operator A that encodes the expectation value into amplitude
    problem = EstimationProblem(
        state_preparation=european_call,
        objective_qubits=[num_uncertainty_qubits]
    )

    # 5. Run Quantum Amplitude Estimation (Iterative)
    # This converges to the expected value quadratically faster than classical MC
    ae = IterativeAmplitudeEstimation(
        epsilon_target=0.01,  # Target precision
        alpha=0.05,           # Confidence level
        sampler=Sampler()
    )
    
    result = ae.estimate(problem)
    
    # Recover the dollar value from the scaled quantum result
    conf_int = result.confidence_interval_processed
    estimated_value = result.estimation_processed
    
    return estimated_value

# --- Bermudan/American Option Pricing Logic (Toy Example) ---

def price_american_option_hybrid(S0, K, T, r, sigma, steps=3):
    """
    A simplified Bermudan pricing loop.
    """
    dt = T / steps
    current_S = S0
    
    print(f"--- Pricing American Call Option (Hybrid Quantum) ---")
    print(f"Start Price: {S0}, Strike: {K}, Steps: {steps}\n")

    # Simulate a single path for demonstration (or a tree in full implementation)
    # We walk FORWARD to generate a path, then would walk BACKWARD to price.
    # For this simplified demo, we check the exercise decision at t=1 on a hypothetical path.
    
    # Let's assume at t=1, the stock price moved up to 105
    hypothetical_S_t1 = 105 
    time_remaining = T - dt
    
    # 1. Calculate Immediate Exercise Value
    exercise_value = max(hypothetical_S_t1 - K, 0)
    
    # 2. Calculate Continuation Value (The Quantum Step)
    # "What is the expected value of holding this option until t=2?"
    print(f"Calculating Quantum Continuation Value at S={hypothetical_S_t1}...")
    continuation_value = get_quantum_continuation_value(
        S_current=hypothetical_S_t1,
        strike=K,
        time_to_maturity=dt, # Looking ahead 1 step
        volatility=sigma,
        risk_free_rate=r
    )
    
    # Discount the future value back to now
    continuation_value_discounted = continuation_value * np.exp(-r * dt)

    print(f"Immediate Exercise Value: {exercise_value:.4f}")
    print(f"Continuation Value (Quantum): {continuation_value_discounted:.4f}")

    # 3. Decision
    if exercise_value > continuation_value_discounted:
        print(">> Decision: EXERCISE EARLY")
        return exercise_value
    else:
        print(">> Decision: HOLD")
        return continuation_value_discounted

# --- Execution ---
if __name__ == "__main__":
    # Parameters
    spot_price = 100
    strike_price = 100
    expiry = 1.0
    risk_free = 0.05
    vol = 0.2
    
    price_american_option_hybrid(spot_price, strike_price, expiry, risk_free, vol)
```

### Key Technical Details

1.  **State Preparation (`LogNormalDistribution`)**: This circuit loads the probability distribution of the asset prices into the quantum state amplitudes. In a classical computer, this is an array of probabilities; on a quantum computer, it's a superposition.
2.  **Payoff Operator (`EuropeanCallPricing`)**: This performs a "Quantum Comparator". It uses ancilla qubits to flip a target qubit if the encoded asset price state $|S\rangle$ is greater than the strike $|K\rangle$. The rotation angle of the target qubit encodes the *value* of the payoff.
3.  **Amplitude Estimation**: This is the engine. Instead of measuring the circuit 10,000 times (Shot noise $\sim 1/\sqrt{N}$), it uses Grover-like reflections to amplify the amplitude of the "correct" answer, converging with error $\sim 1/N$.

### Challenges in Real-World Scale

  * **Loading Arbitrary Functions:** The code above assumes the continuation value is a simple Call payoff. In the middle of an American option simulation, the continuation value is a complex polynomial derived from regression. Loading that arbitrary polynomial into a quantum circuit (Step 3) is the hardest current challenge in Quantum Finance, often requiring "Quantum Signal Processing" or "Quantum Generative Adversarial Networks" (qGANs).

... [Quantum Monte Carlo for Option Pricing](https://www.youtube.com/watch?v=z8WGPbc66Jk)

This video walks through the specific Qiskit components used in the code above, visually explaining how the Log-Normal distribution and Payload circuits are constructed for option pricing.