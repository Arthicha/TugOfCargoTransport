import numpy as np
from scipy.signal import savgol_filter

class RBFNetwork:
	"""Radial Basis Function Network with Gaussian Kernels, Learnable Centers, Bias, and Weight GD."""
	def __init__(self, n_centers=20, sigma=1.0, lr_weights=0.01, lr_centers=0.01, lr_bias=0.01):
		self.n_centers = n_centers
		self.sigma = sigma
		self.lr_weights = lr_weights
		self.lr_centers = lr_centers
		self.lr_bias = lr_bias
		
		self.centers = None
		self.weights = None
		self.bias = None

	def rbf(self, X):
		"""Compute the Gaussian RBF kernel activation matrix."""
		# dist_sq shape: (N, k)
		dist_sq = np.sum((X[:, np.newaxis, :] - self.centers[np.newaxis, :, :]) ** 2, axis=2)
		return np.exp(-dist_sq / (2 * (self.sigma ** 2)))


	def initialize_centers(self, X, Y, window_length=15, polyorder=2):
		"""
		Initialize centers by applying a Savitzky-Golay filter to smooth Y,
		finding local minima and maxima across the target spectrum,
		and placing centers at the corresponding X features.
		"""
		if Y.ndim > 1:
			# If multi-output, average across outputs to find extrema profile
			target = Y.mean(axis=1)
		else:
			target = Y.squeeze()

		# 1. Apply Savitzky-Golay filter to smooth out noise in Y
		n_samples = len(X)
		
		# Ensure window_length is odd and <= n_samples
		window = min(window_length, n_samples if n_samples % 2 != 0 else n_samples - 1)
		if window <= polyorder:
			window = polyorder + 2 if (polyorder + 2) % 2 != 0 else polyorder + 3

		target_smoothed = savgol_filter(target, window_length=window, polyorder=polyorder)

		# 2. Sort indices based on smoothed target Y values
		sorted_indices = np.argsort(target_smoothed)

		# 3. Pick indices spanning from lowest (minima) to highest (maxima)
		k = min(self.n_centers, n_samples)
		extrema_indices = np.linspace(0, n_samples - 1, num=k, dtype=int)
		chosen_indices = sorted_indices[extrema_indices]

		# 4. Assign centers to the corresponding X inputs
		self.centers = X[chosen_indices].copy()

	def initialize_weights(self, output_dim):
		"""Initialize weights with small random Gaussian values."""
		self.weights = np.random.randn(self.n_centers, output_dim) * 0.1

	def update_weights(self, psi, dL_dYhat):
		"""Update weights using Gradient Descent: dL/dW = Psi^T @ dL/dY_hat."""
		grad_weights = psi.T @ dL_dYhat  # (k, output_dim)
		self.weights -= self.lr_weights * grad_weights

	def predict(self, X):
		"""Predict target output for new inputs: Y_hat = Psi @ W + bias."""
		psi = self.rbf(X)
		return (psi @ self.weights) #+ self.bias

	def backward_psi(self, grad_output):
		"""Compute dL/dPsi given dL/dY_hat. Shape: (N, k)"""
		grad_out = np.atleast_2d(grad_output)
		weights = self.weights
		if weights.ndim == 1:
			weights = weights[:, np.newaxis]

		return grad_out @ weights.T

	def backward_phi(self, phi, grad_output):
		"""
		Compute gradient of Loss w.r.t input angles phi of shape (N,).
		Where X = [sin(phi), cos(phi)].
		"""
		phi = np.asarray(phi).flatten()
		X = np.column_stack([np.sin(phi), np.cos(phi)])
		
		# 1. Forward activations Psi (N, k)
		psi = self.rbf(X)
		
		# 2. Downstream gradient dL/dPsi (N, k)
		dL_dpsi = self.backward_psi(grad_output)
		
		# 3. Extract center coordinates: C_x (k,), C_y (k,)
		C_x = self.centers[:, 0]
		C_y = self.centers[:, 1]
		
		# 4. Trigonometric terms for shape broadcast: (N, 1)
		cos_phi = np.cos(phi)[:, np.newaxis]
		sin_phi = np.sin(phi)[:, np.newaxis]
		
		# 5. Compute dPsi / dphi: shape (N, k)
		dpsi_dphi = (psi / (self.sigma ** 2)) * (C_x * cos_phi - C_y * sin_phi)
		
		# 6. Chain rule: sum over all k centers (N,)
		grad_phi = np.sum(dL_dpsi * dpsi_dphi, axis=1)
		
		return grad_phi

	def backward_centers(self, X, psi, dL_dpsi):
		"""Compute gradient of Loss w.r.t center locations C."""
		A = dL_dpsi * psi  # (N, k)
		
		# Difference vector (X_n - C_k): shape (N, k, D)
		diff = X[:, np.newaxis, :] - self.centers[np.newaxis, :, :]
		
		# Sum over N samples: shape (k, D)
		grad_centers = np.sum(A[:, :, np.newaxis] * diff, axis=0) / (self.sigma ** 2)
		return grad_centers

	def fit(self, X, Y, epochs=100, verbose=False):
		"""
		Train the RBF Network updating weights, centers, and bias via Gradient Descent.
		"""
		if Y.ndim == 1:
			Y = Y[:, np.newaxis]

		output_dim = Y.shape[1]

		if self.centers is None:
			self.initialize_centers(X[:140], Y[:140])

		if self.weights is None:
			self.initialize_weights(output_dim)

		if self.bias is None:
			self.bias = np.zeros((1, output_dim))

		for epoch in range(epochs):

			# 1. Forward pass activations and predictions
			psi = self.rbf(X)
			Y_hat = (psi @ self.weights) + self.bias
			
			# 2. Compute loss gradient: dL/dY_hat = (Y_hat - Y) / N
			dL_dYhat = (Y_hat - Y) / len(X)

			self.update_weights(psi, dL_dYhat)
			
			
			# 3. Downstream gradient dL/dPsi
			dL_dpsi = self.backward_psi(dL_dYhat)
			
			# 4. Compute gradients
			grad_centers = self.backward_centers(X, psi, dL_dpsi)
			grad_bias = np.sum(dL_dYhat, axis=0, keepdims=True)  # (1, output_dim)
			
			# 5. Gradient Descent updates
			self.centers -= self.lr_centers * grad_centers

			#self.bias -= self.lr_bias * grad_bias
			
			self.bias *= 0.0
			
			print(epoch)
			#if verbose and (epoch % max(1, (epochs // 10)) == 0 or epoch == epochs - 1):
				#mse = np.mean((Y_hat - Y) ** 2)
				#print(f"Epoch {epoch:4d}/{epochs} - MSE: {mse:.6f}")

	def save(self, filepath):
		"""Save model parameters and hyperparameters to a .npz file."""
		np.savez(
			filepath,
			centers=self.centers,
			weights=self.weights,
			bias=self.bias,
			hyperparameters=np.array([self.n_centers, self.sigma, self.lr_weights, self.lr_centers, self.lr_bias])
		)
		print(f"Model successfully saved to '{filepath}'")

def load_model(filepath):
	"""Load an RBFNetwork instance from a .npz file."""
	data = np.load(filepath)
	hp = data['hyperparameters']
	
	model = RBFNetwork(
		n_centers=int(hp[0]),
		sigma=float(hp[1]),
		lr_weights=float(hp[2]),
		lr_centers=float(hp[3]),
		lr_bias=float(hp[4])
	)
	
	model.centers = data['centers']
	model.weights = data['weights']
	model.bias = data['bias']
	
	print(f"Model successfully loaded from '{filepath}'")
	return model