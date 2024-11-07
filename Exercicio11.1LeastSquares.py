# Least squares solving
# y = m x + b
# y0 = m x0 + b
# y1 = m x1 + b
# y2 = m x2 + b
# y3 = m x3 + b
# y4 = m x4 + b
# equivalent to A p = y
# with
# A = [x0 1
#      x1 1
#      x2 1
#      x3 1
#      x4 1]
#  p = [m
#       b]

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

N = 100
x = np.array(range(N))
m = 0.5
b = 1

y = m * x + b
rng = default_rng(1)

noise_y = rng.standard_normal(N) * 10
y_with_noise = y + noise_y

A = np.array([x, np.ones(N)]).T

solution = np.linalg.lstsq(a=A, b=y_with_noise)
solution_gt = np.linalg.lstsq(a=A, b=y)

m_est, b_est = solution[0]
m_gt, b_gt = solution_gt[0]

y_est = x * m_est + b_est
y_gt = x * m_gt + b_gt

print("modelo real m=", str(m), "; b=", str(b))
print("modelo estimado sem ruido m_gt=", str(m_gt), "; b_gt=", str(b_gt))
print("modelo estimado com ruido m_est=", str(m_est), "; b_est=", str(b_est))

plt.plot(x, y, 'k', label="Original Data")
plt.plot(x, y_gt, 'b', label="Fitted Model (no noise)")
plt.plot(x, y_with_noise, 'rx', label="Noisy data")
plt.plot(x, y_est, 'g', label="Fitted Model (with noise)")
plt.legend()
plt.show()

