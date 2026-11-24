# Experiment 4: Support Vector Machine (SVM)

## Aim

To implement and analyze the performance of a Support Vector Machine
(SVM) model for classification using Python.

## Overview

A Support Vector Machine is a supervised machine-learning algorithm used
for classification. It finds a decision boundary (hyperplane) that
separates classes while maximizing the margin between the closest data
points, called **support vectors**.

This experiment uses the **Iris dataset** and compares two SVM kernels:

-   **Linear kernel** -- suitable when classes can be separated
    approximately by a straight decision boundary.
-   **RBF (Radial Basis Function) kernel** -- can model non-linear
    decision boundaries.

For visualization, the experiment uses the first two features of the
Iris dataset.

## Algorithm

1.  **Start.**
2.  Load the Iris dataset and select the first two features for
    visualization.
3.  Split the data into training and testing sets.
4.  Standardize the features using `StandardScaler`.
5.  Train an SVM classifier with a linear kernel.
6.  Predict the test labels and calculate the classification accuracy.
7.  Train another SVM classifier with the RBF kernel.
8.  Predict the test labels and calculate the classification accuracy.
9.  Plot the decision boundaries for both models.
10. **Stop.**

## Program

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Step 2: Load Iris and select the first two features
iris = load_iris()
X = iris.data[:, :2]
y = iris.target
print("Features used:", iris.feature_names[:2])

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Step 4: Standardize features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Steps 5-6: Linear kernel
linear_svm = SVC(kernel="linear", C=1.0)
linear_svm.fit(X_train_s, y_train)
acc_linear = accuracy_score(y_test, linear_svm.predict(X_test_s))
print("Linear kernel accuracy:", round(acc_linear, 4))

# Steps 7-8: RBF kernel
rbf_svm = SVC(kernel="rbf", C=1.0, gamma="scale")
rbf_svm.fit(X_train_s, y_train)
acc_rbf = accuracy_score(y_test, rbf_svm.predict(X_test_s))
print("RBF kernel accuracy:", round(acc_rbf, 4))

print("Support vectors (linear):", int(linear_svm.n_support_.sum()))
print("Support vectors (RBF):", int(rbf_svm.n_support_.sum()))

# Step 9: Plot decision boundaries
def plot_boundary(model, X, y, title, ax):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Sepal length (standardized)")
    ax.set_ylabel("Sepal width (standardized)")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_boundary(linear_svm, X_train_s, y_train,
              f"Linear Kernel (Test Acc = {acc_linear:.2f})", axes[0])
plot_boundary(rbf_svm, X_train_s, y_train,
              f"RBF Kernel (Test Acc = {acc_rbf:.2f})", axes[1])
plt.tight_layout()
plt.savefig("svm_decision_boundaries.png", dpi=120, bbox_inches="tight")
plt.show()
```

## Output

```
Features used: ['sepal length (cm)', 'sepal width (cm)']
Linear kernel accuracy: 0.6889
RBF kernel accuracy: 0.6889
Support vectors (linear): 51
Support vectors (RBF): 57
```

**Decision boundaries (Linear vs RBF kernel)**

<img width="1428" height="588" alt="image" src="https://github.com/user-attachments/assets/92f307a8-f76c-40e2-99c0-d7612ced8264" />


## Result

The program was executed successfully. SVM classifiers with a linear
kernel and an RBF kernel were trained on the first two Iris features
(sepal length and sepal width) and tested on 45 unseen samples.

-   Both kernels achieved a test accuracy of **68.89%**.
-   The linear kernel produced straight decision boundaries and used
    51 support vectors, while the RBF kernel produced curved boundaries
    and used 57 support vectors.
-   *Setosa* is clearly separated from the other two classes, but
    *Versicolor* and *Virginica* overlap heavily in sepal length and
    width, so neither kernel can separate them well. The accuracy is
    limited by the features used, not by the kernel.
-   Using all four features (including petal length and width) would
    raise the accuracy considerably, but then the decision boundary
    could not be plotted in 2D.

Thus, SVM with linear and RBF kernels was implemented and analyzed on
the Iris dataset, and the decision boundaries were visualized.
