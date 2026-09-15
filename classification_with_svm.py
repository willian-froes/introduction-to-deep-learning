import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("assets/Heart.csv", index_col=0)
cleaned_dataset = dataset.dropna()

y = cleaned_dataset['AHD']
X = cleaned_dataset.drop(columns=['AHD']).select_dtypes(include=np.number)
X = X.loc[:, X.nunique() > 5]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

feature_selector = SelectKBest(score_func=f_classif, k=2)

X_train_selected = feature_selector.fit_transform(X_train, y_train)
X_test_selected = feature_selector.transform(X_test)

selected_features = X_train.columns[feature_selector.get_support()]

print("\nSelected features:", list(selected_features))

feature_x = selected_features[0]
feature_y = selected_features[1]

X_train = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
X_test = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)

no_ahd_train = X_train[y_train == "No"]
yes_ahd_train = X_train[y_train == "Yes"]
no_ahd_test = X_test[y_test == "No"]
yes_ahd_test = X_test[y_test == "Yes"]

fig, axes = plt.subplots(2, 4, figsize=(15, 8))

fig.canvas.manager.set_window_title("Heart Desease SVC")

axes[0][0].scatter(no_ahd_train[feature_x], no_ahd_train[feature_y], label="No AHD")
axes[0][0].scatter(yes_ahd_train[feature_x], yes_ahd_train[feature_y], label="Yes AHD")
axes[0][0].set_title("Train splited data")
axes[0][0].set_xlabel(feature_x)
axes[0][0].set_ylabel(feature_y)

axes[1][0].scatter(no_ahd_test[feature_x], no_ahd_test[feature_y], label="No AHD")
axes[1][0].scatter(yes_ahd_test[feature_x], yes_ahd_test[feature_y], label="Yes AHD")
axes[1][0].set_title("Test splited data")
axes[1][0].set_xlabel(feature_x)
axes[1][0].set_ylabel(feature_y)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

no_ahd_train_scaled = X_train_scaled[y_train == "No"]
yes_ahd_train_scaled = X_train_scaled[y_train == "Yes"]

linear_grid = GridSearchCV(
    SVC(kernel="linear", random_state=42),
    {
        "C": [0.01, 0.1, 1, 10, 100]
    },
    cv=5,
    scoring="accuracy"
)

polynomial_grid = GridSearchCV(
    SVC(kernel="poly", random_state=42),
    {
        "C": [0.1, 1, 10],
        "degree": [2, 3, 4],
        "gamma": ["scale", "auto"]
    },
    cv=5,
    scoring="accuracy"
)

rbf_grid = GridSearchCV(
    SVC(kernel="rbf", random_state=42),
    {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.01, 0.1, 1]
    },
    cv=5,
    scoring="accuracy"
)

linear_grid.fit(X_train_scaled, y_train)
polynomial_grid.fit(X_train_scaled, y_train)
rbf_grid.fit(X_train_scaled, y_train)

linear_model = linear_grid.best_estimator_
y_linear_predict = linear_model.predict(X_test_scaled)
linear_accuracy = accuracy_score(y_test, y_linear_predict)

polynomial_model = polynomial_grid.best_estimator_
y_polynomial_predict = polynomial_model.predict(X_test_scaled)
polynomial_accuracy = accuracy_score(y_test, y_polynomial_predict)

rbf_model = rbf_grid.best_estimator_
y_rbf_predict = rbf_model.predict(X_test_scaled)
rbf_accuracy = accuracy_score(y_test, y_rbf_predict)

x = np.linspace(X_train_scaled[:, 0].min(), X_train_scaled[:, 0].max(), 100)
y = np.linspace(X_train_scaled[:, 1].min(), X_train_scaled[:, 1].max(), 100)

xx, yy = np.meshgrid(x, y)
grid = np.c_[xx.ravel(), yy.ravel()]

D_linear = linear_model.decision_function(grid).reshape(xx.shape)
D_polynomial = polynomial_model.decision_function(grid).reshape(xx.shape)
D_rbf = rbf_model.decision_function(grid).reshape(xx.shape)

axes[0][1].scatter(no_ahd_train_scaled[:, 0], no_ahd_train_scaled[:, 1], label="No AHD")
axes[0][1].scatter(yes_ahd_train_scaled[:, 0], yes_ahd_train_scaled[:, 1], label="Yes AHD")
axes[0][1].scatter(linear_model.support_vectors_[:, 0], linear_model.support_vectors_[:, 1], label="Support Vectors", facecolors="none", edgecolors="black", s=100)
axes[0][1].contourf(xx, yy, D_linear > 0, alpha=0.3)
axes[0][1].contour(xx, yy, D_linear, levels=[-1, 0, 1], colors="black", linestyles=["--", "-", "--"], linewidths=[1, 2, 1])
axes[0][1].set_title("Linear SVC - Train")
axes[0][1].set_ylabel(f"{feature_y} (scaled)")
axes[0][1].set_xlabel(f"{feature_x} (scaled)")

axes[0][2].scatter(no_ahd_train_scaled[:, 0], no_ahd_train_scaled[:, 1], label="No AHD")
axes[0][2].scatter(yes_ahd_train_scaled[:, 0], yes_ahd_train_scaled[:, 1], label="Yes AHD")
axes[0][2].scatter(polynomial_model.support_vectors_[:, 0], polynomial_model.support_vectors_[:, 1], label="Support Vectors", facecolors="none", edgecolors="black", s=100)
axes[0][2].contourf(xx, yy, D_polynomial > 0, alpha=0.3)
axes[0][2].contour(xx, yy, D_polynomial, levels=[-1, 0, 1], colors="black", linestyles=["--", "-", "--"], linewidths=[1, 2, 1])
axes[0][2].set_title("Polynomial SVC - Train")
axes[0][2].set_ylabel(f"{feature_y} (scaled)")
axes[0][2].set_xlabel(f"{feature_x} (scaled)")

axes[0][3].scatter(no_ahd_train_scaled[:, 0], no_ahd_train_scaled[:, 1], label="No AHD")
axes[0][3].scatter(yes_ahd_train_scaled[:, 0], yes_ahd_train_scaled[:, 1], label="Yes AHD")
axes[0][3].scatter(rbf_model.support_vectors_[:, 0], rbf_model.support_vectors_[:, 1], label="Support Vectors", facecolors="none", edgecolors="black", s=100)
axes[0][3].contourf(xx, yy, D_rbf > 0, alpha=0.3)
axes[0][3].contour(xx, yy, D_rbf, levels=[-1, 0, 1], colors="black", linestyles=["--", "-", "--"], linewidths=[1, 2, 1])
axes[0][3].set_title("RBF SVC - Train")
axes[0][3].set_ylabel(f"{feature_y} (scaled)")
axes[0][3].set_xlabel(f"{feature_x} (scaled)")

print("\nSVC models accuracy:\n")
print(f"Linear: {linear_accuracy * 100:.2f}%")
print(f"Polynomial: {polynomial_accuracy * 100:.2f}%")
print(f"RBF: {rbf_accuracy * 100:.2f}%")

print("\nSVC models best parameters:\n")

print("Linear best parameters:", linear_grid.best_params_)
print("Polynomial best parameters:", polynomial_grid.best_params_)
print("RBF best parameters:", rbf_grid.best_params_)

no_ahd_test_scaled = X_test_scaled[y_test == "No"]
yes_ahd_test_scaled = X_test_scaled[y_test == "Yes"]

x_test = np.linspace(X_test_scaled[:, 0].min(), X_test_scaled[:, 0].max(), 100)
y_test_grid = np.linspace(X_test_scaled[:, 1].min(), X_test_scaled[:, 1].max(), 100)

xx_test, yy_test = np.meshgrid(x_test, y_test_grid)
grid_test = np.c_[xx_test.ravel(), yy_test.ravel()]

D_linear_test = linear_model.decision_function(grid_test).reshape(xx_test.shape)
D_polynomial_test = polynomial_model.decision_function(grid_test).reshape(xx_test.shape)
D_rbf_test = rbf_model.decision_function(grid_test).reshape(xx_test.shape)

axes[1][1].scatter(no_ahd_test_scaled[:, 0], no_ahd_test_scaled[:, 1], label="No AHD")
axes[1][1].scatter(yes_ahd_test_scaled[:, 0], yes_ahd_test_scaled[:, 1], label="Yes AHD")
axes[1][1].contourf(xx_test, yy_test, D_linear_test > 0, alpha=0.3)
axes[1][1].contour(xx_test, yy_test, D_linear_test, levels=[1], colors="black", linewidths=2)
axes[1][1].set_title(f"Linear SVC - Test ({linear_accuracy * 100:.2f}%)")
axes[1][1].set_xlabel(f"{feature_x} (scaled)")
axes[1][1].set_ylabel(f"{feature_y} (scaled)")

axes[1][2].scatter(no_ahd_test_scaled[:, 0], no_ahd_test_scaled[:, 1], label="No AHD")
axes[1][2].scatter(yes_ahd_test_scaled[:, 0], yes_ahd_test_scaled[:, 1], label="Yes AHD")
axes[1][2].contourf(xx_test, yy_test, D_polynomial_test > 0, alpha=0.3)
axes[1][2].contour(xx_test, yy_test, D_polynomial_test, levels=[0], colors="black", linewidths=2)
axes[1][2].set_title(f"Polynomial SVC - Test ({polynomial_accuracy * 100:.2f}%)")
axes[1][2].set_xlabel(f"{feature_x} (scaled)")
axes[1][2].set_ylabel(f"{feature_y} (scaled)")

axes[1][3].scatter(no_ahd_test_scaled[:, 0], no_ahd_test_scaled[:, 1], label="No AHD")
axes[1][3].scatter(yes_ahd_test_scaled[:, 0], yes_ahd_test_scaled[:, 1], label="Yes AHD")
axes[1][3].contourf(xx_test, yy_test, D_rbf_test > 0, alpha=0.3)
axes[1][3].contour(xx_test, yy_test, D_rbf_test, levels=[0], colors="black", linewidths=2)
axes[1][3].set_title(f"RBF SVC - Test ({rbf_accuracy * 100:.2f}%)")
axes[1][3].set_xlabel(f"{feature_x} (scaled)")
axes[1][3].set_ylabel(f"{feature_y} (scaled)")

plt.tight_layout()
plt.show()