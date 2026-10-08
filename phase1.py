import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import comb

warnings.filterwarnings("ignore")

MAX_DEG = 10
LASSO_DEGREES = range(3, 7)      
USE_1SE = False #Pick the (method, degree) pair with the lowest validation MSE
ALPHAS_LASSO = np.logspace(-4, -0.5, 40)
ALPHAS_RIDGE = np.logspace(-4, 4, 40)

# ---------- 1. Load data ----------
train = pd.read_csv("IMT2024079_train_var1.csv")
test = pd.read_csv("IMT2024079_test_var1.csv")

feature_cols = ["x1", "x2", "x3", "x4", "x5", "x6"]
X = train[feature_cols].values
y = train["y"].values
X_test = test[feature_cols].values
print("Train:", X.shape, "| Test:", X_test.shape)

# ---------- 2. Build polynomial features (degree 10) ----------
#Columns are ordered by total degree, so degree d = first n_terms[d] columns.
n_terms = {d: comb(d + 6, 6) - 1 for d in range(1, MAX_DEG + 1)}

sx = StandardScaler().fit(X)
poly = PolynomialFeatures(MAX_DEG, include_bias=False)
Phi = poly.fit_transform(sx.transform(X))
Phi_test = poly.transform(sx.transform(X_test))
print("Feature matrix:", Phi.shape)

def cols(d):
    return slice(0, n_terms[d])

# ---------- 3. Training model ----------
def make_model(kind):
    if kind == "ridge":
        reg = RidgeCV(alphas=ALPHAS_RIDGE)
    else:
        reg = LassoCV(alphas=ALPHAS_LASSO, cv=5, max_iter=20000, tol=1e-3, n_jobs=1)
    return make_pipeline(StandardScaler(), reg)   # scales polynomial's terms per fold

# ---------- 4. One CV pass per (kind, degree): MSE and R2 ----------
kf = KFold(n_splits=5, shuffle=True, random_state=42) #splits data into 5 folds
scoring = {"mse": "neg_mean_squared_error", "r2": "r2"}
records = []

def evaluate(kind, d):
    cv = cross_validate(make_model(kind), Phi[:, cols(d)], y, cv=kf,
                        scoring=scoring, return_train_score=True,
                        n_jobs=-1 if kind == "lasso" else 1)
    val_mse = -cv["test_mse"]
    records.append(dict(
        kind=kind, degree=d, terms=n_terms[d],
        train_mse=-cv["train_mse"].mean(), val_mse=val_mse.mean(),
        val_mse_se=val_mse.std(ddof=1) / np.sqrt(len(val_mse)),
        train_r2=cv["train_r2"].mean(), val_r2=cv["test_r2"].mean()))
    r = records[-1]
    print(f"{kind:5s} d={d:2d} terms={r['terms']:5d} | "
          f"MSE train={r['train_mse']:.5f} val={r['val_mse']:.5f} | "
          f"R2 train={r['train_r2']:.5f} val={r['val_r2']:.5f}")

for d in range(1, MAX_DEG + 1):
    evaluate("ridge", d)
for d in LASSO_DEGREES:
    evaluate("lasso", d)

res = pd.DataFrame(records)
res.to_csv("phase1_cv_results.csv", index=False)

# ---------- 5. Choose best (kind, degree) ----------
best = res.loc[res.val_mse.idxmin()]
if USE_1SE:
    same = res[res.kind == best.kind]
    ok = same[same.val_mse <= best.val_mse + best.val_mse_se]
    best = ok.sort_values("degree").iloc[0]
best_kind, best_deg = best.kind, int(best.degree)
print(f"\nBest: {best_kind}, degree {best_deg}  "
      f"(CV MSE={best.val_mse:.5f}, CV R2={best.val_r2:.5f})")

#Hold-out check
idx = np.arange(len(y))
tr, va = train_test_split(idx, test_size=0.2, random_state=0)
m = make_model(best_kind).fit(Phi[tr][:, cols(best_deg)], y[tr])
pv = m.predict(Phi[va][:, cols(best_deg)])
print(f"Hold-out MSE: {mean_squared_error(y[va], pv):.5f}")
print(f"Hold-out R2 : {r2_score(y[va], pv):.5f}")

# ---------- 6. Final fit on all training data, predict test ----------
final = make_model(best_kind).fit(Phi[:, cols(best_deg)], y)
reg = final[-1]
print(f"Chosen alpha: {reg.alpha_:.6g}")
if best_kind == "lasso":
    grid = ALPHAS_LASSO
    if reg.alpha_ <= grid.min() * 1.01 or reg.alpha_ >= grid.max() * 0.99:
        print("WARNING: alpha is at the edge of the grid. Consider widening it.")
    print("Nonzero terms:", (reg.coef_ != 0).sum(), "of", len(reg.coef_))

test_pred = final.predict(Phi_test[:, cols(best_deg)])
pd.DataFrame({"y": test_pred}).to_csv("predictions_phase1.csv", index=False)
print(f"\nSaved predictions_phase1.csv with {len(test_pred)} rows")
print(pd.Series(test_pred).describe())

# ---------- 7. Plots: MSE (train/val) and R2 vs degree ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
for kind, style in [("ridge", "-"), ("lasso", "--")]:
    g = res[res.kind == kind]
    if g.empty:
        continue
    a1.plot(g.degree, g.train_mse, "o" + style, label=f"{kind} train")
    a1.plot(g.degree, g.val_mse, "s" + style, label=f"{kind} validation")
    a1.fill_between(g.degree, g.val_mse - g.val_mse_se,
                    g.val_mse + g.val_mse_se, alpha=0.15)
    a2.plot(g.degree, g.train_r2, "o" + style, alpha=0.5, label=f"{kind} train R²")
    a2.plot(g.degree, g.val_r2, "s" + style, label=f"{kind} val R²")

a1.axvline(best_deg, color="green", ls=":", label=f"chosen = {best_deg}")
a2.axvline(best_deg, color="green", ls=":")
a1.set_yscale("log"); a1.set_title("MSE vs degree"); a1.set_ylabel("MSE (log)")
a2.set_title("R² vs degree"); a2.set_ylabel("R²")
a2.set_ylim(max(-0.1, res.val_r2.min() - 0.02), 1.002)
for a in (a1, a2):
    a.set_xlabel("Polynomial degree"); a.set_xticks(range(1, MAX_DEG + 1))
    a.grid(alpha=0.3); a.legend()
plt.tight_layout()
plt.savefig("phase1_degree_curve.png", dpi=150)
plt.show()

    