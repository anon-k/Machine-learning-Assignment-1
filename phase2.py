import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from itertools import product
from numpy.polynomial import legendre as L
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

MAX_DEG = 20
USE_1SE = False    
RUN_LASSO = True

# ---------- 1. Load data ----------
train = pd.read_csv("IMT2024079_train_var2.csv")
test  = pd.read_csv("IMT2024079_test_var2.csv")
feature_cols = ["x1", "x2", "x3"]

X = train[feature_cols].values
y = train["y"].values
X_test = test[feature_cols].values
print("Train:", X.shape, "| Test:", X_test.shape)
print("Train range:", X.min(0).round(3), X.max(0).round(3))
print("Test  range:", X_test.min(0).round(3), X_test.max(0).round(3))

# ---------- 2. Build Legendre features for degree 20 ----------
#Terms sorted by total degree, so degree d = first n_terms[d] columns.
combos = sorted((c for c in product(range(MAX_DEG + 1), repeat=3)
                 if 0 < sum(c) <= MAX_DEG), key=sum)
total_deg = np.array([sum(c) for c in combos])
n_terms = {d: int((total_deg <= d).sum()) for d in range(1, MAX_DEG + 1)}

lo, hi = X.min(axis=0), X.max(axis=0)          

def legendre_matrix(A):
    Z = 2 * (A - lo) / (hi - lo) - 1
    V = [L.legvander(Z[:, v], MAX_DEG) for v in range(3)]
    return np.column_stack([V[0][:, i] * V[1][:, j] * V[2][:, k]
                            for i, j, k in combos])

Phi = legendre_matrix(X)
Phi_test = legendre_matrix(X_test)

def cols(d):
    return slice(0, n_terms[d])

# ---------- 3. Training model ----------
def make_model(kind):
    if kind == "ridge":
        reg = RidgeCV(alphas=np.logspace(-8, 2, 40))
    else:
        reg = LassoCV(alphas=np.logspace(-6, -1, 25), cv=5,
                      max_iter=20000, tol=1e-3, n_jobs=1)
    return make_pipeline(StandardScaler(), reg)

# ---------- 4. One CV pass per (kind, degree): MSE and R2, train and val ----------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"mse": "neg_mean_squared_error", "r2": "r2"}
records = []

def evaluate(kind, d):
    cv = cross_validate(make_model(kind), Phi[:, cols(d)], y, cv=kf,
                        scoring=scoring, return_train_score=True, n_jobs=-1)
    val_mse = -cv["test_mse"]
    records.append(dict(
        kind=kind, degree=d, terms=n_terms[d],
        train_mse=-cv["train_mse"].mean(), val_mse=val_mse.mean(),
        val_mse_se=val_mse.std(ddof=1) / np.sqrt(len(val_mse)),
        train_r2=cv["train_r2"].mean(), val_r2=cv["test_r2"].mean()))
    r = records[-1]
    print(f"{kind:5s} d={d:2d} terms={r['terms']:4d} | "
          f"MSE train={r['train_mse']:.6f} val={r['val_mse']:.6f} | "
          f"R2 train={r['train_r2']:.6f} val={r['val_r2']:.6f}")

for d in range(1, MAX_DEG + 1):
    evaluate("ridge", d)

if RUN_LASSO:   
    rdf = pd.DataFrame(records)
    best_r = int(rdf.loc[rdf.val_mse.idxmin(), "degree"])
    for d in range(max(1, best_r - 3), min(MAX_DEG, best_r + 3) + 1):
        evaluate("lasso", d)

res = pd.DataFrame(records)
res.to_csv("phase2_cv_results.csv", index=False)

# ---------- 5. Choose the best (kind, degree) ----------
best = res.loc[res.val_mse.idxmin()]
if USE_1SE:
    same = res[res.kind == best.kind]
    ok = same[same.val_mse <= best.val_mse + best.val_mse_se]
    best = ok.sort_values("degree").iloc[0]
best_kind, best_deg = best.kind, int(best.degree)

print(f"\nBest: {best_kind}, degree {best_deg}  "
      f"(CV MSE={best.val_mse:.6f}, CV R2={best.val_r2:.6f})")

# ---------- 6. Hold-out sanity check ----------
idx = np.arange(len(y))
tr, va = train_test_split(idx, test_size=0.2, random_state=0)
m = make_model(best_kind).fit(Phi[tr][:, cols(best_deg)], y[tr])
pv = m.predict(Phi[va][:, cols(best_deg)])
print(f"Hold-out MSE: {mean_squared_error(y[va], pv):.6f}")
print(f"Hold-out R2 : {r2_score(y[va], pv):.6f}")

# ---------- 7. Final fit on all training data, predict test ----------
final = make_model(best_kind).fit(Phi[:, cols(best_deg)], y)
test_pred = final.predict(Phi_test[:, cols(best_deg)])
pd.DataFrame({"y": test_pred}).to_csv("predictions_phase2.csv", index=False)
print(f"\nSaved predictions_phase2.csv with {len(test_pred)} rows")
print(pd.Series(test_pred).describe())

# ---------- 8. Plots: MSE (train/val) and R2 (val) vs degree ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
for kind, style in [("ridge", "-"), ("lasso", "--")]:
    g = res[res.kind == kind]
    if g.empty:
        continue
    a1.plot(g.degree, g.train_mse, "o" + style, label=f"{kind} train")
    a1.plot(g.degree, g.val_mse, "s" + style, label=f"{kind} validation")
    a1.fill_between(g.degree, g.val_mse - g.val_mse_se,
                    g.val_mse + g.val_mse_se, alpha=0.15)
    a2.plot(g.degree, g.val_r2, "s" + style, label=f"{kind} val R²")
    a2.plot(g.degree, g.train_r2, "o" + style, alpha=0.5, label=f"{kind} train R²")

a1.axvline(best_deg, color="green", ls=":", label=f"chosen = {best_deg}")
a2.axvline(best_deg, color="green", ls=":")
a1.set_yscale("log"); a1.set_title("MSE vs degree"); a1.set_ylabel("MSE (log)")
a2.set_title("R² vs degree"); a2.set_ylabel("R²")
a2.set_ylim(max(-0.1, res.val_r2.min() - 0.02), 1.002)
for a in (a1, a2):
    a.set_xlabel("Polynomial degree"); a.set_xticks(range(1, MAX_DEG + 1))
    a.grid(alpha=0.3); a.legend()
plt.tight_layout()
plt.savefig("phase2_degree_curve.png", dpi=150)
plt.show()