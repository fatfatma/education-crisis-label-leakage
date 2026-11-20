"""A+ geleneksel boru hatti: A-C kurgulariyla AYNI veri ve AYNI etiketle yeniden kurulum.
education_crisis_v3 defterindeki boru hattini (lag/trend/temel/sinyal ozellikleri, belirsiz
gozlem temizligi, StandardScaler + BorderlineSMOTE, Optuna TPE 75 deneme, 11 model,
GroupKFold CV, LOSO, hold-out, SelectKBest, SHAP, ablasyon) izler.

Kurulum:  pip install optuna xgboost imbalanced-learn shap scikit-learn pandas numpy matplotlib
Kullanim: python aplus.py <data_proje_klasoru> <metadata_country_csv> <cikti_klasoru> [--trials 75] [--no-loso]
Hizli deneme: --trials 3 --no-loso   (sonuclar makale icin degil, sadece calistigini gormek icin)
Ara kayit: Optuna en iyi parametreleri <cikti>/aplus_params.json'a yazilir; yarida kalirsa
ayni komutla tekrar calistirinca tamamlanan modeller atlanir.
"""
import sys, glob, os, json, time, argparse, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("meta"); ap.add_argument("out")
ap.add_argument("--trials", type=int, default=75)
ap.add_argument("--no-loso", action="store_true")
args = ap.parse_args()
DATA, META, OUT = args.data, args.meta, args.out
os.makedirs(OUT, exist_ok=True)
RNG = 42

CODES = {
    "FP.CPI.TOTL.ZG": "inflation_rate", "IT.NET.USER.ZS": "internet_access",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth", "NY.GNP.PCAP.CD": "gni_per_capita",
    "SE.SEC.CMPT.LO.ZS": "secondary_completion", "SE.TER.ENRR": "tertiary_enrollment",
    "SE.XPD.TOTL.GD.ZS": "edu_spending_gdp", "SL.TLF.CACT.FE.ZS": "female_labor_participation",
    "SL.UEM.TOTL.ZS": "unemployment_rate", "SP.DYN.IMRT.IN": "infant_mortality",
    "SP.DYN.LE00.IN": "life_expectancy", "SP.DYN.TFRT.IN": "fertility_rate",
    "SP.POP.65UP.TO.ZS": "elderly_population_pct", "SP.POP.GROW": "population_growth",
    "SP.URB.TOTL.IN.ZS": "urbanization_rate",
}

# 1) Veri: 15 WB gostergesi, gercek ulkeler, 2000-2023 (GII tek yillik oldugu icin alinmaz)
meta = pd.read_csv(META)
countries = set(meta.loc[meta["Region"].notna(), "Country Code"])
frames = []
for code, name in CODES.items():
    hits = [h for h in glob.glob(os.path.join(DATA, "**", f"API_{code}_DS2*.csv"), recursive=True)
            if not os.path.basename(h).startswith("Metadata")]
    if not hits:
        sys.exit(f"HATA: '{DATA}' klasorunde (alt klasorler dahil) API_{code}_DS2...csv bulunamadi. "
                 f"Klasor yolunu ve CSV dosyalarinin orada oldugunu kontrol edin.")
    f = hits[0]
    d = pd.read_csv(f, skiprows=4)
    d = d[d["Country Code"].isin(countries)]
    years = [str(y) for y in range(1995, 2024)]
    long = d.melt(id_vars="Country Code", value_vars=years, var_name="Year", value_name=name)
    long["Year"] = long["Year"].astype(int)
    frames.append(long.set_index(["Country Code", "Year"]))
df = pd.concat(frames, axis=1).reset_index().sort_values(["Country Code", "Year"])
feat_base = list(CODES.values())

# 2) Etiket (makaledeki kural). Etiket bilesenleri ulke ici ic-enterpolasyonla (disa uzatma yok) tanimlanir.
g = df.groupby("Country Code")
interp = lambda s: s.interpolate(limit_area="inside")
gni_i = g["gni_per_capita"].transform(interp)
edu_i = g["tertiary_enrollment"].transform(interp)
une_i = g["unemployment_rate"].transform(interp)
df["L_gni_chg"] = gni_i.groupby(df["Country Code"]).pct_change(fill_method=None) * 100
df["L_edu_chg"] = edu_i.groupby(df["Country Code"]).diff()
mu = une_i.groupby(df["Country Code"]).transform("mean"); sd = une_i.groupby(df["Country Code"]).transform("std")
df["L_unemp_z"] = (une_i - mu) / sd
ok = df[["L_gni_chg", "L_edu_chg", "L_unemp_z"]].notna().all(axis=1)
score = 2 * (df["L_gni_chg"] < 0) + 1 * (df["L_unemp_z"] > 0) + 1 * (df["L_edu_chg"] < 0)
df["label"] = np.where(ok, np.select([score >= 3, score >= 1], [0, 1], 2), np.nan)  # 0 Kriz, 1 Risk, 2 Stabil
df["score"] = np.where(ok, score, np.nan)
# A+ icin 1995-2023 panel korunur (gecikmeler icin); modelleme 2000-2023 etiketli gozlemlerle yapilir


# ======================================================================
# A+ on isleme: iki yonlu enterpolasyon + global medyan + IQR winsorizasyon
# ======================================================================
import sklearn
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42

lines = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)
def save_txt():
    with open(os.path.join(OUT, "aplus_sonuclar.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

P = df.copy()
gp = P.groupby("Country Code")
for c in feat_base:
    P[c] = gp[c].transform(lambda s: s.interpolate(limit_direction="both"))
    P[c] = P[c].fillna(P[c].median())
    q1, q3 = P[c].quantile(0.25), P[c].quantile(0.75); iqr = q3 - q1
    P[c] = P[c].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

lag_base = ["inflation_rate", "internet_access", "gdp_growth", "secondary_completion",
            "edu_spending_gdp", "female_labor_participation", "infant_mortality",
            "life_expectancy", "fertility_rate", "elderly_population_pct",
            "population_growth", "urbanization_rate"]
gp = P.groupby("Country Code")
lag_f = []
for c in lag_base:
    for L in (1, 2, 3):
        P[f"{c}_lag{L}"] = gp[c].shift(L); lag_f.append(f"{c}_lag{L}")
trend_f = []
for c in ["gdp_growth", "inflation_rate", "internet_access", "population_growth"]:
    P[f"{c}_3yr_avg"] = gp[c].transform(lambda s: s.shift(1).rolling(3).mean()); trend_f.append(f"{c}_3yr_avg")
base_f = ["gni_per_capita", "unemployment_rate", "tertiary_enrollment", "internet_access", "gdp_growth"]
# Sinyal ozellikleri = etiket bilesenlerinin kendisi (A+ boru hattindaki tasarim)
P["gni_pct_change"] = P["L_gni_chg"]; P["gni_shock"] = (P["L_gni_chg"] < 0).astype(float)
P["unemp_z"] = P["L_unemp_z"]; P["edu_decline"] = (P["L_edu_chg"] < 0).astype(float)
sig_f = ["unemp_z", "gni_pct_change", "gni_shock", "edu_decline"]
FEATS = lag_f + trend_f + base_f + sig_f

P = P[(P["Year"] >= 2000) & (P["Year"] <= 2023) & P["label"].notna()].copy()
P = P.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATS).reset_index(drop=True)
X = P[FEATS]; y = P["label"].astype(int); groups = P["Country Code"]; amb = (P["score"] == 2).values

tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(X, y, groups))
Xtr_raw, Xte = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
ytr_raw, yte = y.iloc[tr].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
gtr_raw = groups.iloc[tr].reset_index(drop=True)
amb_tr, amb_te = amb[tr], amb[te]
Xtr, ytr, gtr = Xtr_raw[~amb_tr].reset_index(drop=True), ytr_raw[~amb_tr].reset_index(drop=True), gtr_raw[~amb_tr].reset_index(drop=True)

say("=== A+ boru hatti (A-C ile ayni veri ve etiket) ===")
say(f"Python {sys.version.split()[0]}, scikit-learn {sklearn.__version__}, optuna {optuna.__version__}, deneme sayisi {args.trials}")
say(f"Gozlem: {len(P)} | ulke: {P['Country Code'].nunique()} | ozellik: {len(FEATS)}")
say("Sinif dagilimi (0 Kriz,1 Risk,2 Stabil):", y.value_counts().sort_index().to_dict())
say(f"Puan=2 (belirsiz) gozlem: {int(amb.sum())} | egitimden cikarilan: {int(amb_tr.sum())}")
say(f"Egitim: {len(Xtr)} ({gtr.nunique()} ulke) | Test: {len(Xte)} ({groups.iloc[te].nunique()} ulke)")
save_txt()

def make_pipe(model, smote=True):
    st = [("scaler", StandardScaler())]
    if smote: st.append(("smote", BorderlineSMOTE(random_state=SEED, kind="borderline-1")))
    st.append(("clf", model)); return ImbPipeline(st)

def holdout(pipe, Xa, ya, Xb, yb):
    pipe.fit(Xa, ya); p = pipe.predict(Xb); auc = np.nan
    try: auc = roc_auc_score(label_binarize(yb, classes=[0, 1, 2]), pipe.predict_proba(Xb), multi_class="ovr", average="macro")
    except Exception: pass
    return p, {"Acc": accuracy_score(yb, p), "Prec": precision_score(yb, p, average="macro", zero_division=0),
               "Rec": recall_score(yb, p, average="macro", zero_division=0),
               "F1": f1_score(yb, p, average="macro", zero_division=0), "AUC": auc}

# ---------------- Optuna ----------------
def cv_obj(trial, model):
    sc = []
    for k, (a, b) in enumerate(GroupKFold(n_splits=3).split(Xtr, ytr, gtr)):
        pp = make_pipe(clone(model)); pp.fit(Xtr.iloc[a], ytr.iloc[a])
        s = f1_score(ytr.iloc[b], pp.predict(Xtr.iloc[b]), average="macro", zero_division=0)
        sc.append(s); trial.report(s, k)
        if trial.should_prune(): raise optuna.TrialPruned()
    return float(np.mean(sc))

SPACE = {
 "Lojistik Regresyon": lambda t: LogisticRegression(C=t.suggest_float("C", 1e-3, 10.0, log=True), solver=t.suggest_categorical("solver", ["lbfgs", "saga"]), class_weight="balanced", max_iter=2000, random_state=SEED),
 "SVM": lambda t: SVC(C=t.suggest_float("C", 0.01, 20.0, log=True), gamma=t.suggest_categorical("gamma", ["scale", "auto"]), kernel=t.suggest_categorical("kernel", ["rbf", "poly"]), class_weight="balanced", random_state=SEED),
 "Random Forest": lambda t: RandomForestClassifier(n_estimators=t.suggest_int("n_estimators", 100, 500, step=50), max_depth=t.suggest_int("max_depth", 5, 25), min_samples_split=t.suggest_int("min_samples_split", 2, 10), min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 8), max_features=t.suggest_categorical("max_features", ["sqrt", "log2"]), class_weight="balanced", random_state=SEED, n_jobs=-1),
 "Gradient Boosting": lambda t: GradientBoostingClassifier(n_estimators=t.suggest_int("n_estimators", 100, 400, step=50), learning_rate=t.suggest_float("learning_rate", 0.005, 0.3, log=True), max_depth=t.suggest_int("max_depth", 3, 10), subsample=t.suggest_float("subsample", 0.5, 1.0), max_features=t.suggest_categorical("max_features", ["sqrt", "log2", None]), random_state=SEED),
 "Extra Trees": lambda t: ExtraTreesClassifier(n_estimators=t.suggest_int("n_estimators", 100, 500, step=50), max_depth=t.suggest_int("max_depth", 5, 25), min_samples_split=t.suggest_int("min_samples_split", 2, 10), min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 8), max_features=t.suggest_categorical("max_features", ["sqrt", "log2"]), class_weight="balanced", random_state=SEED, n_jobs=-1),
 "XGBoost": lambda t: XGBClassifier(n_estimators=t.suggest_int("n_estimators", 100, 400, step=50), max_depth=t.suggest_int("max_depth", 3, 12), learning_rate=t.suggest_float("learning_rate", 0.005, 0.3, log=True), subsample=t.suggest_float("subsample", 0.5, 1.0), colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0), min_child_weight=t.suggest_int("min_child_weight", 1, 10), reg_alpha=t.suggest_float("reg_alpha", 1e-4, 10.0, log=True), reg_lambda=t.suggest_float("reg_lambda", 1e-4, 10.0, log=True), eval_metric="mlogloss", random_state=SEED),
 "KNN": lambda t: KNeighborsClassifier(n_neighbors=t.suggest_int("n_neighbors", 3, 20), weights=t.suggest_categorical("weights", ["uniform", "distance"]), metric=t.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"])),
 "Karar Ağacı": lambda t: DecisionTreeClassifier(max_depth=t.suggest_int("max_depth", 3, 20), min_samples_split=t.suggest_int("min_samples_split", 2, 10), min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 8), criterion=t.suggest_categorical("criterion", ["gini", "entropy"]), class_weight="balanced", random_state=SEED),
}
ppath = os.path.join(OUT, "aplus_params.json")
saved = {}
if os.path.exists(ppath):
    with open(ppath, encoding="utf-8") as fh:
        saved = json.load(fh)
if saved.get("_trials") != args.trials: saved = {"_trials": args.trials}
say("\n=== Optuna (TPE, Median Pruner, 3 katli GroupKFold) ===")
opt = {}
for name, build in SPACE.items():
    if name not in saved:
        t0 = time.time()
        st = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED), pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=1))
        st.optimize(lambda t: cv_obj(t, build(t)), n_trials=args.trials)
        saved[name] = {"params": st.best_params, "cv": st.best_value, "sure_dk": round((time.time() - t0) / 60, 1)}
        with open(ppath, "w", encoding="utf-8") as fh:
            json.dump(saved, fh, ensure_ascii=False, indent=1)
    opt[name] = build(optuna.trial.FixedTrial(saved[name]["params"]))
    say(f"{name:20s} en iyi ic CV makro F1 = {saved[name]['cv']:.4f}")
save_txt()
svc_prob = clone(opt["SVM"]).set_params(probability=True)

models = {
 "Lojistik Regresyon": opt["Lojistik Regresyon"], "KNN": opt["KNN"], "Karar Ağacı": opt["Karar Ağacı"],
 "SVM": svc_prob, "Random Forest": opt["Random Forest"], "Gradient Boosting": opt["Gradient Boosting"],
 "Extra Trees": opt["Extra Trees"], "XGBoost": opt["XGBoost"],
 "MLP (Sığ)": MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000, early_stopping=True, validation_fraction=0.1, random_state=SEED),
 "MLP (Derin)": MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32), max_iter=2000, early_stopping=True, validation_fraction=0.1, random_state=SEED),
 "Stacking Ansambl": StackingClassifier(
     estimators=[("rf", opt["Random Forest"]), ("xgb", opt["XGBoost"]), ("gb", opt["Gradient Boosting"]),
                 ("svc", CalibratedClassifierCV(opt["SVM"], cv=3)), ("lr", opt["Lojistik Regresyon"])],
     final_estimator=XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss", random_state=SEED),
     cv=5, passthrough=True),
}

# ---------------- Hold-out + CV + LOSO ----------------
say("\n=== Hold-out, 5 katli GroupKFold CV ve LOSO (egitim kumesi icinde) ===")
rows = []
for name, m in models.items():
    t0 = time.time()
    _, h = holdout(make_pipe(clone(m)), Xtr, ytr, Xte, yte)
    cv = np.nan; lo = np.nan
    if name != "Stacking Ansambl":
        cv = np.mean([f1_score(ytr.iloc[b], make_pipe(clone(m)).fit(Xtr.iloc[a], ytr.iloc[a]).predict(Xtr.iloc[b]), average="macro", zero_division=0)
                      for a, b in GroupKFold(n_splits=5).split(Xtr, ytr, gtr)])
    if name in SPACE and not args.no_loso:
        mm = opt[name]
        lo = np.mean([f1_score(ytr.iloc[b], make_pipe(clone(mm)).fit(Xtr.iloc[a], ytr.iloc[a]).predict(Xtr.iloc[b]), average="macro", zero_division=0)
                      for a, b in LeaveOneGroupOut().split(Xtr, ytr, gtr)])
    rows.append({"Model": name, "CV_F1": cv, "LOSO_F1": lo, "Test_F1": h["F1"], "Acc": h["Acc"], "Prec": h["Prec"], "Rec": h["Rec"], "ROC_AUC": h["AUC"]})
    say(f"{name:20s} CV={cv:.3f}  LOSO={lo:.3f}  Test F1={h['F1']:.3f}  AUC={h['AUC']:.3f}  ({(time.time()-t0)/60:.1f} dk)")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "aplus_modeller.csv"), index=False); save_txt()
R = pd.DataFrame(rows).sort_values("Test_F1", ascending=False)
say("\n" + R.round(3).to_string(index=False))

# ---------------- SelectKBest ----------------
say("\n=== SelectKBest (k=20, ANOVA F) ===")
sk = SelectKBest(f_classif, k=20).fit(Xtr, ytr)
Fs = pd.Series(sk.scores_, index=FEATS).sort_values(ascending=False)
for i, (f, v) in enumerate(Fs.head(10).items(), 1): say(f"{i:2d}. {f:30s} F={v:.1f}")
rf = opt["Random Forest"]
_, mf = holdout(make_pipe(clone(rf)), Xtr, ytr, Xte, yte)
kb = ImbPipeline([("scaler", StandardScaler()), ("smote", BorderlineSMOTE(random_state=SEED, kind="borderline-1")),
                  ("sel", SelectKBest(f_classif, k=20)), ("clf", clone(rf))])
_, mk = holdout(kb, Xtr, ytr, Xte, yte)
say(f"RF tam set ({len(FEATS)}) Test F1={mf['F1']:.3f} AUC={mf['AUC']:.4f} | SelectKBest(20) Test F1={mk['F1']:.3f} AUC={mk['AUC']:.4f}")
save_txt()

# ---------------- Ablasyon ----------------
say("\n=== Ablasyon: belirsiz gozlem temizligi (Random Forest) ===")
for nm, Xa, ya, Xb, yb in [("1: Temizleme yok", Xtr_raw, ytr_raw, Xte, yte),
                           ("2: Yalnizca egitim", Xtr, ytr, Xte, yte),
                           ("3: Egitim ve test", Xtr, ytr, Xte[~amb_te].reset_index(drop=True), yte[~amb_te].reset_index(drop=True))]:
    _, h = holdout(make_pipe(clone(rf)), Xa, ya, Xb, yb)
    say(f"{nm:20s} F1={h['F1']:.3f} Acc={h['Acc']:.3f} Prec={h['Prec']:.3f} AUC={h['AUC']:.4f}")
save_txt()

# ---------------- SHAP ----------------
say("\n=== SHAP (Random Forest, Kriz sinifi) ===")
try:
    import shap, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sp = make_pipe(clone(rf)).fit(Xtr, ytr)
    Xs = pd.DataFrame(sp.named_steps["scaler"].transform(Xte), columns=FEATS)
    sv = shap.TreeExplainer(sp.named_steps["clf"]).shap_values(Xs, check_additivity=False)
    sv0 = sv[:, :, 0] if isinstance(sv, np.ndarray) and sv.ndim == 3 else (sv[0] if isinstance(sv, list) else sv)
    ms = pd.Series(np.abs(sv0).mean(axis=0), index=FEATS).sort_values(ascending=False)
    for i, (f, v) in enumerate(ms.head(10).items(), 1): say(f"{i:2d}. {f:30s} ortalama |SHAP|={v:.4f}")
    plt.figure(figsize=(8, 9))
    shap.summary_plot(sv0, Xs, max_display=20, show=False)
    fig = plt.gcf(); fig.canvas.draw(); fig.axes[0].set_xlabel("SHAP değeri (Kriz tahminine etkisi)")
    for ax in fig.axes[1:]:
        ax.set_ylabel("Özellik değeri")
        tl = [t.get_text().replace("High", "Yüksek").replace("Low", "Düşük") for t in ax.get_yticklabels()]
        if any(tl): ax.set_yticklabels(tl)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "shap_summary_rf.png"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(OUT, "shap_summary_rf.pdf"), bbox_inches="tight")
    say("Kaydedildi: shap_summary_rf.png / .pdf")
except Exception as e:
    say("SHAP calistirilamadi:", repr(e))
save_txt()
say("\nBitti. Sonuclar:", os.path.join(OUT, "aplus_sonuclar.txt"))
save_txt()
