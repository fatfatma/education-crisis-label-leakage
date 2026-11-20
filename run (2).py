"""Egitim erisim krizi: sizintili ve sizintisiz kurgularin karsilastirmasi.
Makaledeki etiket kurali ham Dunya Bankasi verisinden yeniden uretilir.
Kullanim: python -I run.py <data_proje_klasoru> <metadata_country_csv> <cikti_klasoru>
"""
import sys, glob, os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

DATA, META, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
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
df = df[(df["Year"] >= 2000) & (df["Year"] <= 2023)].copy()

# 3) Ozellikler: yalnizca gecmise dayali (ileri doldurma, gecmis degisimler, genisleyen z-skor)
g = df.groupby("Country Code")
for c in feat_base:
    df[c + "_ff"] = g[c].ffill()
g = df.groupby("Country Code")
past = []
for c in feat_base:
    df[c + "_chg1"] = g[c + "_ff"].diff()
    df[c + "_lag1"] = g[c + "_ff"].shift(1)
    past += [c + "_ff", c + "_chg1", c + "_lag1"]
# Makaledeki 4 sinyal ozelliginin gecmise dayali surumleri (t yilinin degerleri)
ff = df.groupby("Country Code")
df["S_gni_pct_change"] = ff["gni_per_capita_ff"].pct_change(fill_method=None) * 100
df["S_gni_shock"] = (df["S_gni_pct_change"] < 0).astype(float)
exp_mu = ff["unemployment_rate_ff"].transform(lambda s: s.expanding(3).mean().shift(1))
exp_sd = ff["unemployment_rate_ff"].transform(lambda s: s.expanding(3).std().shift(1))
df["S_unemp_z"] = (df["unemployment_rate_ff"] - exp_mu) / exp_sd
df["S_edu_decline"] = (ff["tertiary_enrollment_ff"].diff() < 0).astype(float)
signal = ["S_gni_pct_change", "S_gni_shock", "S_unemp_z", "S_edu_decline"]
# Makaledeki sizintili surum: etiket bilesenlerinin kendisi
leaky = ["L_gni_chg", "L_edu_chg", "L_unemp_z"]

num = df.select_dtypes("number").columns
df[num] = df[num].replace([np.inf, -np.inf], np.nan)
df["label_next"] = df.groupby("Country Code")["label"].shift(-1)
df["label_prev"] = df.groupby("Country Code")["label"].shift(1)

def models():
    imp = lambda: SimpleImputer(strategy="median")  # egitim verisinden ogrenilir
    return {
        "LojistikRegresyon": make_pipeline(imp(), StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
        "RandomForest": make_pipeline(imp(), RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=RNG)),
        "GradientBoosting": make_pipeline(imp(), HistGradientBoostingClassifier(class_weight="balanced", random_state=RNG)),
    }

def scores(y, p):
    return {"makro_F1": f1_score(y, p, average="macro"), "Kriz_recall": recall_score(y, p, labels=[0], average="macro", zero_division=0)}

def boot_ci(y, p, groups, n=1000):
    rng = np.random.default_rng(RNG); u = np.unique(groups); out = []
    idx = {k: np.where(groups == k)[0] for k in u}
    for _ in range(n):
        s = np.concatenate([idx[k] for k in rng.choice(u, len(u), replace=True)])
        out.append(f1_score(y[s], p[s], average="macro"))
    return np.percentile(out, [2.5, 97.5])

results = []
def run(name, data, feats, target, split):
    d = data.dropna(subset=[target]).copy()
    if split == "ulke_rastgele":
        tr, te = next(GroupShuffleSplit(test_size=0.2, random_state=RNG).split(d, groups=d["Country Code"]))
        dtr, dte = d.iloc[tr], d.iloc[te]
    else:  # zamana gore: hedef yili <= 2016 egitim, 2018-2023 test
        ty = d["Year"] + (1 if target == "label_next" else 0)
        dtr, dte = d[ty <= 2016], d[ty >= 2018]
    for mname, m in models().items():
        m.fit(dtr[feats], dtr[target].astype(int))
        p = m.predict(dte[feats]); y = dte[target].astype(int).values
        s = scores(y, p); lo, hi = boot_ci(y, p, dte["Country Code"].values)
        results.append({"kurgu": name, "model": mname, "n_egitim": len(dtr), "n_test": len(dte), **s, "F1_GA_alt": lo, "F1_GA_ust": hi})
    # referanslar
    y = dte[target].astype(int).values
    maj = np.full_like(y, dtr[target].astype(int).mode()[0])
    results.append({"kurgu": name, "model": "Cogunluk_sinifi", "n_egitim": len(dtr), "n_test": len(dte), **scores(y, maj), "F1_GA_alt": np.nan, "F1_GA_ust": np.nan})
    if target == "label_next":
        prev = dte["label"].values; mask = ~np.isnan(prev)
        p = np.where(mask, prev, maj).astype(int); s = scores(y, p); lo, hi = boot_ci(y, p, dte["Country Code"].values)
        results.append({"kurgu": name, "model": "Kalicilik(t yilinin etiketi)", "n_egitim": len(dtr), "n_test": len(dte), **s, "F1_GA_alt": lo, "F1_GA_ust": hi})

print("Etiketli gozlem:", int(df["label"].notna().sum()), "| ulke:", df.loc[df["label"].notna(), "Country Code"].nunique())
print("Sinif dagilimi (0 Kriz,1 Risk,2 Stabil):", df["label"].value_counts(normalize=True).round(3).to_dict())

run("A: Ayni yil, etiket bilesenleri girdide, ulke bazli rastgele bolme (makaleye benzer)", df, past + leaky, "label", "ulke_rastgele")
run("B: Ayni yil, etiket bilesenleri cikarildi, zamana gore bolme", df, past, "label", "zaman")
run("C: Gelecek yil (t+1), gecmise dayali ozellikler + sinyal, zamana gore bolme", df, past + signal, "label_next", "zaman")

r = pd.DataFrame(results)
r.to_csv(os.path.join(OUT, "sonuclar.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(r.round(3).to_string(index=False))

# --- Ek kurgular ---
comp = ("gni_per_capita", "tertiary_enrollment", "unemployment_rate")
past_nocomp = [c for c in past if not c.startswith(comp)]
results.clear()
run("B2: Ayni yil, etiket bilesen gostergeleri (GNI, kayit, issizlik) tamamen cikarildi, zamana gore", df, past_nocomp, "label", "zaman")
df["kriz_next"] = (df["label_next"] == 0).astype(float).where(df["label_next"].notna())
df["kriz_now"] = (df["label"] == 0).astype(float).where(df["label"].notna())
from sklearn.metrics import average_precision_score
d = df.dropna(subset=["kriz_next"]); ty = d["Year"] + 1
dtr, dte = d[ty <= 2016], d[ty >= 2018]; y = dte["kriz_next"].astype(int).values
rows = []
for mname, m in models().items():
    m.fit(dtr[past + signal], dtr["kriz_next"].astype(int)); pr = m.predict_proba(dte[past + signal])[:, 1]; p = (pr >= 0.5).astype(int)
    rows.append({"model": mname, "F1_Kriz": f1_score(y, p), "Kriz_recall": recall_score(y, p), "PR_AUC": average_precision_score(y, pr)})
prev = dte["kriz_now"].fillna(0).astype(int).values
rows.append({"model": "Kalicilik", "F1_Kriz": f1_score(y, prev), "Kriz_recall": recall_score(y, prev), "PR_AUC": np.nan})
rows.append({"model": "Rastgele (Kriz orani)", "F1_Kriz": np.nan, "Kriz_recall": np.nan, "PR_AUC": y.mean()})
print(pd.DataFrame(results).round(3).to_string(index=False))
print("\nC2: Gelecek yil ikili (Kriz / Kriz degil), test Kriz orani =", round(y.mean(), 3))
b = pd.DataFrame(rows); print(b.round(3).to_string(index=False))
pd.concat([pd.read_csv(os.path.join(OUT, "sonuclar.csv")), pd.DataFrame(results)]).to_csv(os.path.join(OUT, "sonuclar.csv"), index=False)
b.to_csv(os.path.join(OUT, "sonuclar_ikili.csv"), index=False)
