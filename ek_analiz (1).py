"""Hakem revizyonu icin ek analizler (run.py ile ayni veri, etiket ve ozellikler).
Kullanim: py ek_analiz.py <data_proje_klasoru> <metadata_country_csv> <cikti_klasoru>
Cikti: <cikti_klasoru>/ek_sonuclar.txt
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

# ======================================================================
# Ek analizler
# ======================================================================
lines = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)

# --- 1) Yalnizca gecmise dayali etiket (t yilinda gercekten bilinebilecek etiket) ---
# Bilesenler ileri doldurulmus seriden; issizlik z-skoru t dahil yalnizca t'ye kadarki yillardan.
ffg = df.groupby("Country Code")
po_gni = ffg["gni_per_capita_ff"].pct_change(fill_method=None) * 100
po_edu = ffg["tertiary_enrollment_ff"].diff()
po_mu = ffg["unemployment_rate_ff"].transform(lambda s: s.expanding(3).mean())
po_sd = ffg["unemployment_rate_ff"].transform(lambda s: s.expanding(3).std())
po_z = (df["unemployment_rate_ff"] - po_mu) / po_sd
po_ok = pd.concat([po_gni, po_edu, po_z], axis=1).replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
po_score = 2 * (po_gni < 0) + 1 * (po_z > 0) + 1 * (po_edu < 0)
df["label_po"] = np.where(po_ok, np.select([po_score >= 3, po_score >= 1], [0, 1], 2), np.nan)

# C kurgusunun test kumesi (run.py ile ayni)
d = df.dropna(subset=["label_next"]).copy(); ty = d["Year"] + 1
dtr, dte = d[ty <= 2016], d[ty >= 2018]
y = dte["label_next"].astype(int).values; grp = dte["Country Code"].values
maj = int(dtr["label_next"].astype(int).mode()[0])

both = dte["label"].notna() & dte["label_po"].notna()
say("=== 1) Gecmise dayali etiket ===")
say("C test kumesinde t yili etiketi ile gecmise dayali etiketin ayni oldugu oran:",
    round(float((dte.loc[both, "label"] == dte.loc[both, "label_po"]).mean()), 3), "| n =", int(both.sum()))

def persist(col):
    v = dte[col].values; return np.where(np.isnan(v), maj, v).astype(int)
preds = {"Kalicilik (t etiketi)": persist("label"), "Kalicilik (gecmise dayali etiket)": persist("label_po")}
feats = past + signal
for mname, m in models().items():
    m.fit(dtr[feats], dtr["label_next"].astype(int)); preds[mname] = m.predict(dte[feats])

# --- 2) Esli ulke bootstrap: model - kalicilik makro F1 farki ---
rng = np.random.default_rng(RNG); u = np.unique(grp); idx = {k: np.where(grp == k)[0] for k in u}
samples = [np.concatenate([idx[k] for k in rng.choice(u, len(u), replace=True)]) for _ in range(1000)]
say("\n=== 2) C kurgusu: makro F1 ve esli bootstrap farklari (ayni 1000 ornekleme) ===")
for k, p in preds.items():
    f = f1_score(y, p, average="macro"); bs = [f1_score(y[s], p[s], average="macro") for s in samples]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    say(f"{k:40s} makroF1={f:.3f}  GA=[{lo:.3f}, {hi:.3f}]  KrizRecall={recall_score(y, p, labels=[0], average='macro', zero_division=0):.3f}")
for base in ["Kalicilik (t etiketi)", "Kalicilik (gecmise dayali etiket)"]:
    for mname in models():
        diffs = np.array([f1_score(y[s], preds[mname][s], average="macro") - f1_score(y[s], preds[base][s], average="macro") for s in samples])
        obs = f1_score(y, preds[mname], average="macro") - f1_score(y, preds[base], average="macro")
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        say(f"{mname:18s} - {base:34s} fark={obs:+.3f}  GA=[{lo:+.3f}, {hi:+.3f}]  P(fark<=0)={np.mean(diffs <= 0):.3f}")

# --- 3) A+ tarzi etiketleme: iki yonlu enterpolasyon + global medyan ---
say("\n=== 3) Etiket dagilimi: bu calismanin tanimi vs A+ tarzi doldurma ===")
raw = df  # 2000-2023
g2 = raw.groupby("Country Code")
bi = lambda s: s.interpolate(limit_direction="both")
def label_from(gni, edu, une, zfull=True):
    gc = gni.groupby(raw["Country Code"]).pct_change(fill_method=None) * 100
    ec = edu.groupby(raw["Country Code"]).diff()
    m = une.groupby(raw["Country Code"]).transform("mean"); s = une.groupby(raw["Country Code"]).transform("std")
    z = (une - m) / s
    ok = pd.concat([gc, ec, z], axis=1).replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    sc = 2 * (gc < 0) + 1 * (z > 0) + 1 * (ec < 0)
    zero = ((gc == 0) | (ec == 0))
    return pd.Series(np.where(ok, np.select([sc >= 3, sc >= 1], [0, 1], 2), np.nan), index=raw.index), zero & ok, sc.where(ok)
gni_b = g2["gni_per_capita"].transform(bi); edu_b = g2["tertiary_enrollment"].transform(bi); une_b = g2["unemployment_rate"].transform(bi)
gni_b = gni_b.fillna(gni_b.median()); edu_b = edu_b.fillna(edu_b.median()); une_b = une_b.fillna(une_b.median())
lab_b, zero_b, sc_b = label_from(gni_b, edu_b, une_b)
def rep(name, lab, zero=None, sc=None):
    v = lab.dropna(); dist = v.value_counts(normalize=True).reindex([0, 1, 2]).round(3).to_dict()
    msg = f"{name}: n={len(v)}, ulke={raw.loc[v.index, 'Country Code'].nunique()}, Kriz/Risk/Stabil={dist}"
    if zero is not None: msg += f", GNI ya da kayit degisimi tam 0 olan oran={zero[v.index].mean():.3f}"
    if sc is not None: msg += f", puan=2 orani={(sc[v.index] == 2).mean():.3f}"
    say(msg)
rep("Bu calisma (ic enterpolasyon, disa uzatma yok)", df["label"])
rep("A+ tarzi (iki yonlu enterpolasyon + global medyan)", lab_b, zero_b, sc_b)
# en az 3 yil verisi olan ulkeler filtresi ile
nobs = raw.groupby("Country Code")[["gni_per_capita", "tertiary_enrollment", "unemployment_rate"]].transform("count").min(axis=1)
rep("A+ tarzi, her bilesende >=3 gozlemli ulkeler", lab_b.where(nobs >= 3), zero_b, sc_b)

open(os.path.join(OUT, "ek_sonuclar.txt"), "w", encoding="utf-8").write("\n".join(lines))
print("\nKaydedildi:", os.path.join(OUT, "ek_sonuclar.txt"))
