import pandas as pd
import os

def load_cardio(path="data/cardio_train.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Not found: {path}")
    df = pd.read_csv(path, sep=";")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=",")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=None, engine="python")
    print("\n" + "=" * 60)
    print("  DATASET 1: CARDIOVASCULAR DISEASE (Ulianova, 2019)")
    print("=" * 60)
    print(f"  Shape          : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Columns        : {list(df.columns)}")
    print(f"  Missing values : {df.isnull().sum().sum()}")
    if "cardio" in df.columns:
        print(f"  Target (cardio): {df['cardio'].value_counts().to_dict()}")
    else:
        print(f"  WARNING: cardio column not found. Columns: {list(df.columns)}")
    print("=" * 60)
    return df

def load_uci(path="data/heart_disease_cleveland.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Not found: {path}")
    standard_columns = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal", "target"
    ]
    with open(path, "r") as f:
        first_line = f.readline().strip()
    first_val = first_line.split(",")[0].strip()
    def is_numeric(val):
        try:
            float(val)
            return True
        except:
            return False
    has_header = not is_numeric(first_val)
    try:
        if has_header:
            df = pd.read_csv(path, na_values=["?", ""])
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            for alt in ["num", "condition", "heart_disease", "disease", "output", "class", "label"]:
                if alt in df.columns:
                    df.rename(columns={alt: "target"}, inplace=True)
                    break
            if "target" not in df.columns:
                df.rename(columns={df.columns[-1]: "target"}, inplace=True)
        else:
            df = pd.read_csv(path, header=None, names=standard_columns, na_values=["?", ""])
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", header=None, names=standard_columns, na_values=["?", ""])
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    if "ca" in df.columns:
        df["ca"] = pd.to_numeric(df["ca"], errors="coerce")
    if "thal" in df.columns:
        df["thal"] = pd.to_numeric(df["thal"], errors="coerce")
    df["target"] = (df["target"] > 0).astype(int)
    print("\n" + "=" * 60)
    print("  DATASET 2: UCI HEART DISEASE - CLEVELAND (Detrano et al., 1989)")
    print(f"  File: {path}")
    print("=" * 60)
    print(f"  Shape          : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Columns        : {list(df.columns)}")
    print(f"  Missing values : {df.isnull().sum().sum()}")
    print(f"  Target dist.   : {df['target'].value_counts().to_dict()}")
    print("=" * 60)
    return df

if __name__ == "__main__":
    print("\n[INFO] Testing data_loader.py...")
    print("\n[TEST 1] Loading Dataset 1 (Cardio)...")
    df1 = load_cardio()
    print(f"  Sample: {df1.iloc[0].to_dict()}")
    print("\n[TEST 2] Loading Dataset 2 (UCI)...")
    df2 = load_uci()
    print(f"  Sample: {df2.iloc[0].to_dict()}")
    print("\n[DONE] Both datasets loaded successfully.")
