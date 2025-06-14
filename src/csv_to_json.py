# csv_to_json.py
import pandas as pd, json, pathlib, argparse, datetime as dt

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="data/re_clean.csv")
parser.add_argument("--out", default="data/full_new.json")
args = parser.parse_args()

df = pd.read_csv(args.csv)

records = []
for _, r in df.iterrows():
    # 1.  Keep the long description as search text
    page_content = str(r["doc"])

    # 2.  Build clean metadata – keys must match seed_data.py
    meta = {
        "title":           r["title"],
        "date_posted":     pd.to_datetime(r["date"]).strftime("%Y-%m-%d"),
        "price_vnd":       float(r["price"]),
        "area":            float(r["area_m2"]),
        "price_per_area":  float(r["price_per_m2"]),
        "bedrooms":        int(r["bedrooms"]),
        "toilets":         int(r["toilets"]),
        "direction":       r["direction"],
        "district_county": r["district_county"],
        "province_city":   r["province_city"],
        "url":             r["url"],
    }

    records.append({"page_content": page_content, "metadata": meta})

# 3.  Save
pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"✅  Saved {len(records)} rows to {args.out}")
