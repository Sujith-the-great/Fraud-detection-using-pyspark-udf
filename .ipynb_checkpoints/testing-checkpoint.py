from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.sql.types import BooleanType
from pyspark.sql.functions import pandas_udf
from functools import reduce
import time
import gc

from models.cnn import CNN

import torch


import base64
import io
from PIL import Image
import numpy as np
import pandas as pd
import csv

from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Paths to your models
MODEL_PATH_PYTORCH = "models/fraud_cnn.pt"


spark = SparkSession.builder.appName("FraudModelInference_Tables").getOrCreate()


idimage_df = spark.read.option("header", True).csv("data/idimage_fixed.csv")
idlabel_df = spark.read.option("header", True).csv("data/idlabel.csv") \
                       .withColumn("isfraud", col("isfraud").cast(BooleanType()))
idmeta_df  = spark.read.option("header", True).csv("data/idmeta.csv")


idimage_df.createOrReplaceTempView("idimage_original")
idlabel_df.createOrReplaceTempView("idlabel_original")
idmeta_df.createOrReplaceTempView("idmeta_original")


def duplicate_df(df, n):
    dfs = [df.withColumn("dup_batch", lit(i)) for i in range(n)]
    return reduce(lambda a, b: a.unionByName(b), dfs)


scales = [("original", 1), ("5x", 5), ("10x", 10), ("20x", 20), ("50x", 50), ("100x", 100)]
for name, factor in scales:
    if factor == 1:
        img_df, lbl_df, meta_df = idimage_df, idlabel_df, idmeta_df
    else:
        img_df  = duplicate_df(idimage_df, factor)
        lbl_df  = duplicate_df(idlabel_df, factor)
        meta_df = duplicate_df(idmeta_df, factor)
    img_df.createOrReplaceTempView(f"idimage_{name}")
    lbl_df.createOrReplaceTempView(f"idlabel_{name}")
    meta_df.createOrReplaceTempView(f"idmeta_{name}")


try:
    model = CNN()
    model.load_state_dict(torch.load(MODEL_PATH_PYTORCH, map_location=torch.device('cpu')))
    model.eval()
    broadcast_model = spark.sparkContext.broadcast(model)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    broadcast_model = spark.sparkContext.broadcast(None)


@pandas_udf(BooleanType())
def cnn_fraud_detector(image_col: pd.Series) -> pd.Series:
    mdl = broadcast_model.value
    results = []
    for base64_str in image_col:
        if mdl is None:
            results.append(False)
            continue
        try:
            img_bytes = base64.b64decode(base64_str)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((128,128))
            arr = np.array(img) / 255.0
            arr = np.expand_dims(arr, 0).transpose((0,3,1,2))
            tensor = torch.from_numpy(arr).float()
            with torch.no_grad():
                out = mdl(tensor)
            results.append(bool(out[0][0] > 0.5))
        except Exception:
            results.append(False)
    return pd.Series(results)


spark.udf.register("cnn_fraud_detector_sql", cnn_fraud_detector)


queries = {
    "fraud_predicted": """
        SELECT
            COUNT(*) AS total_ids,
            SUM(CASE WHEN cnn_fraud_detector_sql(imageData) THEN 1 ELSE 0 END) AS fraud_predicted,
            100.0 * SUM(CASE WHEN cnn_fraud_detector_sql(imageData) THEN 1 ELSE 0 END) / COUNT(*) AS fraud_rate_percentage
        FROM {image_table}
    """,

    "fraud_ground_truth": """
        SELECT
            COUNT(*) AS total_ids,
            SUM(CASE WHEN isfraud THEN 1 ELSE 0 END) AS total_fraud,
            SUM(CASE WHEN NOT isfraud THEN 1 ELSE 0 END) AS total_nonfraud,
            100.0 * SUM(CASE WHEN isfraud THEN 1 ELSE 0 END) / COUNT(*) AS fraud_percentage
        FROM {label_table}
    """,

    "fraud_pattern_ethnicity_wise": """
        SELECT
            m.ethnicity,
            COUNT(*) AS total_customers,
            SUM(CASE WHEN cnn_fraud_detector_sql(i.imageData) THEN 1 ELSE 0 END) AS predicted_fraud,
            100.0 * SUM(CASE WHEN cnn_fraud_detector_sql(i.imageData) THEN 1 ELSE 0 END) / COUNT(*) AS fraud_rate_pct
        FROM {image_table} i
        JOIN {label_table} l
          ON i.name = l.id
        JOIN {meta_table} m
          ON l.id = m.id
        GROUP BY m.ethnicity
        ORDER BY fraud_rate_pct DESC
        LIMIT 10
    """,

    "fraud_rate_by_veteran_status": """
        SELECT
            m.is_veteran,
            COUNT(*) AS total_individuals,
            SUM(CASE WHEN cnn_fraud_detector_sql(i.imageData) THEN 1 ELSE 0 END) AS predicted_fraud,
            100.0 * SUM(CASE WHEN cnn_fraud_detector_sql(i.imageData) THEN 1 ELSE 0 END) / COUNT(*) AS fraud_rate_pct
        FROM {image_table} i
        JOIN {label_table} l
          ON i.name = l.id
        JOIN {meta_table} m
          ON l.id = m.id
        GROUP BY m.is_veteran
        ORDER BY fraud_rate_pct DESC
        LIMIT 10
    """
}


timings = {name: [] for name in queries}
for scale_name, _ in scales:
    img_tbl  = f"idimage_{scale_name}"
    lbl_tbl  = f"idlabel_{scale_name}"
    meta_tbl = f"idmeta_{scale_name}"

    for qname, qsql in queries.items():
        # format in the three table names
        sql = qsql.format(
            image_table=img_tbl,
            label_table=lbl_tbl,
            meta_table=meta_tbl
        )
        start = time.time()
        spark.sql(sql).collect()
        timings[qname].append(time.time() - start)

    print(f"==== {scale_name} done ====")
    spark.catalog.clearCache()
    gc.collect()
    spark.sparkContext._jvm.java.lang.System.gc()


scale_labels = [s[0] for s in scales]
x_positions = list(range(len(scale_labels))) 


for qname, times in timings.items():
    plt.figure(figsize=(8, 5))
    plt.plot(x_positions, times, marker='o', color='tab:blue', linewidth=2)
    plt.xlabel('Data Scale')
    plt.ylabel('Execution Time (s)')
    plt.title(f'Execution Time vs Scale for: {qname}')
    plt.grid(which='both', ls='--', linewidth=0.5)
    plt.xticks(ticks=range(0, 101, 10))
    plt.tight_layout()
    plt.savefig(f"{qname}.png")  
    plt.close()  


# 12) Print table
header = ["Query"] + [s[0] for s in scales]
rows = []
for qname, times in timings.items():
    rows.append([qname] + [f"{t:.2f}" for t in times])

print("\n=== Query Execution Times (seconds) ===")
print(tabulate(rows, headers=header, tablefmt="github"))

# =================== USING PANDAS and SAME QUERIES TO COMPARE RESULTS ===================
def predict_fraud(image_b64):
    try:
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((128, 128))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, 0).transpose((0, 3, 1, 2))
        tensor = torch.from_numpy(arr).float()
        with torch.no_grad():
            out = model(tensor)
        return bool(out[0][0] > 0.5)
    except Exception:
        return False


idimage_df = pd.read_csv("data/idimage_fixed.csv")
idlabel_df = pd.read_csv("data/idlabel.csv")
idmeta_df  = pd.read_csv("data/idmeta.csv")


idlabel_df['isfraud'] = idlabel_df['isfraud'].astype(bool)


def fraud_predicted(image_df):
    image_df = image_df.copy()
    image_df['predicted_fraud'] = image_df['imageData'].apply(predict_fraud)
    total_ids = len(image_df)
    fraud_predicted = image_df['predicted_fraud'].sum()
    fraud_rate_percentage = 100.0 * fraud_predicted / total_ids
    return {
        'total_ids': total_ids,
        'fraud_predicted': fraud_predicted,
        'fraud_rate_percentage': fraud_rate_percentage
    }

def fraud_ground_truth(label_df):
    total_ids = len(label_df)
    total_fraud = label_df['isfraud'].sum()
    total_nonfraud = total_ids - total_fraud
    fraud_percentage = 100.0 * total_fraud / total_ids
    return {
        'total_ids': total_ids,
        'total_fraud': total_fraud,
        'total_nonfraud': total_nonfraud,
        'fraud_percentage': fraud_percentage
    }

def fraud_pattern_ethnicity_wise(image_df, label_df, meta_df):
    image_df = image_df.copy()
    image_df['predicted_fraud'] = image_df['imageData'].apply(predict_fraud)
    merged_df = image_df.merge(label_df, left_on='name', right_on='id')
    merged_df = merged_df.merge(meta_df, on='id')
    group = merged_df.groupby('ethnicity').agg(
        total_customers=('id', 'count'),
        predicted_fraud=('predicted_fraud', 'sum')
    ).reset_index()
    group['fraud_rate_pct'] = 100.0 * group['predicted_fraud'] / group['total_customers']
    return group.sort_values('fraud_rate_pct', ascending=False).head(10)

def fraud_rate_by_veteran_status(image_df, label_df, meta_df):
    image_df = image_df.copy()
    image_df['predicted_fraud'] = image_df['imageData'].apply(predict_fraud)
    merged_df = image_df.merge(label_df, left_on='name', right_on='id')
    merged_df = merged_df.merge(meta_df, on='id')
    group = merged_df.groupby('is_veteran').agg(
        total_individuals=('id', 'count'),
        predicted_fraud=('predicted_fraud', 'sum')
    ).reset_index()
    group['fraud_rate_pct'] = 100.0 * group['predicted_fraud'] / group['total_individuals']
    return group.sort_values('fraud_rate_pct', ascending=False).head(10)





timings = {
    'fraud_predicted': {'spark': [], 'pandas': []},
    'fraud_ground_truth': {'spark': [], 'pandas': []},
    'fraud_pattern_ethnicity_wise': {'spark': [], 'pandas': []},
    'fraud_rate_by_veteran_status': {'spark': [], 'pandas': []}
}

for scale_name, factor in scales:
    print(f"=== {scale_name} done ===")
    

    img_df_scaled = pd.concat([idimage_df] * factor, ignore_index=True)
    lbl_df_scaled = pd.concat([idlabel_df] * factor, ignore_index=True)
    meta_df_scaled = pd.concat([idmeta_df] * factor, ignore_index=True)
    

    start = time.time()
    fraud_predicted(img_df_scaled)
    timings['fraud_predicted']['pandas'].append(time.time() - start)
    
    start = time.time()
    fraud_ground_truth(lbl_df_scaled)
    timings['fraud_ground_truth']['pandas'].append(time.time() - start)
    
    start = time.time()
    fraud_pattern_ethnicity_wise(img_df_scaled, lbl_df_scaled, meta_df_scaled)
    timings['fraud_pattern_ethnicity_wise']['pandas'].append(time.time() - start)
    
    start = time.time()
    fraud_rate_by_veteran_status(img_df_scaled, lbl_df_scaled, meta_df_scaled)
    timings['fraud_rate_by_veteran_status']['pandas'].append(time.time() - start)
    
    
    start = time.time()
    timings['fraud_predicted']['spark'].append(time.time() - start)
    
    start = time.time()
    timings['fraud_ground_truth']['spark'].append(time.time() - start)
    
    start = time.time()
    timings['fraud_pattern_ethnicity_wise']['spark'].append(time.time() - start)
    
    start = time.time()
    timings['fraud_rate_by_veteran_status']['spark'].append(time.time() - start)
    
    gc.collect()

scale_labels = [s[0] for s in scales]

for query_name in timings:
    plt.figure(figsize=(10, 6))
    # plt.plot(x_positions, timings[query_name]['spark'], marker='o', label='Spark + UDF')
    plt.plot(x_positions, timings[query_name]['pandas'], marker='s', label='Pandas')
    plt.xlabel('Data Scale')
    plt.ylabel('Execution Time (s)')
    plt.title(f'Execution Time vs Data Scale for {query_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{query_name}_pandas.png")
    plt.close()



# scale_labels = [s[0] for s in scales]

# for qname in timings:
#     plt.figure(figsize=(8, 5))

#     plt.plot(x_positions, timings[qname]['spark'], marker='o', color='tab:blue', linewidth=2, label='Spark + UDF')
    
#     plt.plot(x_positions, timings[qname]['pandas'], marker='s', color='tab:green', linewidth=2, label='Pandas')

#     plt.xlabel('Data Scale')
#     plt.ylabel('Execution Time (s)')
#     plt.title(f'Execution Time vs Scale for: {qname}')
#     plt.yscale('log')
#     plt.grid(which='both', ls='--', linewidth=0.5)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f"{qname}_combined.png")  
#     plt.close()
