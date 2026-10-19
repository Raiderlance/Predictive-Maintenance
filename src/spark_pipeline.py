"""Optional PySpark pipeline demonstrating scalable Bronze -> Silver processing."""
from __future__ import annotations

import argparse
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/telemetry.csv")
    parser.add_argument("--output", default="data/processed/silver_telemetry")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("predictive-maintenance-silver").getOrCreate()
    df = spark.read.option("header", True).option("inferSchema", True).csv(args.input)
    df = df.withColumn("timestamp", F.to_timestamp("timestamp"))

    w24 = Window.partitionBy("machine_id").orderBy(F.col("timestamp").cast("long")).rangeBetween(-23 * 3600, 0)
    silver = (
        df.withColumn("temperature_roll24_mean", F.avg("temperature_c").over(w24))
          .withColumn("vibration_roll24_mean", F.avg("vibration_mm_s").over(w24))
          .withColumn("power_roll24_mean", F.avg("power_kw").over(w24))
          .dropDuplicates(["machine_id", "timestamp"])
    )
    silver.write.mode("overwrite").parquet(args.output)
    spark.stop()


if __name__ == "__main__":
    main()
