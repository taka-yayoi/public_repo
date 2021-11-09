# Databricks notebook source
username = "brickster" # 共通のユーザーを使用

# データベース:SQL Analyticsから参照します
db_name_silver = f"dns_analytics_silver_{username}"
db_name_bronze = f"dns_analytics_bronze_{username}"

# データ格納場所
work_path = f"/tmp/{username}/dns"
default_file_path =  f'/dbfs{work_path}/tables/datasets'

print("default_file_path:", default_file_path)
print("db_name_silver:", db_name_silver)
print("db_name_bronze:", db_name_bronze)
