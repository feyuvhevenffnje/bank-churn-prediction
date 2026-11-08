from predict import predict_one

sample = {
    "Age": 42,
    "Geography": "France",
    "Balance": 15000,
    "NumOfProducts": 2,
    "IsActiveMember": 1,
}

r = predict_one(sample)
print("实际送入模型的字段:", r["used_fields"])
print("预测结果:", "可能流失" if r["prediction"] else "不太可能流失")
print(f"流失概率: {r['churn_probability'] * 100:.1f}%")

print("\n--- 未知地区容错测试 ---")
r2 = predict_one({**sample, "Geography": "Italy"})
print("预测概率:", r2["churn_probability"])
