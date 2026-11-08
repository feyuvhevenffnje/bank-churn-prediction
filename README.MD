# 银行客户流失预测

基于 Kaggle `Churn_Modelling.csv`（10000 行），预测客户是否流失（Exited=1，流失率 20.37%）。

## 环境
Python 3.11.15 / conda pytorch_env。数据集自行下载放入 `data/raw/`，不随仓库分发。

## 复现
pip install -r requirements.txt
python src/load_to_db.py       # 阶段 2：建 bank.db
python src/run_sql.py          # 阶段 2：执行分析 SQL
python src/train.py            # 阶段 5：三模型对比
python src/model_selection.py  # 阶段 5：网格搜索
python src/evaluate.py         # 阶段 6：指标、图、阈值表，并保存模型
python src/demo_predict.py     # 阶段 7：单条预测
uvicorn api.main:app --reload --port 8000   # 阶段 8：/docs 测接口

## 关键结果
最优模型 RandomForest(min_samples_leaf=5)，测试集 ROC-AUC 0.8602、
Precision 0.8214、Recall 0.4521。按概率取 Top-100 命中 91 人。
阈值表见 reports/threshold_table.csv。

## 已知限制
- API 示例只传 5 个字段，其余由训练集中位数/众数补全，概率是条件概率。
- 每次请求都重新加载 joblib 模型，生产环境应启动时常驻内存。
- /docs 依赖 Swagger UI CDN，已替换为国内可访问地址。
