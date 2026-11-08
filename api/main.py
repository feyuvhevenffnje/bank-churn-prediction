import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import CustomerInput, PredictionOutput
from predict import predict_one

MODEL_PATH = ROOT / "models" / "churn_model.joblib"

# docs_url=None / redoc_url=None：关闭 FastAPI 默认文档页。
# 默认文档页从 cdn.jsdelivr.net 拉 Swagger UI 的 JS/CSS，国内网络常被墙，
# 表现就是浏览器打开 /docs 一片空白。这里改用项目内 static/ 下的本地资源。
app = FastAPI(title="银行客户流失预测服务", version="1.0",
              docs_url=None, redoc_url=None)

STATIC_DIR = ROOT / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 刻意不引用 SwaggerUIStandalonePreset：swagger-ui-bundle.js 里并不含该对象，
# 引用到 undefined 会让整个页面渲染失败（白屏）。
DOCS_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>银行客户流失预测服务 - API 文档</title>
  <link rel="stylesheet" href="/static/swagger-ui/swagger-ui.css">
  <style>
    body { margin: 0; background: #fafafa; }
    .local-tip {
      font: 13px/1.6 -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
      background: #e8f4ff; color: #0b4a7a;
      border-bottom: 1px solid #b6dbff;
      padding: 8px 16px;
    }
    .local-tip code { background: #fff; padding: 1px 5px; border-radius: 3px; }
  </style>
</head>
<body>
  <div class="local-tip">
    本地离线文档（不依赖外网 CDN）。测接口：展开 <code>POST /predict</code> &rarr;
    <b>Try it out</b> &rarr; 改请求体 &rarr; <b>Execute</b>。OpenAPI 描述见
    <a href="/openapi.json">/openapi.json</a>。
  </div>
  <div id="swagger-ui"></div>
  <script src="/static/swagger-ui/swagger-ui-bundle.js"></script>
  <script>
    window.ui = SwaggerUIBundle({
      url: "/openapi.json",
      dom_id: "#swagger-ui",
      layout: "BaseLayout",
      deepLinking: true,
      presets: [SwaggerUIBundle.presets.apis],
      displayRequestDuration: true
    });
  </script>
</body>
</html>
"""


@app.get("/", include_in_schema=False)
def root():
    return {"service": "bank-churn-prediction", "status": "running", "docs": "/docs"}


@app.get("/docs", include_in_schema=False)
def docs():
    return HTMLResponse(DOCS_HTML)


@app.get("/health")
def health():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="模型文件缺失")
    return {"status": "healthy", "model_exists": True}


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    try:
        payload = customer.model_dump(exclude_none=True)
        r = predict_one(payload)
        return {"prediction": r["prediction"],
                "churn_probability": r["churn_probability"]}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"预测失败: {type(e).__name__}: {e}")
