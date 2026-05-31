from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

from src.predict import Predictor

CHECKPOINT_PATH = Path("models/checkpoints/best.pth")
predictor: Predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    if CHECKPOINT_PATH.exists():
        predictor = Predictor(str(CHECKPOINT_PATH))
        print(f"모델 로드 완료: {CHECKPOINT_PATH}")
    else:
        print(f"[경고] 체크포인트 없음: {CHECKPOINT_PATH} — 학습 후 실행하세요.")
    yield


app = FastAPI(
    title="Trash-Smart API",
    description="컴퓨터 비전 기반 한국형 재활용 분류 가이드",
    version="1.0.0",
    lifespan=lifespan,
)

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

    if predictor is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다. 먼저 학습을 완료하세요.")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="파일 크기는 10MB 이하여야 합니다.")

    try:
        results = predictor.predict(contents, top_k=3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")

    return {"predictions": results, "filename": file.filename}


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.get("/classes")
async def get_classes():
    import json
    with open("data/classes.json", encoding="utf-8") as f:
        return json.load(f)
