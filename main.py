from typing import Union
from fastapi import FastAPI, File, UploadFile, Request, Response, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from background_remover_class import U2NETPredictor
import numpy as np
import base64
import cv2
import time

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=['null'],
    allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

templates = Jinja2Templates(directory="static")

u2pred = U2NETPredictor()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/bg")
async def upload_root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
      
    output = u2pred.remove_bg(blur, 50)

    _, encoded_img = cv2.imencode('.PNG', output)
    #encoded_img = base64.b64encode(encoded_img)
    time.sleep(0.3)

    return Response(content=encoded_img.tobytes(), media_type="image/png")


from pydantic import BaseModel

@app.post("/uploadfile2/")
async def create_upload_file(
    file: UploadFile = File(...), 
    number: int = Form(...)
):
    contents = await file.read()
    number = number*(255/100)
    print(number)
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    output = u2pred.remove_bg(img, number)

    _, encoded_img = cv2.imencode('.PNG', output)
    #encoded_img = base64.b64encode(encoded_img)
    time.sleep(0.3)

    return Response(content=encoded_img.tobytes(), media_type="image/png")