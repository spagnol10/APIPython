from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2
from PIL import Image
import io
import face_recognition
import os
import time
import logging
from typing import List
import numpy as np
from database import database  

# Configurando logging
logging.basicConfig(level=logging.INFO)

class ImgComp(BaseModel):
    img1: str
    img2: str

class ImgCad(BaseModel):
    img: str
    name: str
    cpf: str

class ImgRec(BaseModel):
    img: str

class Data(BaseModel):
    platform: str
    fotPar: str
    participantes: str

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

known_faces = []  

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

@app.get("/")
async def root():
    return {"message": "Hello World"}

class PersonInDB(BaseModel):
    name: str
    cpf: str
    encoding: list

def decode_and_process_image(image_data: str) -> Image:
    img_data = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(img_data))
    img.convert('RGB')
    return img

def encode_face(img: Image) -> list:
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    encoding = face_recognition.face_encodings(img_cv)
    if encoding:
        return encoding[0]
    raise HTTPException(status_code=400, detail="Nenhuma face encontrada")

@app.post("/CadastroImagem")
async def CadastroImagem(images: List[ImgCad]):
    for image in images:
        try:
            img = decode_and_process_image(image.img)
            img_encoding = encode_face(img)
            
            query = """
            INSERT INTO known_faces(name, cpf, encoding) 
            VALUES (:name, :cpf, :encoding)
            """
            await database.execute(query, values={
                "name": image.name,
                "cpf": image.cpf,
                "encoding": img_encoding.tolist()  
            })
            logging.info(f"Imagem de {image.name} cadastrada com sucesso.")
        except Exception as e:
            logging.error(f"Erro ao processar imagem de {image.name}: {e}")
            raise HTTPException(status_code=400, detail="Erro ao cadastrar imagem")
    return {"message": "Imagens cadastradas com sucesso"}

@app.post("/Reconhecimento")
async def Reconhecimento(image: ImgRec):
    img = decode_and_process_image(image.img)
    img_encoding = encode_face(img)
    
    if not known_faces:
        logging.error("Nenhuma face cadastrada no sistema.")
        raise HTTPException(status_code=404, detail="Nenhuma face cadastrada.")
    
    for person in known_faces:
        logging.info(f"Comparando com {person['name']}")
        if face_recognition.compare_faces([person["encoding"]], img_encoding)[0]:
            return {"message": "Pessoa encontrada", "name": person["name"], "cpf": person["cpf"]}
    
    return {"message": "Pessoa não encontrada"}


@app.post("/ComparaImagens")
async def ComparaImagens(image: ImgComp):
    img1 = decode_and_process_image(image.img1)
    img2 = decode_and_process_image(image.img2)
    
    img1_encoding = encode_face(img1)
    img2_encoding = encode_face(img2)
    
    if face_recognition.compare_faces([img1_encoding], img2_encoding)[0]:
        return {"message": "Mesma pessoa"}
    else:
        return {"message": "Não é a mesma pessoa"}

@app.post("/verifica-presenca")
async def VerificaPresenca(data: Data):
    try:
        start_time = time.time()
        
        img_participante1 = decode_and_process_image(data.fotPar)
        img_participante2 = decode_and_process_image(data.participantes)
        
        encoding1 = encode_face(img_participante1)
        encoding2 = encode_face(img_participante2)
        
        result = face_recognition.compare_faces([encoding1], encoding2)[0]
        elapsed_time = time.time() - start_time
        
        if result:
            return {"detail": {"codRet": 0, "msgRet": "Autenticado com sucesso", "tempo": elapsed_time}}
        else:
            return {"detail": {"codRet": 1, "msgRet": "Usuário não autenticado", "tempo": elapsed_time}}
    
    except Exception as e:
        logging.error(f"Erro ao verificar presença: {e}")
        raise HTTPException(status_code=400, detail="Erro ao verificar presença")
