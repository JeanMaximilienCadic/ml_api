from fastapi import FastAPI, Request
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow_hub as hub
import os
import numpy as np
import time
import tensorflow as tf
import torch
USE_URL = os.environ.get("USE_URL", 'https://tfhub.dev/google/universal-sentence-encoder/3')
NLLB_ID = os.environ.get("TOKENIZER_URL", "facebook/nllb-200-distilled-600M")
DEVICE = os.environ.get("DEVICE", "cpu" if not torch.cuda.is_available() else "cuda")

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(NLLB_ID, src_lang="ja", tgt_lang="en", device_map=DEVICE)
model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_ID).to(DEVICE)
model_embeddings = hub.load(USE_URL)

@app.post("/translate")
async def root(request: Request):
    # Read content
    input = await request.body()
    input = str(input.decode())
    output = None
    try:
        t0 = time.time()        
        batch = tokenizer(input, return_tensors="pt").to(DEVICE)
        gen = model.generate(**batch,forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"])
        output = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        elapsed = (time.time()-t0)*1000
        return {"input": input, "output": output, "error": None, "elapsed_ms": elapsed}
    except Exception as e:
        return {"input": str(input), "output":output, "error": str(e), "elapsed_ms": None}
    
    
@app.post("/embeddings/benchmark")
async def root(request: Request):
    # Read content
    input = await request.body()
    input = str(input.decode())
    def embed(input):
        return model_embeddings([input]*64)
    try:
        t0 = time.time()
        embed(input)
        elapsed = (time.time()-t0)*1000
        return {"error": None, "elapsed_ms": elapsed, "input": input}
    except Exception as e:
        return {"error": str(e), "elapsed_ms": None, "input": str(input)}
    
@app.post("/embeddings")
async def root(request: Request):
    # Read content
    input = await request.body()
    input = str(input.decode())
    def embed(input):
        return model_embeddings([input])
    output = None
    try:
        t0 = time.time()
        output = np.array(embed(input)).tolist()[0]
        elapsed = (time.time()-t0)*1000
        return {"input": input, "output": output, "error": None, "elapsed_ms": elapsed}
    except Exception as e:
        return {"input": str(input), "output":output, "error": str(e), "elapsed_ms": None}
