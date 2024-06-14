from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
import pandas as pd

app = FastAPI()

model = joblib.load('decision_tree_model.pkl')

class InputData(BaseModel):
    input_1: float
    input_2: float

@app.get("/")
@app.get("/health_check")
async def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(item: InputData):
    data_dict = {"X1": item.input_1, "X2": item.input_2}
    data = pd.DataFrame([data_dict])
    print(data)

    prediction = model.predict(data)
    
    output = {'predictions': prediction.tolist()}
    return output

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


