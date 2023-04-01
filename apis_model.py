from fastapi import FastAPI, File, UploadFile
from typing import Optional
from prediction_new import model_prediction
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# Set up CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Import your pre-trained model here
# Example:
# from my_model import predict

# Define your prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Optional[str]:
    try:
        # Save the uploaded file to disk temporarily
        with open(file.filename, "wb") as buffer:
            buffer.write(await file.read())

        # Call your prediction function with the file as input
        result = model_prediction(file.filename)

        # Return the prediction result as a string
        return JSONResponse(content={"result": result})

    except Exception as e:
        # If there's an error, return "fake" as a response
        return "fake"

