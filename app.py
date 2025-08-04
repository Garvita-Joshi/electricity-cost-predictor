# importing the necessary libraires
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import uvicorn

# Initializing FastAPI and Jinja2
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Loading the trained pipeline (this has been extracted from model.py)
model = joblib.load("model.pkl")

# using this decorator for defining the home route
# and is requested to return a HTML response
@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

@app.post("/predict", response_class=HTMLResponse)
# here we are defining the predict function which will be defining the columns which 
# will are being used to predict our target variable 
async def predict(
    # Request is used to handle the incoming request from the user
    request: Request, 
    # Using Form to accept input data from the HTML form
    site_area: float = Form(...),
    structure_type: str = Form(...),
    water_consumption: float = Form(...),
    recycling_rate: float = Form(...),
    utilisation_rate: float = Form(...),
    air_qality_index: float = Form(...),  
    issue_reolution_time: float = Form(...), 
    resident_count: float = Form(...)
):
    try:
        # Creating a DataFrame from the input data
        input_data = pd.DataFrame([{
            "site area": site_area,
            "structure type": structure_type,
            "water consumption": water_consumption,
            "recycling rate": recycling_rate,
            "utilisation rate": utilisation_rate,
            "air qality index": air_qality_index,  
            "issue reolution time": issue_reolution_time,  
            "resident count": resident_count,
        }])

        # Prediction
        # Using the loaded model to make a prediction
        prediction = model.predict(input_data)[0]
        # this ([0]) has been used since it is defined as a 2D array and we need the first element only 

        # this time since we have suffiecent data to predict the target variable
        # we will be returning the some prediction in the HTML response
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": f"Predicted Electricity Cost: {prediction:.2f}"
        })
    # Handling any error which may occur either while entering the data 
    # or while predicting the target variable
    except Exception as error:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": f"Error: {str(error)}"
        })

