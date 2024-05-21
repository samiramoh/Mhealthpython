import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel,ValidationError

from datetime import date, datetime
import processing,processing_brain,processing_cardio
import numpy as np
from processing import create_dataframe,encoded_scaled,model,message
app = FastAPI()

class HealthData(BaseModel):
    user_stress_level: int
    weight: float
    height: float
    heart_rate: int
    systolic: float
    diastolic: float
    physical_activity_minutes: int
    steps: int
    sleep_duration: str
    age : int
    gender :bool
    alcohol : bool
    Glucose : str
    smoking_status:str
    Heart_disease:bool
    chol :str

@app.post("/submit_health_data/")
async def submit_health_data(data: HealthData):
    try:
        stress=int(data.user_stress_level)
        weightt=float(data.weight)
        heightt=float(data.height)
        heart=int(data.heart_rate)
        sys=float(data.systolic)
        dia=float(data.diastolic)
        phy=int(data.physical_activity_minutes)
        step=int(data.steps)
        agee=int(data.age)
        sleep=str(data.sleep_duration)
        gen=bool(data.gender)
        dis=bool(data.Heart_disease)
        smok=str(data.smoking_status)
        gluc=str(data.Glucose)
        alc=bool(data.alcohol)
        chol=str(data.chol)
        sleep = processing.create_dataframe(stress, weightt, heightt, heart,
                     sys, dia, phy,
                     step, sleep, agee, gen)
        brain = processing_brain.create_dataframe(weightt, heightt,
                     sys, dia, smok.lower(),
                     dis, gluc, agee, gen)
        cardio = processing_cardio.create_dataframe(gen,weightt, heightt,chol,
                     sys, dia,phy,gluc, smok.lower(),agee, alc)
        sleep = processing.encoded_scaled(sleep)
        pred = processing.model(sleep)
        messages = processing.message(pred)
        brain = processing_brain.encoded_scaled(brain)
        pred_brain = processing_brain.model(brain)
        messages_brain = processing_brain.message(pred_brain)
        cardio = processing_cardio.encoded_scaled(cardio)
        cardio=processing_cardio.custom_label_encoder(cardio)
        cardio=processing_cardio.converter(cardio)
        pred_cardio = processing_cardio.model(cardio)
        messages_cardio = processing_cardio.message(pred_cardio)
        print(messages_cardio)
        msg='All Good!'
        if messages=='All Good!':
            if messages_brain=='You are experiencing symptoms of Brain Stroke, please visit the nearest hospital or call emergency services immediately':
                if messages_cardio=='You are experiencing symptoms of Cardio Vascular Disease, please visit the nearest hospital or call emergency services immediately':
                    msg='You are experiencing symptoms of Cardio Vascular Disease and Brain Stroke, please visit the nearest hospital or call emergency services immediately'
                else:
                    msg=messages_brain
            elif messages_brain=='All Good!':
                if messages_cardio=='You are experiencing symptoms of Cardio Vascular Disease, please visit the nearest hospital or call emergency services immediately':
                    msg=messages_cardio
            else:
                msg =messages
        elif messages =='You are experiencing symptoms of Insomnia, please visit the nearest hospital or call emergency services immediately':
            if messages_brain=='You are experiencing symptoms of Brain Stroke, please visit the nearest hospital or call emergency services immediately':
                if messages_cardio=='You are experiencing symptoms of Cardio Vascular Disease, please visit the nearest hospital or call emergency services immediately':
                    msg='You are experiencing symptoms of Cardio Vascular Disease and Brain Stroke and Insomnia , please visit the nearest hospital or call emergency services immediately'
                else:
                    msg='You are experiencing symptoms of Brain Stroke and Insomnia , please visit the nearest hospital or call emergency services immediately'
            elif messages_brain=='All Good!':
                msg='You are experiencing symptoms of  Insomnia , please visit the nearest hospital or call emergency services immediately'
            else:
                msg= messages
        elif messages=='You are experiencing symptoms of Sleep Apnea, please visit the nearest hospital or call emergency services immediately':
            if messages_brain=='You are experiencing symptoms of Brain Stroke, please visit the nearest hospital or call emergency services immediately':
                if messages_cardio=='You are experiencing symptoms of Cardio Vascular Disease, please visit the nearest hospital or call emergency services immediately':
                    msg='You are experiencing symptoms of Cardio Vascular Disease and Brain Stroke and Sleep Apnea , please visit the nearest hospital or call emergency services immediately'
                else:
                    msg='You are experiencing symptoms of  Brain Stroke and Sleep Apnea , please visit the nearest hospital or call emergency services immediately'
            elif messages_brain=='All Good!':
                msg='You are experiencing symptoms of Sleep Apnea , please visit the nearest hospital or call emergency services immediately'
            else:
                msg= messages
        return str(msg)
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e), "context": "Error in processing data"})
''' df=create_dataframe(data)
    df=encoded_scaled(df)
    pred=model(df)
    messages=message(pred.item())'''
   #return {"message": data}

'''try:
        # Your logic here
        return {"message": str(type(data.gender))}
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))'''

@app.get("/test/")
async def test_endpoint():
    return {"message": "Server is up and running!"}
