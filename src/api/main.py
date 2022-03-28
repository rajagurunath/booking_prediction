# File name: summarizer_on_ray_serve.py
import ray
from ray import serve
import pickle
from fastapi import FastAPI
import os.path
import sys
import json
from sklearn.preprocessing import LabelEncoder
app = FastAPI()

from pydantic import BaseModel
import pandas as pd
from typing import Dict,List
import importlib
import os
ray.init(address="127.0.0.1:6379", namespace="serve")
serve.start(detached=True)

MODEL_EXPORT_PATH = os.environ['MODEL_EXPORT_PATH']

# FACED SOME IMPORT ISSUE so made everything as single file -DEBUG HERE
# reason:  Worker didnt finding the required models if placed separately - need more exploration
# So added all the dependencies classes in single file :)

class Encoder(object):
    def __init__(self,cat_cols=None,path="../../../models/") -> None:
        self.cat_cols = cat_cols
        self.encoders = {}
        self.path = path

    def fit(self,df):

        for col in self.cat_cols:
            print(f"Label encoding - {col}")
            _enc = LabelEncoder()
            _enc.fit_transform(df[col])
            self.encoders[col] = _enc
        return self.encoders

    def transform(self,df:pd.DataFrame):
        if len(self.encoders) <=0:
            print("Encoder Not fitted")
            return 
        for col in self.cat_cols:
            print(f"Label transforming - {col}")
            df[col]=self.encoders[col].transform(df[col])
        return df

    def save(self):
        print("saving ...")
        print(self.path)
        pickle.dump(self.encoders,open(self.path+"encoders.pkl","wb"))
        pickle.dump(self.cat_cols,open(self.path+"encoder_meta","wb"))

    def load(self):
        print("Loading ...")
        self.encoders = pickle.load(open(self.path+"encoders.pkl","rb"))
        self.cat_cols = pickle.load(open(self.path+"encoder_meta","rb"))
        return self.encoders,self.cat_cols


class Predictor(object):
    def __init__(self,models:List=None,path="../../../models/") -> None:
        self.models = models # 1 or 3 models
        self.predictor= {}
        self.path = path

    def fit(self,X,y):
        for model in self.models:
            print("Fitting ...")
            print(model.__name__)
            self.predictor[model.__name__] = model().fit(X,y)
        return self.predictor

    def predict(self,X):
        if len(self.predictor) <=0:
            print("Predictor Not fitted")
            return 
        pred_df = {}
        for model in self.models:
            print(f"predicting with {model.__name__}")
            pred_df[str(model.__name__)] = self.predictor[str(model.__name__)].predict(X).tolist()
        return pred_df

    def save(self):
        print("Saving the model...")
        pickle.dump(self.predictor,open(self.path+"model.pkl","wb"))
        pickle.dump(self.models,open(self.path+"model_meta","wb"))

    def load(self):
        print("Loading the model ...")
        self.predictor = pickle.load(open(self.path+"model.pkl","rb"))
        self.models = pickle.load(open(self.path+"model_meta","rb"))
        return self.predictor


class Booking(BaseModel):
    """
    for request schema validation
    """
    hotel: str
    lead_time: int
    arrival_date_year: int
    arrival_date_month: str
    arrival_date_week_number: int
    arrival_date_day_of_month: int
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    adults: int
    children: int
    babies: int
    meal: str
    country: str
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    reserved_room_type: str
    assigned_room_type: str
    booking_changes: int
    deposit_type: str
    agent: float
    company: float
    days_in_waiting_list: int
    customer_type: str
    adr: float
    required_car_parking_spaces: int
    total_of_special_requests: int
    reservation_status: str
    reservation_status_date: str
    name: str
    email: str
    phonenumber: str
    credit_card: str


@app.get("/")
def f():
    return "Health check"

#Actor 1
@serve.deployment
class ProdEncoder:
    DONT_CONSIDER = ['reservation_status_date',"name","email","phonenumber","credit_card"]

    def __init__(self) -> None:
        # from ml import Encoder
        # pp = importlib.import_module("preprocess")

        self.encoder = Encoder(path = MODEL_EXPORT_PATH)
        self.encoder.load()
    
    def __call__(self,X:List[Dict])->pd.DataFrame:
        
        X = pd.DataFrame(X).drop(self.DONT_CONSIDER,axis=1)
        # pickle.dump(X,open("debug.pkl","wb"))
        encX = self.encoder.transform(X)
        return encX



# Actor 2
@serve.deployment
class ProdPredictor:
    def __init__(self) -> None:
        # from ml import Predictor
        # train = importlib.import_module("train.py")

        self.predictor = Predictor(path = MODEL_EXPORT_PATH)
        self.predictor.load()

    def __call__(self,X:pd.DataFrame)->Dict:
        preds = self.predictor.predict(X)
        return preds



# Actor 3 /ingress Actor
@serve.deployment(route_prefix="/model")
@serve.ingress(app)
class HotelCancellationPreds:
    def __init__(self):
        self.encoder = ProdEncoder.get_handle()
        self.model = ProdPredictor.get_handle()

    def _validate_schema(self):
        """
        Done by pydantic 
        """
        ...

    @app.post("/booking_cancel_prediction")
    async def hotel_cancellation_prediction(self,booking_details:List[Booking]): # validates schema
        # print(booking_details)
        # booking_details = pd.DataFrame(booking_details)
        preproceed_data = await self.encoder.remote([bd.__dict__ for bd in booking_details])
        preds = await self.model.remote(preproceed_data)
        return preds
        # txt = request.query_params["booking_params"]
        # encoded_row = self.encoder(txt)
        # preds = self.model(encoded_row)
        # return preds

ProdEncoder.deploy()
ProdPredictor.deploy()
HotelCancellationPreds.deploy()
