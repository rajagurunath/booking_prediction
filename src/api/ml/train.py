from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from typing import List
import pandas as pd
import numpy as np
import pickle
import json



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
            pred_df[str(model.__name__)] = self.predictor[str(model.__name__)].predict(X)
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


if __name__ == "__main__":
    # testing 
    from preprocess import Encoder
    import os
    EXPORT_PATH = os.environ['MODEL_EXPORT_PATH']
    DONT_CONSIDER = ['reservation_status_date',"name","email","phone-number","credit_card"]

    df = pd.read_csv("../../../data/hotel_booking.csv")
    df = df.fillna(df.mode().iloc[0]).iloc[:100,:] # ideally handled by encoders
    df= df.drop(DONT_CONSIDER,axis=1)
    encoder = Encoder(path=EXPORT_PATH)
    encoder.load()
    enc_df = encoder.transform(df)

    X = df.drop("is_canceled",axis=1)
    y = df['is_canceled']
    predictor = Predictor(models=[RandomForestClassifier,GradientBoostingClassifier,KNeighborsClassifier])
    predictor.fit(X,y)
    predictor.save()

    predictor = Predictor()
    predictor.load()
    preds = predictor.predict(X.iloc[:5,:])
    a = pd.DataFrame(preds).to_dict("records")
    print(json.dumps(a))

    
