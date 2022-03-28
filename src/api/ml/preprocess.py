from cmath import pi
from sklearn.preprocessing import LabelEncoder
import pickle,os
import pandas as pd

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


if __name__ == "__main__":
    # testing

    #remove this
    # "reservation_status_date":"2015-07-02",
    # "name":"Rebecca Parker",
    # "email":"Rebecca_Parker@comcast.net",
    # "phonenumber":"652-885-2745",
    # "credit_card":"************3734"
    EXPORT_PATH = os.environ['MODEL_EXPORT_PATH']
    DONT_CONSIDER = ['reservation_status_date',"name","email","phone-number","credit_card"]
    df = pd.read_csv("../../../data/hotel_booking.csv")
    cat_cols = df.select_dtypes("object").columns
    cat_cols = [col for col in cat_cols if col not in DONT_CONSIDER ]
    print(cat_cols)
    encoder = Encoder(cat_cols=cat_cols,path=EXPORT_PATH)
    encoder.fit(df)
    encoder.save()
    encoder = Encoder(path=EXPORT_PATH)
    encoder.load()
    print(df.iloc[0])
    print(encoder.transform(df.iloc[:5,:]))