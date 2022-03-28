
# preprocess 

```bash
cd ml/
export MODEL_EXPORT_PATH = "/Users/GLanku/Desktop/not_sure/models/"
python preprocess.py
```

# train 

```bash
cd ml/
export MODEL_EXPORT_PATH = "/Users/GLanku/Desktop/not_sure/models/"
python train.py
```


# start ray server
./start.sh



# start the api

```bash
export MODEL_EXPORT_PATH = "/Users/GLanku/Desktop/not_sure/models/"
python main.py
```

    Distribuited service Deployment

                                    |----Encoder Actor
    FastAPI ----> Actor Ingress ----|                                   
                                    |----predictor Actor