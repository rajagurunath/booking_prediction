## Quick POC using Ray serve

- install dependencies(requirements.txt)
- start Ray cluster (using `start.sh`)
- start the server
    - which starts ingress and other deployments (Ray Actors)
- before starting the above server
    - preprocess
        - encodes, handles missing values
        - saves sklearn transformer
    - Train 
        - trains some sklearn classifier
        - saves the model


### Future scope:
    - MLOps Tools can used along with Github Actions
    - Visualize and show metrics more clearly in streamlit APP
        - currently dataset was limited for metrics analysis
    - More EDA
    - More statistics
    - More Tools 😉
    - More Fun !!!

