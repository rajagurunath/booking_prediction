import ray
from fastapi import FastAPI
from ray import serve

app = FastAPI()
ray.init(address="auto", namespace="summarizer")
serve.start(detached=True)

@app.get("/")
def f():
    return "Hello from the root!"

@serve.deployment(route_prefix="/api1")
@serve.ingress(app)
class FastAPIWrapper1:
    @app.get("/subpath")
    def method(self):
        return "Hello 1!"

@serve.deployment(route_prefix="/api2")
@serve.ingress(app)
class FastAPIWrapper2:
    @app.get("/subpath")
    def method(self):
        return "Hello 2!"

FastAPIWrapper1.deploy()
FastAPIWrapper2.deploy()