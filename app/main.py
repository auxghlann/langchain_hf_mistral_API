from fastapi import FastAPI
#from fastapi.middleware.cors import CORSMiddleware
#from app.routers import routers

app = FastAPI()
# app.include_router(//your router)
# app.include_router(//your router)


# Add CORS middleware for localhost
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


@app.get('/')
def root():
    return {
        "hello": "world",
        "About": "This is a FASTAPI python code template",
        "Author": "auxghlann",
    }

