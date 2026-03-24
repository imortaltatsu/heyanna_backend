"""Run the FastAPI server for real-time prediction market analysis."""

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "src.api.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )