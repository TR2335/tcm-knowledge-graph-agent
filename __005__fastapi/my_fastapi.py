import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
os.chdir(_project_root)

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
from __004__langgraph_agent.__004__langgraph_more_agent import call_langgraph_ai

app = FastAPI(title="中医知识图谱项目", version="1.0.0")


class DataRequest(BaseModel):
    data: Dict[str, Any]


class DataResponse(BaseModel):
    result: Dict[str, Any]


@app.post("/process", response_model=DataResponse)
def process_data(req: DataRequest):
    input_dict = req.data
    user_content = input_dict.get("user_content", "")
    session_id = input_dict.get("session_id", "")
    user_id = input_dict.get("user_id", "")
    is_has_xhs_intent, reply = call_langgraph_ai(user_id, session_id, user_content)
    return {"result": {"is_has_xhs_intent": is_has_xhs_intent, "reply": reply}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
