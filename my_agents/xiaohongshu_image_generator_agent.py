from langchain_core.runnables import Runnable
import requests
import uuid
from pathlib import Path
from agent_state import AgentState
from common.config import Config

conf = Config()


class XiaohongshuImageGeneratorAgent(Runnable):
    def invoke(self, agent_state: AgentState, config: dict = None) -> AgentState:
        content = agent_state.get('xiaohongshu_tcm_post_content', '')
        if not content:
            agent_state['xiaohongshu_tcm_post_image_path'] = ""
            return agent_state
        try:
            api_url = "https://api.jimeng.jian Shu.com/v1/images/generations"
            headers = {"Authorization": f"Bearer {conf.JIMENG_AK}", "Content-Type": "application/json"}
            payload = {"prompt": f"中医科普配图，风格唯美古风，主题：{content[:50]}", "image_size": "1024x1024"}
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                image_url = result.get("data", [{}])[0].get("url", "")
                if image_url:
                    img_resp = requests.get(image_url, timeout=30)
                    img_path = Path(conf.FAISS_INDEX_BASE) / f"xhs_{uuid.uuid4().hex}.png"
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                    img_path.write_bytes(img_resp.content)
                    agent_state['xiaohongshu_tcm_post_image_path'] = str(img_path)
                    return agent_state
        except Exception as e:
            print(f"图片生成失败: {e}")
        agent_state['xiaohongshu_tcm_post_image_path'] = ""
        return agent_state
