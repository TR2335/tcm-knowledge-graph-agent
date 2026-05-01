import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# 给一个方剂网址，爬回这个方剂的详细内容。
def fetch_formula_details(url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers)
    # 用utf-8编码解析响应内容
    response.encoding = 'utf-8'
    # 解析HTML内容
    soup = BeautifulSoup(response.text, "html.parser")

    # 提取标题
    title_tag = soup.find("h1", class_="p_title")
    # 提取标题文字  strip=True 去除首尾空格
    # 如果有标题，返回标题文字；否则返回未知标题
    title_text = title_tag.get_text(strip=True) if title_tag else "未知标题"

    #  找到网页里 class="p_main_container" 的那个区块
    main_div = soup.select_one(".p_main_container")
    if not main_div:
        print(f"❌ 未找到正文区域：{url}")
        return None, None

    # 提取结构化文本
    content_lines = []
    # 遍历所有h1、h2、h3、p、ul、ol、div标签
    for tag in main_div.find_all(["h1", "h2", "h3", "p", "ul", "ol", "div"]):
        # 如果是p、h2、h3标签，直接提取文本
        if tag.name in ["p", "h2", "h3"]:
            text = tag.get_text(separator="", strip=True)
            # 如果有文本，添加到内容列表
            if text:
                content_lines.append(text)
        # 如果是ul、ol标签，遍历所有li标签，提取文本
        elif tag.name in ["ul", "ol"]:
            # 遍历所有li标签，提取文本
            # 如果有文本，添加到内容列表
            for li in tag.find_all("li", recursive=False):
                li_text = li.get_text(separator="", strip=True)
                # 如果有文本，添加到内容列表
                if li_text:
                    content_lines.append(f"- {li_text}")

    # 去除空行并组合文本
    content_lines = [line for line in content_lines if line.strip()]
    # 组合文本  标题+内容
    full_text = f"【方剂名称】{title_text}\n" + "\n".join(content_lines)

    return title_text, full_text

# 每个方剂的详细内容保存到一个txt文件里，文件名是方剂名称。
# 读取 Excel，批量爬方剂。
def batch_fetch_from_excel(excel_path, overwrite=False):
    df = pd.read_excel(excel_path)
    output_dir = "方剂"
    # 创建一个叫 方剂 的文件夹，用来放爬下来的文本。
    os.makedirs(output_dir, exist_ok=True)
    # 一行一行循环遍历 Excel 里的所有方剂，并显示进度条。
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="📦 抓取中医方剂内容"):
        # 提取方剂名称和完整链接去除首尾空格
        formula_name = str(row["方剂名称"]).strip()
        formula_url = str(row["完整链接"]).strip()

        # 把名字里的 / 换成 _，防止文件名出错。
        safe_title = formula_name.replace("/", "_")
        # 构造要保存的 txt 文件路径。
        filepath = os.path.join(output_dir, f"{safe_title}.txt")

        # 如果文件已经存在，就不重新爬，直接跳过。
        if os.path.exists(filepath) and not overwrite:
            tqdm.write(f"⏩ 已存在，跳过：{safe_title}.txt")
            continue

        try:
            # 调用之前的函数爬虫，访问链接，抓取方剂内容。
            title, content = fetch_formula_details(formula_url)
            if content:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                tqdm.write(f"✅ 保存成功：{safe_title}.txt")
            else:
                tqdm.write(f"⚠️ 内容为空：{formula_name}")
        except Exception as e:
            tqdm.write(f"❌ 出错：{formula_name}，错误信息：{e}")


if __name__ == "__main__":
    batch_fetch_from_excel("中医方剂列表.xlsx", overwrite=False)