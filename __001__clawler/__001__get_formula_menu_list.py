import requests
from bs4 import BeautifulSoup
import pandas as pd


def fetch_formulas(url):
    headers = {
        "User-Agent": "Mozilla/5.0" # 模拟浏览器访问，避免被目标网站识别为爬虫
    }

    response = requests.get(url, headers=headers)
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, "html.parser")

    result = []
    # 查找主内容区块
    # 该区块包含所有方剂的分类和链接 查找div块里的class属性为"p_content"
    main_div = soup.find("div", class_="p_content")
    # 检查是否找到主内容区块
    if not main_div:
        print("未找到主内容区块")
        return result

    current_category = None  # 方剂大类，如“解表剂”
    current_subcategory = None  # 功效分类，如“辛温解表”
    # 遍历主内容区块里的所有h2、strong、a标签
    for tag in main_div.find_all(["h2", "strong", "a"]):
        # 如果是h2标签，更新方剂大类
        if tag.name == "h2":
            current_category = tag.text.strip()
        # 如果是strong标签，更新功效分类
        elif tag.name == "strong":
            current_subcategory = tag.text.strip()
        # 如果是a标签，且href属性以"/wiki/"开头，提取方剂名称和链接
        elif tag.name == "a" and tag.get("href", "").startswith("/wiki/"):
            # 链接文字 = 方剂名字
            # 链接地址 = 方剂网址
            formula_name = tag.text.strip()
            formula_url = tag["href"]
            # 把 方剂名、小分类、大分类、网址 打包成一条数据，存进结果列表。
            result.append([
                formula_name,
                current_subcategory,
                current_category,
                formula_url
            ])
    return result


if __name__ == "__main__":
    url = "https://zhongyibaike.com/wiki/中医方剂"
    data = fetch_formulas(url)

    df = pd.DataFrame(data, columns=["方剂名称", "功效分类", "方剂大类", "相对链接"])
    df["完整链接"] = "https://zhongyibaike.com" + df["相对链接"]

    # 保存为 Excel 文件
    df.to_excel("中医方剂列表.xlsx", index=False)
    print("已成功保存为：中医方剂列表.xlsx")
