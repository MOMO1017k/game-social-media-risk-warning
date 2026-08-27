import time
import json
from selenium import webdriver

def get_bilibili_cookies_manual_login():
    """
    启动浏览器，由用户手动完成登录（扫码或短信），程序等待后自动获取Cookies。
    """
    driver = webdriver.Chrome() # 确保chromedriver已配置好
    driver.get("https://www.bilibili.com/")

    # 增加等待时间，让用户有足够的时间手动操作
    print("浏览器已打开，请在60秒内手动完成扫码或短信登录...")
    time.sleep(60) # 等待60秒

    # 假设用户已经登录成功，此时浏览器会话中已经包含了有效的Cookies
    try:
        cookies = driver.get_cookies()
        if not any('SUB' in cookie['name'] for cookie in cookies):
             print("错误：似乎没有登录成功，未能获取关键Cookie。请重试。")
             driver.quit()
             return

        with open("weibo_cookies.json", "w") as f:
            json.dump(cookies, f)
        
        print("登录成功！Cookies已成功保存到 weibo_cookies.json")

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        driver.quit()

# --- 执行 ---
get_bilibili_cookies_manual_login()
