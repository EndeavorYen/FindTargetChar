import cv2
import numpy as np
import pyautogui
import time
from pynput import keyboard
from PIL import Image
import sys
import os
from datetime import datetime
from pyautogui import screenshot as take_screenshot
import ctypes
from sklearn.cluster import DBSCAN
import argparse
import glob
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# 用來停止腳本的標誌
stop_script = False
start_script = False

star_match_count = 3  # 預設值
target_match_count = 1  # 預設值
save_star_screenshot = True  # 預設值
save_target_screenshot = False  # 預設值
delay_time = 1.0 # 預設值

width = 3840

def get_screen_width():
    """獲取螢幕寬度解析度"""
    try:
        # 移除 Windows 專用的程式碼
        # 改用 pyautogui 來獲取螢幕解析度
        screen_width = pyautogui.size()[0]  # 螢幕寬度
        print(f"螢幕解析度寬度：{screen_width}")
        return screen_width
    except Exception as e:
        print(f"獲取螢幕解析度時發生錯誤：{e}")
        return 1920  # 返回預設值

def get_btn_folder():
    global width
    """根據解析度選擇正確的資料夾"""
    width = get_screen_width()
    folder = (f"btns/{width}")
    print(f"選擇的資料夾：{folder}")
    return folder

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def load_config():
    try:
        import configparser
        config = configparser.ConfigParser()
        config.read('config.ini', encoding='utf-8')
        
        return {
            "star_count": config.getint('Settings', 'star_count', fallback=3),
            "target_count": config.getint('Settings', 'target_count', fallback=1),
            "save_star_screenshot": config.getboolean('Settings', 'save_star_screenshot', fallback=True),
            "save_target_screenshot": config.getboolean('Settings', 'save_target_screenshot', fallback=True),
            "delay_time": config.getfloat('Settings', 'delay_time', fallback=1.0)
        }
    except Exception as e:
        print(f"讀取設定檔時發生錯誤：{e}，將使用預設值")
        return {
            "star_count": 3,
            "target_count": 1,
            "save_star_screenshot": True,
            "save_target_screenshot": True,
            "delay_time": 1.0
        }

def load_image(image_path):
    absolute_path = resource_path(image_path)
    image = cv2.imread(absolute_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"錯誤：無法載入圖片 {image_path}")
    else:
        print(f"已載入圖片：{image_path}")
    return image

def process_scale(screenshot, template, scale, methods, threshold):
    """處理單一縮放比例的所有預處理方法"""
    new_width = int(template.shape[1] * scale)
    new_height = int(template.shape[0] * scale)
    scaled_template = cv2.resize(template, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    scale_points = []
    scale_scores = []
    
    for preprocess, weight in methods:
        try:
            img1, img2 = preprocess(screenshot, scaled_template)
            result = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
            
            loc = np.where(result >= threshold)
            points = list(zip(*loc[::-1]))
            
            if points:
                w, h = img2.shape[::-1]
                for pt in points:
                    score = result[pt[1], pt[0]] * weight
                    if score > threshold * 0.9:
                        center_pt = (pt[0] + w//2, pt[1] + h//2)
                        scale_points.append(center_pt)
                        scale_scores.append(score)
                        
        except Exception as e:
            print(f"預處理方法發生錯誤: {e}")
            continue
    
    return scale_points, scale_scores

def template_matching_advanced(screenshot, template, threshold=0.8):
    """使用多執行緒的進階模板匹配函式"""
    scale_factor = screenshot.shape[1] / 1920
    
    # 調整縮放範圍
    if scale_factor > 1.5:
        scale_ranges = [scale_factor * x for x in [0.95, 0.97, 0.99, 1.0, 1.01, 1.03, 1.05]]
    else:
        scale_ranges = [scale_factor * x for x in [0.93, 0.96, 0.98, 1.0, 1.02, 1.04, 1.07]]
    
    # 預處理方法組合
    methods = [
        (lambda s, t: (cv2.cvtColor(s, cv2.COLOR_BGR2GRAY),
                      cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)), 1.0),
        (lambda s, t: (cv2.Canny(cv2.cvtColor(s, cv2.COLOR_BGR2GRAY), 100, 200),
                      cv2.Canny(cv2.cvtColor(t, cv2.COLOR_BGR2GRAY), 100, 200)), 0.6),
        (lambda s, t: (cv2.Canny(cv2.cvtColor(s, cv2.COLOR_BGR2GRAY), 50, 150),
                      cv2.Canny(cv2.cvtColor(t, cv2.COLOR_BGR2GRAY), 50, 150)), 0.6),
        (lambda s, t: (cv2.cvtColor(s, cv2.COLOR_BGR2HSV)[:,:,0],
                      cv2.cvtColor(t, cv2.COLOR_BGR2HSV)[:,:,0]), 0.7),
        (lambda s, t: (cv2.cvtColor(s, cv2.COLOR_BGR2HSV)[:,:,1],
                      cv2.cvtColor(t, cv2.COLOR_BGR2HSV)[:,:,1]), 0.5)
    ]
    
    all_points = []
    all_scores = []
    
    # 使用線程池處理不同的縮放比例
    with ThreadPoolExecutor(max_workers=min(len(scale_ranges), 4)) as executor:
        process_func = partial(process_scale, screenshot, template, methods=methods, threshold=threshold)
        results = list(executor.map(process_func, scale_ranges))
        
        # 合併所有結果
        for points, scores in results:
            all_points.extend(points)
            all_scores.extend(scores)
    
    if all_points:
        all_points = np.array(all_points)
        all_scores = np.array(all_scores)
        
        # 群集處理
        clustering = DBSCAN(eps=30, min_samples=2).fit(all_points)
        
        final_points = []
        for label in set(clustering.labels_):
            if label == -1:
                continue
            mask = clustering.labels_ == label
            cluster_points = all_points[mask]
            cluster_scores = all_scores[mask]
            
            mean_score = np.mean(cluster_scores)
            if mean_score > threshold * 0.95:
                weights = cluster_scores / np.sum(cluster_scores)
                center = np.average(cluster_points, weights=weights, axis=0)
                final_points.append(tuple(map(int, center)))
        
        if final_points:
            final_points.sort(key=lambda p: p[1])
            
        return final_points
    
    return []

def template_matching(screenshot, template, threshold=0.8, use_rect=True):
    """原始的模板匹配函式，用於按鈕和星星等簡單元素"""
    # 計算縮放比例
    scale_factor = screenshot.shape[1] / 1920
    
    # 根據解析度調整縮放範圍
    if scale_factor > 1.5:
        scale_ranges = [scale_factor * x for x in [0.98, 1.0, 1.02]]  # 4K解析度用較少的縮放範圍
    else:
        scale_ranges = [scale_factor * x for x in [0.95, 1.0, 1.05]]  # 減少縮放範圍，提高準確度
    
    all_points = []
    for scale in scale_ranges:
        # 重新調整模板大小
        new_width = int(template.shape[1] * scale)
        new_height = int(template.shape[0] * scale)
        scaled_template = cv2.resize(template, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # 轉換為灰度圖
        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(scaled_template, cv2.COLOR_BGR2GRAY)
        
        # 執行模板匹配
        result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)
        points = list(zip(*loc[::-1]))
        
        if not points:
            continue
            
        w, h = template_gray.shape[::-1]
        if use_rect:
            rect_points = [[pt[0], pt[1], w, h] for pt in points]
            if rect_points:
                rects, _ = cv2.groupRectangles(rect_points, groupThreshold=1, eps=0.5)
                points = [(int(x + w/2), int(y + h/2)) for x, y, w, h in rects]
                all_points.extend(points)
        else:
            points = [(pt[0] + w//2, pt[1] + h//2) for pt in points]
            all_points.extend(points)
    
    # 合併相近的點
    if all_points:
        all_points = np.array(all_points)
        clustering = DBSCAN(eps=30, min_samples=1).fit(all_points)
        
        final_points = []
        for label in set(clustering.labels_):
            if label == -1:
                continue
            mask = clustering.labels_ == label
            cluster_points = all_points[mask]
            center = np.mean(cluster_points, axis=0)
            final_points.append(tuple(map(int, center)))
        
        # 根據y座標排序
        final_points.sort(key=lambda p: p[1])
        return final_points
    
    return []

def btn_matching(screenshot, template):
    """按鈕匹配，使用原始版本"""
    return template_matching(screenshot, template, 0.6, False)

def star_matching(screenshot, template):
    """5星角色匹配，使用原始版本"""
    global width
    threshold = 0.8 if width > 1920 else 0.78
    return template_matching(screenshot, template, threshold, True)

def character_matching(screenshot, template):
    """角色圖片匹配，使用進階版本"""
    global width
    if width > 1920:
        threshold = 0.68
    elif width > 1440:
        threshold = 0.65
    else:
        threshold = 0.62
    
    template = cv2.GaussianBlur(template, (3, 3), 0)
    screenshot = cv2.GaussianBlur(screenshot, (3, 3), 0)
    
    return template_matching_advanced(screenshot, template, threshold)

def check_star_count(screenshot, template):
    """檢查是否找到足夠的5星角色
    Args:
        screenshot: 螢幕截圖影像 (BGR 格式)
        template: 5星角色範例圖片
    Returns:
        bool: 是否找到足夠的角色圖片
    """
    if star_match_count == 0:
        print(f"未設置5星數量目標，直接進行目標匹配")
        if save_star_screenshot:
            save_screenshot()
        return True
    
    star_count = 0
    points = star_matching(screenshot, template)
    if points:
        star_count = len(points)
        if star_count >= star_match_count:
            print(f"已找到 {star_count} 個5星角色，已滿足條件({star_match_count}以上)")
            if save_star_screenshot:
                save_screenshot()
            return True
        else:
            print(f"未找到足夠的5星角色，目前找到 {star_count} 個")
            return False
    else:
        print("未找到任何5星角色")
        return False

def check_templates(screenshot, templates):
    """檢查是否找到足夠的角色圖片
    Args:
        screenshot: 螢幕截圖影像 (BGR 格式)
        templates: 角色圖片列表
    Returns:
        bool: 是否找到足夠的角色圖片
    """
    if target_match_count == 0:
        print(f"未設置目標匹配數量，已滿足條件")
        if save_target_screenshot:
            save_screenshot()
        return True
    
    match_count = 0
    for template in templates:
        # 改用 character_matching 而不是 template_matching
        points = character_matching(screenshot, template)
        if points:
            count = len(points)
            match_count += count
            print(f"找到角色圖片，共找到 {count} 個角色匹配點")
        if match_count >= target_match_count:
            print(f"已找到至少 {target_match_count} 張角色圖片")
            if save_target_screenshot:
                save_screenshot()
            return True
    if match_count == 0:
        print("未找到任何匹配角色圖片")
    return False

def capture_screenshot(save_to_file=False):
    """擷取螢幕畫面，可選擇是否儲存檔案
    Args:
        save_to_file (bool): 是否儲存為檔案
    Returns:
        np.array: 螢幕截圖的numpy陣列
    """
    screenshot = pyautogui.screenshot()
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    if save_to_file:
        # 確保 screenshots 資料夾存在
        screenshots_dir = os.path.join(os.getcwd(), "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)

        # 使用時間戳命名並儲存截圖
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(screenshots_dir, f"screenshot_{timestamp}.png")
        cv2.imwrite(filename, screenshot)
        print(f"已截圖，已存放在目錄: {filename}")
    
    return screenshot

# 為了向後相容，可以保留save_screenshot函數
def save_screenshot():
    capture_screenshot(save_to_file=True)


def click_buttons( retry_template, retry_confirm_template, skip_template):
    """點擊按鈕操作
    Args:
        retry_template: 重試按鈕範例圖片
        retry_confirm_template: 重試確認按鈕範例圖片
        skip_template: 跳過按鈕範例圖片
    Returns:
        bool: 是否成功點擊按鈕
    """
    global delay_time
    for btn, btn_name in [(retry_template, "retry"), (retry_confirm_template, "retry_confirm"), (skip_template, "skip")]:
        screenshot = capture_screenshot()
        points = btn_matching(screenshot, btn)
        if not points:
            print(f"未找到{btn_name}按鈕")
            return False
        pyautogui.click(points[0])
        print(f"已點擊{btn_name}按鈕")
        time.sleep(delay_time)  # 根據電腦效能修改,建議為 1~2秒

def process_buttons_and_templates(retry_template, retry_confirm_template, skip_template, star_template, templates):
    global stop_script
    global delay_time

    click_buttons(retry_template, retry_confirm_template, skip_template)

    screenshot = capture_screenshot()
    if check_star_count(screenshot, star_template):
        if check_templates(screenshot, templates):
            print("已滿足所有條件。退出...")
            return True
    return False

def toggle_start_stop():
    global stop_script, start_script
    
    def on_press(key):
        global stop_script, start_script
        if key == keyboard.Key.f9:
            if start_script:
                print("按下F9鍵。停止腳本...")
                stop_script = True
                start_script = False
            else:
                print("按下F9鍵。啟動腳本...")
                stop_script = False
                start_script = True
            time.sleep(0.5)

    # 啟動鍵盤監聽
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

def get_template_count():
    templates_dir = os.path.join(os.getcwd(), "templates")
    if not os.path.exists(templates_dir):
        print("templates 資料夾不存在！請確認路徑。")
        return 0

    files = [f for f in os.listdir(templates_dir) if os.path.isfile(os.path.join(templates_dir, f))]
    return len(files)

def rewrite_log(iteration, found):
    try:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('log.txt', 'w', encoding='utf-8') as f:
            f.write(f"運行次數: {iteration}\n是否符合條件: {found}\n紀錄時間: {date}\n")
    except Exception as e:
        print(f"寫入記錄檔時發生錯誤: {e}")
        print("繼續執行腳本...")

def test_template_matching():
    """測試 template_matching 函式的正確性"""
    print("\n開始測試 template_matching 函式...")
    
    # 取得所有模板檔案
    template_files = glob.glob(os.path.join('templates', 't*.png'))
    if not template_files:
        print("templates 資料夾中沒有找到 t*.png 檔案。")
        return
        
    # 取得所有截圖檔案
    screenshot_files = glob.glob(os.path.join('screenshots', '*.png'))
    if not screenshot_files:
        print("screenshots 資料夾中沒有 PNG 檔案供測試。")
        return
    
    # 載入按鈕模板
    folder = get_btn_folder()
    star_template = load_image(f'{folder}/5star.png')
    
    print(f"\n找到 {len(template_files)} 個模板檔案")
    print(f"找到 {len(screenshot_files)} 個測試截圖\n")
    
    # 對每個截圖進行測試
    for screenshot_file in screenshot_files:
        print(f"\n處理截圖：{screenshot_file}")
        screenshot = cv2.imread(screenshot_file, cv2.IMREAD_COLOR)
        if screenshot is None:
            print(f"無法載入截圖：{screenshot_file}")
            continue
            
        # 測試5星辨識
        star_matches = star_matching(screenshot, star_template)
        print(f"5星辨識結果: 找到 {len(star_matches)} 個匹配點")
        if star_matches:
            print(f"匹配座標: {star_matches}")
            
        # 測試角色辨識
        print("\n角色辨識測試:")
        for template_file in template_files:
            template = load_image(template_file)
            if template is None:
                continue
                
            matches = character_matching(screenshot, template)
            print(f"{os.path.basename(template_file)}: 找到 {len(matches)} 個匹配點")
            if matches:
                print(f"匹配座標: {matches}")
        print("-" * 50)

def test_character_matching():
    """測試 character_matching 函式的正確性"""
    print("開始測試 character_matching 函式...")
    character_template_path = resource_path('templates/character_template.png')  # 替換為實際的角色模板路徑
    character_template = load_image(character_template_path)
    if character_template is None:
        print(f"無法載入角色模板圖片：{character_template_path}")
        return
    
    screenshot_files = glob.glob(os.path.join('screenshots', '*.png'))
    if not screenshot_files:
        print("screenshots 資料夾中沒有 PNG 檔案供測試。")
        return
    
    for screenshot_file in screenshot_files:
        print(f"\n處理截圖：{screenshot_file}")
        screenshot = cv2.imread(screenshot_file, cv2.IMREAD_COLOR)
        if screenshot is None:
            print(f"無法載入截圖：{screenshot_file}")
            continue
        matches = character_matching(screenshot, character_template)
        print(f"找到 {len(matches)} 個角色匹配點：{matches}")

def run_tests():
    """執行所有測試"""
    test_template_matching()
    test_character_matching()

def main():
    global stop_script, start_script
    global star_match_count, target_match_count
    global save_star_screenshot, save_target_screenshot

    parser = argparse.ArgumentParser(description="Template Matching Script")
    parser.add_argument('--test', action='store_true', help='啟動測試模式')
    args = parser.parse_args()

    if args.test:
        test_template_matching()  # 只需要這一個測試函數就足夠了
        return

    try:
        toggle_start_stop()

        # 從config讀取target_count
        config = load_config()
        star_match_count = config.get('star_count', 3)
        target_match_count = config.get('target_count', 1)
        save_star_screenshot = config.get('save_star_screenshot', True)
        save_target_screenshot = config.get('save_target_screenshot', True)
        delay_time = config.get('delay_time', 1)

        print(f"""
設定資訊：
- 5 星數量目標: {star_match_count}
- 目標匹配數量: {target_match_count}
- 5星數量達標時截圖: {save_star_screenshot}
- 目標匹配達標時截圖: {save_target_screenshot}
- 延遲時間: {delay_time}
""")

        template_count = get_template_count()
        if template_count == 0:
            print("未找到任何範例圖片，強制將匹配數量設置為0")
            target_match_count = 0
        else:
            print(f"範例圖片數量自動設定為: {template_count}")

        folder = get_btn_folder()

        retry_template = load_image(f'{folder}/retry.png')
        retry_confirm_template = load_image(f'{folder}/retry_confirm.png')
        skip_template = load_image(f'{folder}/skip.png')

        star_template = load_image(f'{folder}/5star.png')

        templates = [load_image(f'templates/t{i + 1}.png') for i in range(template_count)]
        print("範例圖片載入完成，請於遊戲抽卡畫面按下 F9 開始運行腳本")

        while True:
            while not start_script:
                time.sleep(0.1)

            iteration = 0
            stop_script = False

            while start_script:
                iteration += 1
                print(f"運行次數 {iteration}")
                found = process_buttons_and_templates(retry_template, retry_confirm_template, skip_template, star_template, templates)
                rewrite_log(iteration, found)
                if found or stop_script:
                    start_script = False
                    stop_script = True
                    break

                time.sleep(0.1)

            print("腳本已停止，等待重新啟動...")
    except Exception as e:
        print(f"運行過程中出現錯誤: {e}")
    finally:
        input("按下Enter鍵退出終端...")

if __name__ == "__main__":
    main()