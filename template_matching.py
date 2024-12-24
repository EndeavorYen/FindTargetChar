import cv2
import numpy as np
import time
import sys
import os
from datetime import datetime
from sklearn.cluster import DBSCAN
import argparse
import glob
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from sklearn.cluster import MeanShift
import json
import itertools
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

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
        # Lazy import
        import pyautogui
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
            "delay_time": config.getfloat('Settings', 'delay_time', fallback=1.0),
            "thread_count": config.getint('Settings', 'thread_count', fallback=4)
        }
    except Exception as e:
        print(f"讀取設定檔時發生錯誤：{e}，將使用預設值")
        return {
            "star_count": 3,
            "target_count": 1,
            "save_star_screenshot": True,
            "save_target_screenshot": True,
            "delay_time": 1.0,
            "thread_count": 4
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
                        # print(f"找到匹配點 - 座標: {center_pt}, 相似度: {score:.4f}, 縮放比例: {scale:.2f}")
                        
        except Exception as e:
            print(f"預處理方法發生錯誤: {e}")
            continue
    
    return scale_points, scale_scores

# def template_matching_advanced(screenshot, template, threshold=0.8):
#     """使用多執行緒的進階模板匹配函式"""
#     scale_factor = screenshot.shape[1] / 1920
    
#     # 調整縮放範圍
#     if scale_factor > 1.5:
#         scale_ranges = [scale_factor * x for x in [0.95, 0.97, 0.99, 1.0, 1.01, 1.03, 1.05]]
#     else:
#         scale_ranges = [scale_factor * x for x in [0.93, 0.96, 0.98, 1.0, 1.02, 1.04, 1.07]]
    
#     # 預處理方法組合
#     methods = [
#         (lambda s, t: (cv2.cvtColor(s, cv2.COLOR_BGR2GRAY),
#                       cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)), 1.0),
#         (lambda s, t: (cv2.Canny(cv2.cvtColor(s, cv2.COLOR_BGR2GRAY), 100, 200),
#                       cv2.Canny(cv2.cvtColor(t, cv2.COLOR_BGR2GRAY), 100, 200)), 0.6),
#         (lambda s, t: (cv2.Canny(cv2.cvtColor(s, cv2.COLOR_BGR2GRAY), 50, 150),
#                       cv2.Canny(cv2.cvtColor(t, cv2.COLOR_BGR2GRAY), 50, 150)), 0.6),
#         (lambda s, t: (cv2.cvtColor(s, cv2.COLOR_BGR2HSV)[:,:,0],
#                       cv2.cvtColor(t, cv2.COLOR_BGR2HSV)[:,:,0]), 0.7),
#         (lambda s, t: (cv2.cvtColor(s, cv2.COLOR_BGR2HSV)[:,:,1],
#                       cv2.cvtColor(t, cv2.COLOR_BGR2HSV)[:,:,1]), 0.5)
#     ]
    
#     all_points = []
#     all_scores = []
    
#     # 使用線程池處理不同的縮放比例
#     with ThreadPoolExecutor(max_workers=min(len(scale_ranges), 4)) as executor:
#         process_func = partial(process_scale, screenshot, template, methods=methods, threshold=threshold)
#         results = list(executor.map(process_func, scale_ranges))
        
#         # 合併所有結果
#         for points, scores in results:
#             all_points.extend(points)
#             all_scores.extend(scores)
    
#     if all_points:
#         all_points = np.array(all_points)
#         all_scores = np.array(all_scores)
        
#         # 群集處理
#         clustering = DBSCAN(eps=30, min_samples=2).fit(all_points)
        
#         final_points = []
#         for label in set(clustering.labels_):
#             if label == -1:
#                 continue
#             mask = clustering.labels_ == label
#             cluster_points = all_points[mask]
#             cluster_scores = all_scores[mask]
            
#             mean_score = np.mean(cluster_scores)
#             if mean_score > threshold * 0.95:
#                 weights = cluster_scores / np.sum(cluster_scores)
#                 center = np.average(cluster_points, weights=weights, axis=0)
#                 final_points.append(tuple(map(int, center)))
        
#         if final_points:
#             final_points.sort(key=lambda p: p[1])
            
#         return final_points
    
#     return []

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

def load_optimization_params():
    """載入優化後的參數"""
    try:
        with open('optimal_params.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            'template_threshold': 0.6,  # 模板匹配閾值
            'feature_threshold': 40,    # 特徵匹配閾值
            'cluster_threshold': 30     # 群集閾值
        }

def save_optimization_params(params):
    """儲存優化後的參數"""
    with open('optimal_params.json', 'w') as f:
        json.dump(params, f, indent=4)

class EvaluationMetrics:
    """評估指標類別，用於處理所有評估相關的功能"""
    def __init__(self, star_template, verbose=False):
        self.star_template = star_template
        self.verbose = verbose

    def evaluate_single_case(self, screenshot, template_files, expected, params=None):
        """評估單一測試案例"""
        # 檢查5星數量
        star_matches = star_matching(screenshot, self.star_template)
        detected_star_count = len(star_matches)
        star_correct = (detected_star_count == expected['star_count'])
        
        # 檢查角色匹配
        detected_matches = set()
        for template_file in template_files:
            template = cv2.imread(template_file)
            if template is None:
                print(f"無法載入模板 {template_file}")
                continue
            
            template_name = os.path.basename(template_file).split('.')[0]
            # 提取基本角色ID (例如: 't2_1' -> 't2')
            base_character_id = template_name.split('_')[0]
            
            matches = character_matching(screenshot, template, params)
            if matches:
                detected_matches.add(base_character_id)
        
        # 驗證角色匹配結果
        expected_matches = expected['matches']
        if expected_matches == set() or expected_matches == {'-'}:
            match_correct = (not detected_matches)
        else:
            # 將檢測到的角色ID與期望的角色ID進行比對
            match_correct = (detected_matches == expected_matches)
        
        if self.verbose:
            print(f"\n驗證結果:")
            print(f"5星數量 - 標準答案: {expected['star_count']}, 偵測到: {detected_star_count}, 正確: {star_correct}")
            print(f"角色匹配 - 標準答案: {expected['matches']}, 偵測到: {detected_matches}, 正確: {match_correct}")
        
        return star_correct, match_correct

    def evaluate_accuracy(self, screenshot_files, answers, template_files, params=None):
        """評估整體準確率"""
        total_tests = 0
        correct_stars = 0
        correct_matches = 0
        
        for screenshot_file in screenshot_files:
            filename = os.path.basename(screenshot_file)
            if filename not in answers:
                continue
                
            if self.verbose:
                print(f"\n處理截圖：{filename}")
            
            screenshot = cv2.imread(screenshot_file)
            if screenshot is None:
                continue
            
            expected = answers[filename]
            # 檢查 matches 是否已經是 set
            if not isinstance(expected['matches'], set):
                expected['matches'] = set() if expected['matches'] == '-' else set(expected['matches'].split(','))
            
            star_correct, match_correct = self.evaluate_single_case(
                screenshot, template_files, expected, params)
            
            total_tests += 1
            if star_correct:
                correct_stars += 1
            if match_correct:
                correct_matches += 1
        
        return total_tests, correct_stars, correct_matches

def run_tests():
    """執行測試模式"""
    print("\n開始測試 template_matching 函式...")
    
    # 載入必要資料
    answers = load_test_answers()
    if not answers:
        print("無法進行準確度驗證")
        return
    
    # 修改模板文件搜尋邏輯
    template_files = []
    templates_dir = 'templates'
    if os.path.exists(templates_dir):
        # 搜尋主目錄中的模板文件
        template_files.extend(glob.glob(os.path.join(templates_dir, '*.png')))
        # 搜尋子文件夾中的模板文件
        template_files.extend(glob.glob(os.path.join(templates_dir, '*', '*.png')))
    
    if not template_files:
        print("templates 資料夾中沒有找到任何 PNG 檔案。")
        print("請確認：")
        print("1. templates 資料夾存在")
        print("2. 資料夾中有 PNG 格式的模板文件")
        print("3. 檔案權限正確")
        return
        
    screenshot_files = glob.glob(os.path.join('screenshots', '*.png'))
    if not screenshot_files:
        print("screenshots 資料夾中沒有 PNG 檔案供測試。")
        return
    
    star_template = cv2.imread('btns/1920/5star.png', cv2.IMREAD_COLOR)
    if star_template is None:
        print("無法載入 5star.png 範本")
        return
    
    print(f"\n找到 {len(template_files)} 個模板檔案")
    for template in template_files:
        print(f"- {template}")
    print(f"找到 {len(screenshot_files)} 個測試截圖\n")
    
    # 建立評估器並執行評估
    evaluator = EvaluationMetrics(star_template, verbose=True)
    total_tests, correct_stars, correct_matches = evaluator.evaluate_accuracy(
        screenshot_files, answers, template_files
    )
    
    # 輸出結果
    if total_tests > 0:
        print(f"\n測試總結:")
        print(f"總測試數: {total_tests}")
        print(f"5星辨識準確率: {(correct_stars/total_tests)*100:.2f}%")
        print(f"角色匹配準確率: {(correct_matches/total_tests)*100:.2f}%")

def evaluate_params(args):
    """評估單一參數組合"""
    params, screenshot_files, answers, template_files, star_template = args
    try:
        print(f"正在評估參數組合: {params}")  # 新增這行來顯示正在測試的參數
        evaluator = EvaluationMetrics(star_template, verbose=False)
        total_tests, correct_stars, correct_matches = evaluator.evaluate_accuracy(
            screenshot_files, answers, template_files, params)
        
        star_accuracy = correct_stars / total_tests if total_tests > 0 else 0
        match_accuracy = correct_matches / total_tests if total_tests > 0 else 0
        
        # 只有當5星辨識準確率達到100%時，才考慮該參數組合
        if star_accuracy == 1.0:
            accuracy = match_accuracy
        else:
            accuracy = 0
            
        return params, accuracy, star_accuracy, match_accuracy
    except Exception as e:
        print(f"評估參數時發生錯誤: {str(e)}")
        return params, 0, 0, 0

def optimize_parameters(test_cases):
    """使用網格搜索和多執行緒優化參數"""
    # 載入設定
    config = load_config()
    thread_count = min(config['thread_count'], os.cpu_count() or 4)
    
    # 從測試案例中提取所需資訊
    screenshot_files = [case[0] for case in test_cases]
    template_files = list(set([template for case in test_cases for template in case[1]]))
    
    # 將 test_cases 轉換為 answers 格式
    answers = {}
    for screenshot_path, template_paths, expected in test_cases:
        filename = os.path.basename(screenshot_path)
        answers[filename] = expected
    
    star_template = cv2.imread('btns/1920/5star.png')
    
    # 創建評估器實例
    evaluator = EvaluationMetrics(star_template, verbose=False)

    # 第一階段：粗略搜索
    param_ranges_coarse = {
        'template_threshold': [0.4, 0.5, 0.6, 0.7, 0.8],
        'feature_threshold': [30, 40, 50, 60],
        'cluster_threshold': [20, 30, 40, 50]
    }
    
    print("第一階段：粗略搜索")
    param_combinations = [dict(zip(param_ranges_coarse.keys(), v)) 
                        for v in itertools.product(*param_ranges_coarse.values())]
    
    # 準備參數組合
    eval_args = [(params, screenshot_files, answers, template_files, star_template) 
                 for params in param_combinations]
    
    # 使用設定的執行緒數量
    max_workers = thread_count
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_params, args) for args in eval_args]
        total = len(futures)
        
        print(f"\n開始評估參數組合（粗略），共 {total} 組")
        for i, future in enumerate(as_completed(futures), 1):
            try:
                params, accuracy, star_accuracy, match_accuracy = future.result(timeout=60)
                if accuracy > 0:  # 只記錄有效的結果
                    results.append((params, accuracy))
                print(f"進度: {i}/{total} ({i/total*100:.1f}%), "
                      f"5星準確率: {star_accuracy:.3f}, "
                      f"角色準確率: {match_accuracy:.3f}, "
                      f"總分: {accuracy:.3f}")
            except Exception as e:
                print(f"處理結果時發生錯誤: {str(e)}")
                continue
    
    if not results:
        print("未找到有效的參數組合，使用預設參數")
        return {
            'template_threshold': 0.6,
            'feature_threshold': 40,
            'cluster_threshold': 30
        }
    
    # 選擇前3個最佳結果進行細化
    top_results = sorted(results, key=lambda x: x[1], reverse=True)[:3]
    
    # 第二階段：細化搜索
    print("\n第二階段：細化搜索")
    refined_results = []
    
    for base_params, base_accuracy in top_results:
        refined_ranges = {
            'template_threshold': np.linspace(
                max(0.3, base_params['template_threshold'] - 0.05),
                min(0.9, base_params['template_threshold'] + 0.05),
                5
            ),
            'feature_threshold': range(
                max(20, base_params['feature_threshold'] - 5),
                min(70, base_params['feature_threshold'] + 6),
                2
            ),
            'cluster_threshold': range(
                max(15, base_params['cluster_threshold'] - 5),
                min(60, base_params['cluster_threshold'] + 6),
                2
            )
        }
        
        refined_combinations = [dict(zip(refined_ranges.keys(), v)) 
                              for v in itertools.product(*map(list, refined_ranges.values()))]
        
        # 準備細化參數組合
        refined_eval_args = [(params, screenshot_files, answers, template_files, star_template) 
                            for params in refined_combinations]
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(evaluate_params, args) for args in refined_eval_args]
            total_refined = len(futures)
            
            print(f"\n開始細化評估，基準精確度: {base_accuracy:.3f}，共 {total_refined} 組")
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    params, accuracy, star_accuracy, match_accuracy = future.result(timeout=60)
                    if accuracy > 0:
                        refined_results.append((params, accuracy))
                    print(f"進度: {i}/{total_refined} ({i/total_refined*100:.1f}%), "
                          f"5星準確率: {star_accuracy:.3f}, "
                          f"角色準確率: {match_accuracy:.3f}, "
                          f"總分: {accuracy:.3f}")
                except Exception as e:
                    print(f"處理結果時發生錯誤: {str(e)}")
                    continue

    # 找出最佳參數
    best_params, best_accuracy = max(refined_results, key=lambda x: x[1])
    
    # 使用最佳參數重新評估以顯示詳細結果
    evaluator.verbose = True
    total_tests, correct_stars, correct_matches = evaluator.evaluate_accuracy(
        screenshot_files,
        answers,
        template_files,
        params=best_params
    )
    
    print(f"\n最佳參數組合:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"5星辨識準確率: {(correct_stars/total_tests)*100:.2f}%")
    print(f"角色匹配準確率: {(correct_matches/total_tests)*100:.2f}%")
    print(f"總體準確率: {best_accuracy*100:.2f}%")
    
    save_optimization_params(best_params)
    return best_params

def prepare_test_cases():
    """準備測試案例"""
    test_cases = []
    answers = load_test_answers()
    
    for filename, data in answers.items():
        screenshot_path = os.path.join('screenshots', filename)
        if not os.path.exists(screenshot_path):
            continue
            
        template_paths = []
        matches = set() if data['matches'] == '-' else set(data['matches'].split(','))
        
        for template_name in matches:
            template_path = os.path.join('templates', f'{template_name}.png')
            if os.path.exists(template_path):
                template_paths.append(template_path)
        
        test_cases.append((
            screenshot_path,
            template_paths,
            {
                'star_count': data['star_count'],
                'matches': matches
            }
        ))
    
    return test_cases

def character_matching(screenshot, template, params=None):
    """使用模板匹配來進行角色圖片匹配"""
    if params is None:
        params = load_optimization_params()
    
    # 使用原始的 template_matching 函數，但調整閾值
    threshold = params.get('template_threshold', 0.6)
    # 只在非優化模式下顯示閾值
    if not sys.argv[1:] or '--optimize' not in sys.argv[1:]:
        print(f"使用閾值: {threshold}")
    points = template_matching(screenshot, template, threshold, True)
    
    return points

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

def get_template_count():
    """獲取角色資料夾的數量"""
    templates_dir = os.path.join(os.getcwd(), "templates")
    if not os.path.exists(templates_dir):
        print("templates 資料夾不存在！請確認路徑。")
        return 0

    # 計算 t* 資料夾的數量
    folders = [f for f in os.listdir(templates_dir) 
              if os.path.isdir(os.path.join(templates_dir, f)) and f.startswith('t')]
    return len(folders)

def load_character_templates():
    """載入所有角色的範本圖片
    Returns:
        dict: 角色編號為key，包含該角色所有範本圖片的列表為value
    """
    templates_dir = os.path.join(os.getcwd(), "templates")
    character_templates = {}
    
    if not os.path.exists(templates_dir):
        print("templates 資料夾不存在！")
        return character_templates
    
    # 遍歷每個 t* 資料夾
    for character_folder in sorted(os.listdir(templates_dir)):
        if not character_folder.startswith('t'):
            continue
            
        folder_path = os.path.join(templates_dir, character_folder)
        if not os.path.isdir(folder_path):
            continue
            
        # 載入該角色的所有範本圖片
        templates = []
        for img_file in sorted(os.listdir(folder_path)):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, img_file)
                template = cv2.imread(img_path)
                if template is not None:
                    templates.append(template)
                    print(f"已載入範本：{img_path}")
        
        if templates:
            character_templates[character_folder] = templates
    
    return character_templates

def check_templates(screenshot, character_templates):
    """檢查是否找到足夠的角色圖片
    Args:
        screenshot: 螢幕截圖影像 (BGR 格式)
        character_templates: 角色範本字典，key為角色編號(t1, t2等)，value為該角色的範本圖片列表
    Returns:
        bool: 是否找到足夠的角色圖片
    """
    if target_match_count == 0:
        print(f"未設置目標匹配數量，已滿足條件")
        if save_target_screenshot:
            save_screenshot()
        return True
    
    match_count = 0
    matched_characters = set()
    
    # 遍歷每個角色
    for character_id, templates in character_templates.items():
        character_matched = False
        
        # 遍歷該角色的所有範本圖片
        for template in templates:
            points = character_matching(screenshot, template)
            if points:
                character_matched = True
                print(f"找到角色 {character_id}")
                break  # 只要找到該角色的任一範本就可以了
        
        if character_matched:
            matched_characters.add(character_id)
            match_count = len(matched_characters)
            
        if match_count >= target_match_count:
            print(f"已找到 {match_count} 個不同角色，已滿足條件({target_match_count}個以上)")
            print(f"匹配到的角色：{', '.join(sorted(matched_characters))}")
            if save_target_screenshot:
                save_screenshot()
            return True
    
    if match_count == 0:
        print("未找到任何匹配角色")
    else:
        print(f"目前找到 {match_count} 個不同角色：{', '.join(sorted(matched_characters))}")
    return False

def capture_screenshot(save_to_file=False):
    """擷取螢幕畫面，可選擇是否儲存檔案"""
    try:
        # Lazy import
        import pyautogui
        screenshot = pyautogui.screenshot()
        screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        if save_to_file:
            screenshots_dir = os.path.join(os.getcwd(), "screenshots")
            os.makedirs(screenshots_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(screenshots_dir, f"screenshot_{timestamp}.png")
            cv2.imwrite(filename, screenshot)
            print(f"已截圖，已存放在目錄: {filename}")
        
        return screenshot
    except Exception as e:
        print(f"截圖時發生錯誤：{e}")
        return None

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
    import pyautogui
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
    """監聽鍵盤事件"""
    try:
        # Lazy import
        from pynput import keyboard
        
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

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
    except Exception as e:
        print(f"設置鍵盤監聽時發生錯誤：{e}")

def get_template_count():
    """獲取角色資料夾的數量"""
    templates_dir = os.path.join(os.getcwd(), "templates")
    if not os.path.exists(templates_dir):
        print("templates 資料夾不存在！請確認路徑。")
        return 0

    # 計算 t* 資料夾的數量
    folders = [f for f in os.listdir(templates_dir) 
              if os.path.isdir(os.path.join(templates_dir, f)) and f.startswith('t')]
    return len(folders)

def load_character_templates():
    """載入所有角色的範本圖片
    Returns:
        dict: 角色編號為key，包含該角色所有範本圖片的列表為value
    """
    templates_dir = os.path.join(os.getcwd(), "templates")
    character_templates = {}
    
    if not os.path.exists(templates_dir):
        print("templates 資料夾不存在！")
        return character_templates
    
    # 遍歷每個 t* 資料夾
    for character_folder in sorted(os.listdir(templates_dir)):
        if not character_folder.startswith('t'):
            continue
            
        folder_path = os.path.join(templates_dir, character_folder)
        if not os.path.isdir(folder_path):
            continue
            
        # 載入該角色的所有範本圖片
        templates = []
        for img_file in sorted(os.listdir(folder_path)):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, img_file)
                template = cv2.imread(img_path)
                if template is not None:
                    templates.append(template)
                    print(f"已載入範本：{img_path}")
        
        if templates:
            character_templates[character_folder] = templates
    
    return character_templates

def check_templates(screenshot, character_templates):
    """檢查是否找到足夠的角色圖片
    Args:
        screenshot: 螢幕截圖影像 (BGR 格式)
        character_templates: 角色範本字典，key為角色編號(t1, t2等)，value為該角色的範本圖片列表
    Returns:
        bool: 是否找到足夠的角色圖片
    """
    if target_match_count == 0:
        print(f"未設置目標匹配數量，已滿足條件")
        if save_target_screenshot:
            save_screenshot()
        return True
    
    match_count = 0
    matched_characters = set()
    
    # 遍歷每個角色
    for character_id, templates in character_templates.items():
        character_matched = False
        
        # 遍歷該角色的所有範本圖片
        for template in templates:
            points = character_matching(screenshot, template)
            if points:
                character_matched = True
                print(f"找到角色 {character_id}")
                break  # 只要找到該角色的任一範本就可以了
        
        if character_matched:
            matched_characters.add(character_id)
            match_count = len(matched_characters)
            
        if match_count >= target_match_count:
            print(f"已找到 {match_count} 個不同角色，已滿足條件({target_match_count}個以上)")
            print(f"匹配到的角色：{', '.join(sorted(matched_characters))}")
            if save_target_screenshot:
                save_screenshot()
            return True
    
    if match_count == 0:
        print("未找到任何匹配角色")
    else:
        print(f"目前找到 {match_count} 個不同角色：{', '.join(sorted(matched_characters))}")
    return False

def rewrite_log(iteration, found):
    try:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('log.txt', 'w', encoding='utf-8') as f:
            f.write(f"運行次數: {iteration}\n是否符合條件: {found}\n紀錄時間: {date}\n")
    except Exception as e:
        print(f"寫入記錄檔時發生錯誤: {e}")
        print("繼續執行腳本...")

def load_test_answers():
    """載入測試答案"""
    answers = {}
    try:
        with open('test_answers.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                filename, star_count, matches = line.split('|')
                answers[filename] = {
                    'star_count': int(star_count),
                    'matches': matches
                }
        return answers
    except FileNotFoundError:
        print("未找到 test_answers.txt")
        return {}

def main():
    parser = argparse.ArgumentParser(description="Template Matching Script")
    parser.add_argument('--test', action='store_true', help='啟動測試模式')
    parser.add_argument('--optimize', action='store_true', help='啟動參數優化模式')
    args = parser.parse_args()

    if args.optimize:
        print("啟動參數優化模式...")
        test_cases = prepare_test_cases()
        if not test_cases:
            print("無法找到測試案例，請確認 test_answers.txt 和相關圖片檔案是否存在")
            return
        optimize_parameters(test_cases)
        return
    elif args.test:
        run_tests()
        return

    # 載入優化後的參數
    optimal_params = load_optimization_params()
    
    try:
        # 只在需要時才導入相關模組
        toggle_start_stop()
        
        # 原本的 main 邏輯
        global stop_script, start_script
        global star_match_count, target_match_count
        global save_star_screenshot, save_target_screenshot

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

        # 載入所有角色的範本
        character_templates = load_character_templates()
        print(f"已載入 {len(character_templates)} 個角色的範本")

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