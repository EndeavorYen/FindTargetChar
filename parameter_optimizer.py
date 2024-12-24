import copy
import itertools
import glob
import os
import numpy as np
from template_matching import load_test_answers, character_matching, capture_screenshot
import cv2
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from tqdm import tqdm

def evaluate_parameters(screenshot, template_path, params, expected_matches):
    """評估單一參數組合的效果"""
    # 載入模板圖片
    template = cv2.imread(template_path)
    if template is None:
        print(f"無法載入模板圖片: {template_path}")
        return False
    
    # 備份原始參數
    original_params = {
        'base_threshold': 0.65,
        'complexity_weight': -0.05,
        'resolution_adjustment': {
            1920: -0.03,
            1440: 0.0,
            3840: 0.03
        }
    }
    
    # 套用新參數
    try:
        # 使用新的參數進行匹配
        matches = character_matching(screenshot, template)
        detected_template_name = os.path.basename(template_path).split('.')[0]
        
        # 計算準確度
        is_correct = (detected_template_name in expected_matches) == (len(matches) > 0)
        
        return is_correct
    finally:
        # 還原原始參數
        pass

def evaluate_param_combination(param_combination, answers):
    """評估單一參數組合的效果"""
    correct_matches = 0
    total_tests = 0
    
    # 測試每個截圖
    for screenshot_file, expected in answers.items():
        screenshot = cv2.imread(f"screenshots/{screenshot_file}")
        if screenshot is None:
            continue
            
        # 測試每個模板
        template_files = glob.glob('templates/t*.png')
        for template_file in template_files:
            total_tests += 1
            if evaluate_parameters(screenshot, template_file, param_combination, expected['matches']):
                correct_matches += 1
    
    accuracy = correct_matches / total_tests if total_tests > 0 else 0
    return accuracy, param_combination

def grid_search():
    """使用多執行緒執行網格搜索來找出最佳參數"""
    # 定義參數搜索範圍
    param_grid = {
        'base_threshold': np.arange(0.55, 0.75, 0.02),
        'complexity_weight': np.arange(-0.08, -0.02, 0.01),
        'resolution_adjustment_1920': np.arange(-0.05, 0.0, 0.01),
        'resolution_adjustment_1440': np.arange(-0.02, 0.02, 0.01),
        'resolution_adjustment_3840': np.arange(0.01, 0.06, 0.01)
    }
    
    # 載入測試資料
    answers = load_test_answers()
    best_accuracy = 0
    best_params = None
    
    # 產生所有可能的參數組合
    param_combinations = [dict(zip(param_grid.keys(), v)) 
                        for v in itertools.product(*param_grid.values())]
    
    total_combinations = len(param_combinations)
    print(f"開始測試 {total_combinations} 種參數組合...")
    
    # 使用 ThreadPoolExecutor 進行並行處理
    max_workers = min(os.cpu_count(), 8)  # 限制最大執行緒數
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 建立進度條
        with tqdm(total=total_combinations, desc="參數測試進度") as pbar:
            # 提交所有任務
            future_to_params = {
                executor.submit(evaluate_param_combination, params, answers): params 
                for params in param_combinations
            }
            
            # 處理完成的任務結果
            for future in concurrent.futures.as_completed(future_to_params):
                try:
                    accuracy, params = future.result()
                    pbar.update(1)
                    
                    # 更新最佳參數
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = params
                        print(f"\n找到更好的參數組合:")
                        print(f"準確率: {best_accuracy*100:.2f}%")
                        print("參數:", best_params)
                except Exception as e:
                    print(f"處理參數時發生錯誤: {e}")
    
    return best_params, best_accuracy

def save_best_params(params, accuracy):
    """儲存最佳參數到檔案"""
    with open('best_params.txt', 'w') as f:
        f.write(f"# 最佳參數組合 (準確率: {accuracy*100:.2f}%)\n")
        for key, value in params.items():
            f.write(f"{key} = {value}\n")

if __name__ == "__main__":
    best_params, best_accuracy = grid_search()
    save_best_params(best_params, best_accuracy)
    print("\n最終結果:")
    print(f"最佳準確率: {best_accuracy*100:.2f}%")
    print("最佳參數:", best_params)