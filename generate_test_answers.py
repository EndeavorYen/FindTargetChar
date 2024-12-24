import os

def generate_test_answers():
    # 讀取 screenshots 目錄下的所有檔案
    screenshot_dir = 'screenshots'
    screenshot_files = [f for f in os.listdir(screenshot_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # 寫入 test_answers.txt
    with open('test_answers.txt', 'w', encoding='utf-8') as f:
        # 寫入格式說明
        f.write("# Format: screenshot_filename|star_count|t1,t2,t3\n")
        f.write("# Example: screenshot_20240101_120000.png|2|t1,t3\n")
        f.write("# Use '-' for no matches: screenshot_20240101_120000.png|0|-\n\n")
        
        # 為每個截圖檔案建立一個預設的測試案例
        for filename in sorted(screenshot_files):
            f.write(f"{filename}|0|-\n")

if __name__ == "__main__":
    generate_test_answers() 