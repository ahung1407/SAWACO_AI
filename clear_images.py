import os
import glob

def clear_folders(folders=("debug", "results")):
    for folder in folders:
        if os.path.exists(folder):
            # lấy tất cả file ảnh trong thư mục
            files = glob.glob(os.path.join(folder, "*.*"))
            for f in files:
                try:
                    os.remove(f)
                    print(f"Deleted {f}")
                except Exception as e:
                    print(f"Cannot delete {f}: {e}")
        else:
            print(f"Folder '{folder}' not found")

if __name__ == "__main__":
    clear_folders()
