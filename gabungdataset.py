import os

def numeric_sort_key(filename):
    base_name = os.path.splitext(filename)[0]
    # If the base name is purely numeric, use it for sorting
    return int(base_name) if base_name.isdigit() else float('inf')

def rename_images(folders, start_number):
    for folder in folders:
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        files.sort(key=numeric_sort_key)  # Sort files numerically

        count = start_number
        for file in files:
            old_path = os.path.join(folder, file)
            extension = os.path.splitext(file)[1]
            new_filename = "{}{}".format(count, extension)
            new_path = os.path.join(folder, new_filename)
            
            if old_path != new_path:
                os.rename(old_path, new_path)
                print("Renamed {} to {}".format(old_path, new_path))
            count += 1

def main():
    # List of folders to process
    folders = ["/home/zephyr/vision/Linemod_Custom/LINEMOD/vitamin3/JPEGImages", "/home/zephyr/vision/Linemod_Custom/LINEMOD/vitamin3/depth"]
    start_number = 506

    rename_images(folders, start_number)

if __name__ == "__main__":
    main()

