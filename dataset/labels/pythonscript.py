import os

# Path to the directory containing label files
labels_dir = "/Users/jayeshpatil/Desktop/ML_Project/dataset/labels/train"  # Update this to your actual labels directory

# Loop through all .txt files in the labels directory
for file in os.listdir(labels_dir):
    if file.endswith(".txt"):
        file_path = os.path.join(labels_dir, file)
        
        with open(file_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if parts and parts[0] == "15":  # If class index is 15, change to 0
                parts[0] = "0"
            new_lines.append(" ".join(parts))

        # Write the modified content back to the file
        with open(file_path, "w") as f:
            f.write("\n".join(new_lines))

print("All label files have been updated successfully.")
