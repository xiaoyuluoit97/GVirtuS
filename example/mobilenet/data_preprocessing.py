import os
import numpy as np
from torchvision.datasets import ImageNet
from PIL import Image

imagenet_path = ''  
output_dir = './imagenet_test_1000'  
os.makedirs(output_dir, exist_ok=True)

dataset = ImageNet(root=imagenet_path, split='val')

class_ids = dataset.wnids      
class_names = [dataset.classes[dataset.wnid_to_idx[wnid]] for wnid in class_ids]  

label_file = open(os.path.join(output_dir, 'labels.csv'), 'w')
label_file.write("image_path,class_id,class_name\n")

selected_classes = set()

for idx in np.random.permutation(len(dataset)):
    image, label = dataset[idx]
    class_id = dataset.wnids[label]
    
    if class_id not in selected_classes:
        # img_name = f"{class_id}_{len(selected_classes)}.JPEG" 
        img_name = f"{dataset.wnid_to_idx[class_id]}_{len(selected_classes)}.JPEG"  
        img_path = os.path.join(output_dir, img_name)
        image.save(img_path)

        class_name = dataset.classes[dataset.wnid_to_idx[class_id]]
        label_file.write(f"{img_name},{class_id},{class_name}\n")
        
        selected_classes.add(class_id)
        print(f"Saved {img_name} -> {class_name}")
        
        if len(selected_classes) >= 1000:
            break

label_file.close()
print(f"Done! Saved {len(selected_classes)} images with labels.")