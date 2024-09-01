import os
import cv2

input_path = r'C:\Users\seahc\Downloads\Lab Sheet (1)\data'
output_path = r'detection_result\data result'
os.makedirs(output_path, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def classify_face(filename):
    return 'real' if filename.startswith('P') else 'not real'

TP = TN = FP = FN = 0

for filename in os.listdir(input_path):
    if filename.endswith('.jpg'):
        img = cv2.imread(os.path.join(input_path, filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.equalizeHist(img)
        
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30))
        classification = classify_face(filename)
        
        if len(faces) == 0:
            TN += (classification == 'not real')
            FN += (classification == 'real')
        else:
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
                cv2.putText(img, classification, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            TP += (classification == 'real')
            FP += (classification == 'not real')
        
        cv2.imwrite(os.path.join(output_path, filename), img)
        print(f'Saved {filename} with classification {classification}')

total_images = TP + TN + FP + FN


print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
print(f"Accuracy: {(TP + TN) / total_images * 100:.2f}%")