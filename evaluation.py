from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_score, classification_report

# Load YOLO models
model1 = YOLO(r"D:\VITCS_FINAL-20250814T172227Z-1-001\VITCS_FINAL\models\best_helmet.pt")
model2 = YOLO(r"D:\VITCS_FINAL-20250814T172227Z-1-001\VITCS_FINAL\models\best_vehicle.pt")

# Run evaluation on a test set
results1 = model1.val(data=r"D:\VITCS_FINAL-20250814T172227Z-1-001\VITCS_FINAL\data.yaml", split="test")
results2 = model2.val(data=r"D:\VITCS_FINAL-20250814T172227Z-1-001\VITCS_FINAL\data.yaml", split="test")

print("\nModel 1 (Helmet Detection) metrics:")
print(results1)

print("\nModel 2 (Vehicle Detection) metrics:")
print(results2)
