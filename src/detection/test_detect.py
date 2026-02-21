from detect_ui import UIDetector

detector = UIDetector()
detections = detector.detect(
    r"C:\Workspcae\SelfHealing_UI_Automation\src\sample_ui.jpg")

for d in detections:
    print(d)
