from rico_parser import RICOParser

parser = RICOParser()
detections = parser.parse_json(
    r"C:\Workspcae\SelfHealing_UI_Automation\data\101.json")

for d in detections[:5]:
    print(d)
