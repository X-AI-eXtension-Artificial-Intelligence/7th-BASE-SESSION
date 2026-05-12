"""
VOC 2007 test 세트(raw)를 이 저장소의 코드가 기대하는 구조로 재구성하고
.xml 라벨을 단순화된 .csv 라벨로 변환한다.

기대 구조:
    data/VOC_Detection/
    └── test/
        ├── images/   (.jpg)
        └── targets/  (.csv)

사용법 (myyolo 폴더 안에서):
    python prepare_voc_test_windows.py
"""
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "data" / "VOC_Detection"           # 현재 raw 폴더 위치
DST_TEST_IMG = SRC_DIR / "test" / "images"
DST_TEST_TGT = SRC_DIR / "test" / "targets"

# ---------- 1. 목적지 폴더 생성 ----------
DST_TEST_IMG.mkdir(parents=True, exist_ok=True)
DST_TEST_TGT.mkdir(parents=True, exist_ok=True)

# ---------- 2. test.txt에 적힌 ID만 옮긴다 ----------
test_list = SRC_DIR / "ImageSets" / "Main" / "test.txt"
if not test_list.exists():
    raise FileNotFoundError(f"{test_list} 가 없습니다. VOCtest_06-Nov-2007.tar를 제대로 풀었는지 확인.")

with open(test_list, "r") as f:
    ids = [line.strip() for line in f if line.strip()]

print(f"[INFO] test.txt에서 {len(ids)}개의 이미지 ID를 읽었습니다.")

moved_img, moved_xml = 0, 0
for pid in ids:
    jpg_src = SRC_DIR / "JPEGImages" / f"{pid}.jpg"
    xml_src = SRC_DIR / "Annotations" / f"{pid}.xml"
    jpg_dst = DST_TEST_IMG / f"{pid}.jpg"
    xml_dst = DST_TEST_TGT / f"{pid}.xml"

    if jpg_src.exists() and not jpg_dst.exists():
        shutil.move(str(jpg_src), str(jpg_dst))
        moved_img += 1
    if xml_src.exists() and not xml_dst.exists():
        shutil.move(str(xml_src), str(xml_dst))
        moved_xml += 1

print(f"[INFO] 이미지 {moved_img}장, 어노테이션 {moved_xml}개 이동 완료.")

# ---------- 3. .xml -> .csv 변환 (difficult==1 객체는 제외) ----------
converted, total_objects = 0, 0
for xml_path in DST_TEST_TGT.glob("*.xml"):
    csv_path = xml_path.with_suffix(".csv")
    root = ET.parse(xml_path).getroot()
    rows = []
    for obj in root.findall("object"):
        difficult = obj.find("difficult").text
        if difficult != "0":
            continue
        label = obj.find("name").text
        bb = {b.tag: b.text for b in obj.find("bndbox")}
        rows.append(f"{label},{bb['xmin']},{bb['ymin']},{bb['xmax']},{bb['ymax']}")

    with open(csv_path, "w") as f:
        f.write("object,xmin,ymin,xmax,ymax")
        for r in rows:
            f.write("\n" + r)

    xml_path.unlink()  # 원본 .xml 삭제
    converted += 1
    total_objects += len(rows)

print(f"[INFO] .xml -> .csv 변환 {converted}개 완료 (총 객체 {total_objects}개, difficult 제외).")

# ---------- 4. 결과 확인 ----------
print("\n[결과 디렉토리]")
print(f"  {DST_TEST_IMG}  -> {len(list(DST_TEST_IMG.glob('*.jpg')))} jpg")
print(f"  {DST_TEST_TGT}  -> {len(list(DST_TEST_TGT.glob('*.csv')))} csv")
print("\n[DONE] evaluate.py 를 실행할 준비가 끝났습니다.")
