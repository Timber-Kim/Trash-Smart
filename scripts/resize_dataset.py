import os
from pathlib import Path
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

SRC = Path('data')
DST = Path('data_256')
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

all_imgs = [p for p in SRC.rglob('*') if p.suffix in IMG_EXTS]
total = len(all_imgs)
print(f"총 {total}장 리사이즈 시작...")

done = 0
errors = 0
for img_path in all_imgs:
    out = DST / img_path.relative_to(SRC)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        done += 1
        continue
    try:
        with Image.open(img_path) as im:
            im = im.convert('RGB')
            im.thumbnail((256, 256), Image.LANCZOS)
            im.save(out, 'JPEG', quality=85)
    except Exception as e:
        errors += 1
        continue
    done += 1
    if done % 1000 == 0:
        print(f"  {done}/{total} ({done/total*100:.1f}%)")

print(f"\n완료: {done}장 / 오류: {errors}장")
print(f"저장 위치: {DST.resolve()}")
