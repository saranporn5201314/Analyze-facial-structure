# เตรียมเครื่องสำหรับเทรนโมเดล 7 รูปทรงใบหน้า ใน VSCode

คู่มือนี้เขียนจากการติดตั้งจริงบน **Windows 11** ถ้าใช้ macOS / Linux ขั้นตอนเหมือนกัน แค่เปลี่ยนคำสั่งตามที่ระบุไว้

## แหล่งข้อมูล (Dataset)

โปรเจกต์นี้รวมข้อมูลจาก 2 แหล่งเพื่อให้ได้ครบ 7 คลาส:

1. **niten19/face-shape-dataset**: https://www.kaggle.com/datasets/niten19/face-shape-dataset
   มี Heart, Oblong, Oval, Round, Square (~1,000 รูป/คลาส)
2. **minhquangbui/face-shapes**: https://www.kaggle.com/datasets/minhquangbui/face-shapes
   มีครบ 7 คลาส: diamond, heart, oblong, oval, pear, round, square (~65-95 รูป/คลาส)

> ⚠️ **ข้อจำกัดที่ควรรู้:** คลาส Diamond และ Pear มีข้อมูลจากแหล่งที่ 2 เท่านั้น (คลาสละ 60–80 รูป) ขณะที่คลาสอื่นมีคลาสละราว 1,000 รูป ความแม่นยำของ 2 คลาสนี้จึงต่ำกว่าคลาสอื่นอย่างเลี่ยงไม่ได้ สคริปต์ใช้ class weighting ช่วยชดเชยได้บางส่วน ถ้าอยากได้ผลดีขึ้น ต้องหารูปเพิ่มเองสำหรับ 2 คลาสนี้

---

## 0. สิ่งที่ต้องมีก่อน

- **Python 3.11** (ใช้ 3.9–3.11 ได้ แต่ **3.12 ขึ้นไปใช้ไม่ได้** เพราะ TensorFlow 2.15 ไม่รองรับ)
  เช็กเวอร์ชันที่มีในเครื่องด้วย `py -0p` ถ้าไม่มี 3.11 ให้ดาวน์โหลดจาก https://www.python.org/downloads/
- บัญชี Kaggle (สมัครฟรีที่ https://www.kaggle.com)

> ถ้ารัน `pip install` แล้วเจอ `No matching distribution found for tensorflow==2.15.0` แปลว่ากำลังใช้ Python ผิดเวอร์ชัน

## 1. สร้าง venv และติดตั้ง package

เปิด cmd ในโฟลเดอร์ `faceshape-local` แล้วรัน:

```bash
# Windows:
py -3.11 -m venv venv
venv\Scripts\activate
# macOS / Linux:
python3.11 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install --no-deps tensorflowjs==4.20.0 tensorflow-hub==0.16.1 tf-keras==2.15.1 importlib_resources
```

ถ้าเปิด venv สำเร็จ จะมี `(venv)` ขึ้นหน้าบรรทัดคำสั่ง **ทุกครั้งที่เปิด cmd ใหม่ ต้องรัน `venv\Scripts\activate` ก่อน**

> **ทำไม tensorflowjs ต้องติดตั้งแยกด้วย `--no-deps`:** ถ้าติดตั้งแบบปกติ มันจะดึง flax / jax / uvloop / tensorflow-decision-forests มาด้วย
> ซึ่งติดตั้งหรือรันบน Windows ไม่ได้ (บางตัวโดน Smart App Control ของ Windows บล็อก) ทั้งที่การแปลงโมเดล Keras ไม่ได้ใช้แพ็กเกจพวกนี้เลย
> สคริปต์จึงแปลงโมเดลด้วย `convert_to_tfjs.py` แทนคำสั่ง `tensorflowjs_converter`
>
> **ห้ามติดตั้ง `jax` ลงใน venv นี้** เพราะพอมี jax แล้ว `import tensorflow` จะพังทั้งหมด

> มีการ์ดจอ NVIDIA + ตั้งค่า CUDA/cuDNN แล้วจะเทรนเร็วขึ้นมาก ไม่มีก็ใช้ CPU ได้ แค่ช้ากว่า

### ให้ VSCode ใช้ venv นี้ (ไม่บังคับ แต่ช่วยให้ VSCode ไม่ขึ้นเตือน)

venv อยู่ในโฟลเดอร์ย่อย VSCode จึงไม่เจอเอง ต้องตั้งค่าเองตามนี้:

1. กด `Ctrl+Shift+P` → พิมพ์ **Python: Select Interpreter**
2. เลือก **Enter interpreter path...**
3. ใส่ path เต็มของ `faceshape-local\venv\Scripts\python.exe`

ถ้าตั้งถูก มุมขวาล่างของ VSCode จะขึ้น `3.11.x ('venv')` ถ้าไม่ตั้งค่า VSCode จะขีดเส้นเหลืองว่า "could not be resolved" แต่รันใน cmd ได้ปกติ

## 2. ตั้งค่า Kaggle API Token (ทำครั้งเดียว)

1. ไปที่ kaggle.com → ล็อกอิน → คลิกรูปโปรไฟล์ → **Settings** → หัวข้อ **API** → กด **Create New Token**
2. Kaggle จะแสดง token ที่ขึ้นต้นด้วย `KGAT_...` **ให้ copy เก็บไว้ทันที** เพราะปิดหน้าต่างไปแล้วจะดูอีกไม่ได้ (ถ้าลืม ให้สร้างใหม่)
3. บันทึก token ลงไฟล์ `access_token` (ไม่มีนามสกุล) ในโฟลเดอร์ `.kaggle` ของผู้ใช้:
   - Windows (cmd):
     ```
     mkdir %USERPROFILE%\.kaggle
     echo KGAT_xxxxxxxx> %USERPROFILE%\.kaggle\access_token
     ```
     (เปลี่ยน `KGAT_xxxxxxxx` เป็น token ของตัวเอง และ**ห้ามเว้นวรรค**ระหว่าง token กับ `>`)
   - macOS / Linux:
     ```
     mkdir -p ~/.kaggle && echo KGAT_xxxxxxxx > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
     ```

> token แบบใหม่ (`KGAT_...`) ต้องใช้ kaggle **2.x** ขึ้นไป ซึ่งระบุไว้ใน `requirements.txt` แล้ว
> ถ้ามีไฟล์ `kaggle.json` แบบเก่า (มี username + key) ก็ใช้ได้เหมือนกัน วางไว้ที่ `C:\Users\<ชื่อผู้ใช้>\.kaggle\kaggle.json`
>
> **อย่าแชร์ token ให้คนอื่นหรืออัปขึ้น GitHub** ถ้าเผลอหลุดไป ให้ลบ token นั้นทิ้งในหน้า Settings ของ Kaggle แล้วสร้างใหม่

## 3. ดาวน์โหลด Dataset ทั้งสองชุด

รันในโฟลเดอร์ `faceshape-local` (เปิด venv แล้ว):

```bash
python -m kaggle.cli datasets download -d niten19/face-shape-dataset -p ./data/dataset1 --unzip
python -m kaggle.cli datasets download -d minhquangbui/face-shapes -p ./data/dataset2 --unzip
```

ขนาดรวมประมาณ 750 MB

> ใช้ `python -m kaggle.cli` แทนคำสั่ง `kaggle` ตรง ๆ เพราะบน Windows ไฟล์ `kaggle.exe` อาจโดน Smart App Control บล็อก

## 4. รันสคริปต์เทรน

```bash
python train_face_shape.py
```

ตัวเลือก `QUICK_TEST` ใน `CONFIG` ด้านบนของ `train_face_shape.py`:

| ค่า | ใช้ทำอะไร | เวลาโดยประมาณ (CPU) |
|---|---|---|
| `True` | ทดสอบว่าทุกขั้นตอนรันผ่าน (ใช้รูปไม่เกิน 60 รูป/คลาส, 5 epoch) โมเดลที่ได้**ยังใช้งานจริงไม่ได้** | ไม่กี่นาที |
| `False` | เทรนเต็มด้วยข้อมูลทั้งหมด (สูงสุด 20 + 12 epoch) | ประมาณ 1 ชั่วโมง |

แนะนำให้รันด้วย `True` หนึ่งครั้งก่อน ถ้าผ่านค่อยเปลี่ยนเป็น `False` แล้วรันใหม่ **ระหว่างเทรนห้ามปิดหน้าต่าง cmd**

สคริปต์จะ:
1. ค้นหารูปของแต่ละคลาสจากทั้งสอง dataset อัตโนมัติ (ขอแค่ชื่อโฟลเดอร์ตรงกับชื่อคลาส จะอยู่ใน subfolder ไหนก็ได้)
2. ตรวจรูปทุกไฟล์ แล้ว**ข้ามรูปที่ไฟล์เสีย** (ตอนนี้ dataset1 มีไฟล์เสีย 2 ไฟล์ จะขึ้นข้อความ `ข้ามรูปที่ไฟล์เสีย: ...` ซึ่งเป็นเรื่องปกติ)
3. แบ่ง train/val/test เอง (70/15/15) ลงโฟลเดอร์ `data_7class/`
4. คำนวณ class weight อัตโนมัติจากจำนวนรูปจริงของแต่ละคลาส
5. เทรนด้วย MobileNetV2 (transfer learning + fine-tuning)
6. แสดงผล `Test accuracy` และตารางความแม่นยำแยกคลาส
7. บันทึก `output/face_shape_model.h5` แล้วแปลงเป็น TensorFlow.js ไว้ที่ `output/tfjs_model/`

เมื่อเสร็จจะขึ้นข้อความ `เสร็จแล้ว! ไฟล์สำหรับใช้บนเว็บอยู่ที่: ./output/tfjs_model`

### ข้อความที่ขึ้นระหว่างรัน แต่ไม่ใช่ error

ข้อความพวกนี้ไม่ต้องสนใจ:
- `oneDNN custom operations are on...`
- `This TensorFlow binary is optimized to use available CPU instructions...`
- `WARNING:tensorflow: ... is deprecated`

error จริงจะมีคำว่า **`Traceback`** แล้วบรรทัดสุดท้ายจะลงท้ายด้วย `...Error: ...`

## 5. เอาโมเดลไปใส่เว็บ

คัดลอก**ไฟล์ทั้งหมดในโฟลเดอร์** `output/tfjs_model/` (ไม่ใช่ตัวโฟลเดอร์ `tfjs_model`) ไปวางใน `faceshape-web/model/`
- ถ้า Windows ถามว่าจะแทนที่ไฟล์เดิมไหม ให้กด **แทนที่ (Replace)** ทั้งหมด
- ถ้าจำนวนไฟล์ `group1-shard...bin` ไม่เท่ากับของเดิม ให้ลบไฟล์ `.bin` เก่าใน `faceshape-web/model/` ออกก่อน

ขั้นตอนต่อจากนี้ (ทดสอบเว็บและ deploy) ดูใน `DEPLOY_GUIDE.md`

---

## แก้ปัญหาที่เคยเจอ

| อาการ | สาเหตุ / วิธีแก้ |
|---|---|
| `No matching distribution found for tensorflow==2.15.0` | ใช้ Python 3.12 ขึ้นไป ให้สร้าง venv ใหม่ด้วย `py -3.11 -m venv venv` |
| `uvloop does not support Windows` ตอน `pip install` | ติดตั้ง tensorflowjs แบบไม่ใส่ `--no-deps` ให้ทำตามขั้นที่ 1 |
| `An Application Control policy has blocked this file` | Smart App Control ของ Windows บล็อกไฟล์ .exe ให้ใช้ `python -m ...` แทน (ตามคำสั่งในคู่มือนี้) |
| `Could not find kaggle.json` หรือ Kaggle ล็อกอินไม่ได้ | ยังไม่ได้ทำขั้นที่ 2 หรือ kaggle ในเครื่องเป็นรุ่นเก่ากว่า 2.x |
| `OSError: image file is truncated` | รูปใน dataset เสีย สคริปต์เวอร์ชันปัจจุบันข้ามรูปเสียให้อัตโนมัติแล้ว ถ้ายังเจอ ให้เช็กว่าใช้ `train_face_shape.py` ตัวล่าสุด |
| `import tensorflow` พังและมีคำว่า `jaxlib` / `mlir` | มี jax ติดตั้งอยู่ใน venv ให้ถอนด้วย `pip uninstall -y jax jaxlib` |
