# คู่มือ: เอาเว็บไซต์ขึ้นให้คนทั่วไปใช้งานได้ฟรี ผ่าน Firebase Hosting (Google)

โปรเจกต์นี้วิเคราะห์ใบหน้าเป็น 7 ทรง: Oval, Round, Square, Heart, Diamond, Oblong, Pear

## ภาพรวมขั้นตอนทั้งหมด

1. เทรนโมเดลบนเครื่องตัวเองผ่าน VSCode (โฟลเดอร์ `faceshape-local/`)
2. เอาไฟล์โมเดลที่ได้ไปวางในโฟลเดอร์ `faceshape-web/model/`
3. ทดสอบเว็บในเครื่อง
4. ติดตั้ง Node.js + Firebase CLI แล้ว deploy
5. ได้ลิงก์เว็บไซต์สาธารณะ เช่น `https://your-project.web.app`

ต้องทำตามลำดับ ข้ามขั้นไม่ได้

---

## ขั้นตอนที่ 1: เทรนโมเดล (ใน VSCode)

ทำตาม **`faceshape-local/README_SETUP.md` ให้ครบทุกขั้น** ในนั้นมีรายละเอียดเรื่องเวอร์ชัน Python, การตั้งค่า Kaggle token และวิธีแก้ปัญหาบน Windows ที่สรุปไว้ที่นี่ไม่หมด

สรุปคำสั่ง (Windows, รันในโฟลเดอร์ `faceshape-local`):

```bash
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install --no-deps tensorflowjs==4.20.0 tensorflow-hub==0.16.1 tf-keras==2.15.1 importlib_resources

python -m kaggle.cli datasets download -d niten19/face-shape-dataset -p ./data/dataset1 --unzip
python -m kaggle.cli datasets download -d minhquangbui/face-shapes -p ./data/dataset2 --unzip

python train_face_shape.py
```

ครั้งแรกให้ตั้ง `QUICK_TEST = True` ในไฟล์ `train_face_shape.py` เพื่อเช็กว่าทุกขั้นตอนรันผ่าน จากนั้นเปลี่ยนเป็น `QUICK_TEST = False` แล้วรันใหม่เพื่อเทรนเต็มรูปแบบ (ประมาณ 1 ชั่วโมงบน CPU) **โมเดลจากรอบ `QUICK_TEST = True` ยังไม่แม่นพอจะเอาขึ้นเว็บ**

> หมายเหตุ: คลาส Diamond และ Pear มีข้อมูลน้อยกว่าคลาสอื่นมาก (คลาสละไม่ถึง 100 รูป) ความแม่นยำของ 2 คลาสนี้จึงต่ำกว่าคลาสอื่น

เมื่อรันจบจะได้โฟลเดอร์ `faceshape-local/output/tfjs_model/` ที่มี `model.json`, ไฟล์ `.bin` และ `class_indices.json`

## ขั้นตอนที่ 2: ใส่ไฟล์โมเดลลงเว็บ

คัดลอก**ไฟล์ทั้งหมดในโฟลเดอร์** `faceshape-local/output/tfjs_model/` ไปวางใน `faceshape-web/model/` (วางไฟล์ลงไปตรง ๆ ไม่ใช่วางทั้งโฟลเดอร์ `tfjs_model`)

- ถ้ามีโมเดลเก่าอยู่แล้ว ให้เลือก **แทนที่ (Replace)** ทั้งหมด
- ถ้าจำนวนไฟล์ `.bin` ไม่เท่ากับของเดิม ให้ลบไฟล์ `.bin` เก่าออกก่อน

โครงสร้างสุดท้ายควรเป็น:
```
faceshape-web/
├── index.html
├── firebase.json
└── model/
    ├── README.md
    ├── model.json
    ├── group1-shard1of3.bin
    ├── group1-shard2of3.bin
    ├── group1-shard3of3.bin
    └── class_indices.json
```
(จำนวนไฟล์ `.bin` อาจไม่ใช่ 3 เสมอไป)

## ขั้นตอนที่ 3: ทดสอบเว็บในเครื่อง (แนะนำ)

**ห้ามดับเบิลคลิกเปิด `index.html` ตรง ๆ** เพราะเบราว์เซอร์จะไม่ยอมให้หน้าเว็บโหลดไฟล์โมเดลจากเครื่อง ให้รันเซิร์ฟเวอร์จำลองแทน:

```bash
cd faceshape-web
py -3.11 -m http.server 8000
```

แล้วเปิดเบราว์เซอร์ไปที่ http://localhost:8000 ลองใส่รูปหน้าดู ถ้าขึ้นข้อความ "ยังไม่พบไฟล์โมเดลในโฟลเดอร์ model/" แปลว่าวางไฟล์ในขั้นที่ 2 ผิดที่

ทดสอบเสร็จแล้วกด `Ctrl+C` ใน cmd เพื่อปิดเซิร์ฟเวอร์

## ขั้นตอนที่ 4: ติดตั้ง Node.js และ Firebase CLI (ทำครั้งเดียว)

1. ดาวน์โหลดและติดตั้ง **Node.js รุ่น LTS** จาก https://nodejs.org (กด Next ไปเรื่อย ๆ ได้)
2. **ปิด cmd แล้วเปิดใหม่** ไม่อย่างนั้นจะยังหาคำสั่ง `npm` ไม่เจอ
3. รันคำสั่งต่อไปนี้ (รันจากโฟลเดอร์ไหนก็ได้ และไม่ต้องเปิด venv):

```bash
npm install -g firebase-tools
firebase login
```
`firebase login` จะเปิดเบราว์เซอร์ให้ล็อกอินด้วยบัญชี Google

> ถ้าขึ้นว่า `'npm' is not recognized...` แปลว่ายังไม่ได้ติดตั้ง Node.js หรือยังไม่ได้เปิด cmd ใหม่

## ขั้นตอนที่ 5: สร้างโปรเจกต์ Firebase (ทำครั้งเดียว)

1. ไปที่ https://console.firebase.google.com
2. กด "เพิ่มโปรเจกต์" (Create a project) ตั้งชื่อ เช่น `faceshape-ai`
3. ไม่ต้องเปิด Google Analytics ก็ได้
4. จด **Project ID** ไว้ ซึ่งอาจไม่ตรงกับชื่อที่ตั้งเป๊ะ ๆ เช่น `faceshape-ai-1a2b3`

## ขั้นตอนที่ 6: Deploy

โปรเจกต์นี้มีไฟล์ `firebase.json` ตั้งค่าไว้แล้ว **จึงไม่ต้องรัน `firebase init hosting`** (ถ้ารัน อาจไปเขียนทับ `firebase.json` หรือ `index.html` ได้) ให้รันแค่:

```bash
cd faceshape-web
firebase use --add
firebase deploy
```

- `firebase use --add` ให้เลือกโปรเจกต์ที่สร้างในขั้นที่ 5 ถ้าถามชื่อ alias ให้พิมพ์ `default` ขั้นนี้ทำครั้งเดียว
- `firebase deploy` จะอัปโหลดเว็บ พอเสร็จจะขึ้นบรรทัด `Hosting URL: https://....web.app`

ลิงก์นั้นคือเว็บไซต์ แชร์ให้ใครก็เข้าใช้ได้ทันที ฟรี

---

## อัปเดตเว็บภายหลัง

- **แก้ดีไซน์หรือข้อความ:** แก้ไฟล์ `faceshape-web/index.html` แล้วรัน `firebase deploy` ในโฟลเดอร์ `faceshape-web`
- **เปลี่ยนโมเดลใหม่:** เทรนใหม่ (ขั้นที่ 1) → คัดลอกไฟล์ไปทับ (ขั้นที่ 2) → `firebase deploy`

`firebase.json` ตั้งให้เบราว์เซอร์เช็กไฟล์โมเดลกับเซิร์ฟเวอร์ทุกครั้ง (`no-cache`) พอ deploy โมเดลใหม่ ผู้ใช้จะได้โมเดลใหม่ทันที ไม่ค้างโมเดลเก่าอยู่ในเครื่อง

## หมายเหตุสำคัญ

- โมเดลทำงานทั้งหมดในเบราว์เซอร์ของผู้ใช้ รูปไม่ถูกส่งขึ้นเซิร์ฟเวอร์ และไม่มีค่าใช้จ่ายฝั่งเซิร์ฟเวอร์ไม่ว่าจะมีคนใช้กี่คน
- Firebase Hosting แพ็กเกจฟรี (Spark) ให้ bandwidth 10 GB/เดือน โมเดลขนาดประมาณ 10 MB จึงรองรับการเปิดเว็บครั้งแรกได้ราว 1,000 ครั้ง/เดือน (คนที่เคยเปิดแล้วจะใช้ไฟล์ที่เบราว์เซอร์จำไว้ จึงแทบไม่กิน bandwidth เพิ่ม)
