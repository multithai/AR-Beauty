# ตั้งค่า "AI แต่งหน้าสวย" (Generative)

ระบบนี้ให้ AI จริงแต่งรูปที่ถ่าย (ผิวเนียน กระจ่างใส ออร่า) โดยรักษาหน้าเดิม
โครงสร้าง: หน้าเว็บ (GitHub Pages) → **Cloudflare Worker** (ถือ API key) → **Replicate** (โมเดล AI)

> ทำครั้งเดียว ~10 นาที · ฟรีทั้ง Cloudflare Worker (มีโควตาฟรีต่อวัน) แต่ Replicate คิดเงินตามรูป (~$0.04/รูป สำหรับ flux-kontext)

## 1) สมัคร Replicate + เอา token
1. ไปที่ https://replicate.com สมัคร/ล็อกอิน
2. เติมเงินขั้นต่ำ (Billing) เพราะโมเดลคิดเงินตามการใช้
3. คัดลอก **API token** จาก https://replicate.com/account/api-tokens (ขึ้นต้น `r8_...`)

## 2) Deploy Cloudflare Worker
ต้องมี Node.js ในเครื่อง แล้วรันในโฟลเดอร์ `backend/`:

```bash
cd backend
npx --yes wrangler login              # ล็อกอิน Cloudflare (เปิดเบราว์เซอร์)
npx --yes wrangler secret put REPLICATE_API_TOKEN
#   ↑ วาง token r8_... ตอนมันถาม
npx --yes wrangler deploy
```

เสร็จแล้วจะได้ URL เช่น `https://pan-ai.<ชื่อคุณ>.workers.dev`

> ไม่อยากใช้ command line? ทำผ่านหน้าเว็บ Cloudflare ได้:
> Workers & Pages → Create → Worker → วางโค้ดจาก `backend/worker.js` → Deploy →
> Settings → Variables → เพิ่ม secret `REPLICATE_API_TOKEN`

## 3) บอกหน้าเว็บว่า backend อยู่ไหน
เปิดไฟล์ `config.js` แล้วใส่ URL ของ Worker:

```js
window.AI_ENDPOINT = "https://pan-ai.yourname.workers.dev";
```

commit + push → GitHub Pages จะ deploy อัตโนมัติ → ปุ่ม **"✨ AI แต่งให้สวย"** ในหน้าผลลัพธ์จะทำงาน

## เปลี่ยนโมเดล / ปรับสไตล์ (ไม่บังคับ)
ตั้งค่าเพิ่มใน Worker (dashboard → Variables หรือ `wrangler secret put`):
- `REPLICATE_MODEL` — เช่น `tencentarc/gfpgan` (ฟื้นฟูหน้า, ถ้าเปลี่ยนตัวนี้ต้องแก้ input ใน `worker.js` เป็น `{ img: image }`)
- `BEAUTY_PROMPT` — คำสั่งแต่งภาพ (ปรับความสวย/สไตล์ได้)
- `ALLOW_ORIGIN` — จำกัดโดเมนที่เรียกได้ เช่น `https://multithai.github.io`

## ความเป็นส่วนตัว
รูปจะถูกส่งไป Replicate เพื่อประมวลผล — ควรแจ้งผู้ใช้และมี consent
(โดยเฉพาะงานคลินิก/ข้อมูลใบหน้า)
