"""
test_nan_simulation.py
======================
Kiem thu kha nang phat hien NaN cua he thong SAWACO AI.

Luong xu ly dung:
  1. Doc anh dong ho goc (anh_2.jpg)
  2. Dung segmentation.py de cat tung o digit (100x100px)
  3. Gia lap che khuat len TUNG O digit
  4. Goi inference_lib.predict() de kiem tra co tra NaN khong

Cac tinh huong:
  la_cay, bun_dat, mo_kinh, choi_loa, vet_xay, nut_bun

Su dung:
  python test_nan_simulation.py
  python test_nan_simulation.py --source anh_2.jpg
  python test_nan_simulation.py --save-only
"""

import argparse, os, sys
import cv2, numpy as np
os.makedirs("test_images/nan_sim", exist_ok=True)

# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------
def _jitter(base_bgr, amount=20):
    return tuple(int(np.clip(c+np.random.randint(-amount,amount),0,255)) for c in base_bgr)

def _blob_mask(h, w, n_blobs=5, max_r=60):
    mask = np.zeros((h,w),dtype=np.float32)
    for _ in range(n_blobs):
        cx=np.random.randint(0,w); cy=np.random.randint(0,h)
        rx=np.random.randint(max_r//3,max_r); ry=np.random.randint(max_r//3,max_r)
        tmp=np.zeros((h,w),dtype=np.uint8)
        cv2.ellipse(tmp,(cx,cy),(rx,ry),np.random.randint(0,180),0,360,255,-1)
        mask+=tmp.astype(np.float32)/255.0
    return np.clip(mask,0,1)

# ------------------------------------------------------------------
# 6 Gia lap che khuat tren TUNG O DIGIT (100x100)
# ------------------------------------------------------------------

def sim_la_cay(img):
    out=img.copy(); h,w=out.shape[:2]
    tip_x=np.random.randint(5,w-5); tip_y=np.random.randint(-15,h//4)
    base_x=np.random.randint(5,w-5); base_y=np.random.randint(3*h//4,h+10)
    length=np.hypot(tip_x-base_x,tip_y-base_y)
    half_w=int(length*np.random.uniform(0.20,0.38))
    dx=(tip_y-base_y)/(length+1e-6); dy=-(tip_x-base_x)/(length+1e-6)
    mx=(tip_x+base_x)//2; my=(tip_y+base_y)//2
    pts=np.array([[tip_x,tip_y],[mx+int(half_w*dx),my+int(half_w*dy)],
                  [base_x,base_y],[mx-int(half_w*dx),my-int(half_w*dy)]],dtype=np.int32)
    leaf_variants=[(30,140,30),(10,100,10),(20,160,60),(0,80,20),(30,130,90)]
    leaf_bgr=_jitter(leaf_variants[np.random.randint(0,len(leaf_variants))],25)
    leaf_layer=np.zeros_like(out); cv2.fillPoly(leaf_layer,[pts],leaf_bgr)
    grad=np.zeros((h,w),dtype=np.float32)
    for y in range(h): grad[y,:]=0.65+0.35*(y/h)
    leaf_layer=(leaf_layer.astype(np.float32)*np.stack([grad,grad,grad],axis=2)).astype(np.uint8)
    vc=(max(0,leaf_bgr[0]-20),max(0,leaf_bgr[1]-30),max(0,leaf_bgr[2]-20))
    cv2.line(leaf_layer,(base_x,base_y),(tip_x,tip_y),vc,1)
    mask_leaf=np.zeros((h,w),np.uint8); cv2.fillPoly(mask_leaf,[pts],255)
    out=cv2.add(cv2.bitwise_and(out,out,mask=cv2.bitwise_not(mask_leaf)),
                cv2.bitwise_and(leaf_layer,leaf_layer,mask=mask_leaf))
    return cv2.GaussianBlur(out,(3,3),0)

def sim_bun_dat(img):
    out=img.copy().astype(np.float32); h,w=img.shape[:2]
    mask=_blob_mask(h,w,n_blobs=np.random.randint(2,5),max_r=min(h,w)//2)
    mask=np.clip(mask+np.random.normal(0,0.12,(h,w)).astype(np.float32),0,1)
    mud_variants=[(20,50,80),(30,55,70),(15,35,55),(50,60,50)]
    mud_bgr=np.array(_jitter(mud_variants[np.random.randint(0,len(mud_variants))],20),dtype=np.float32)
    mud_layer=np.random.normal(mud_bgr,15,(h,w,3)).clip(0,255).astype(np.float32)
    alpha=(mask*np.random.uniform(0.6,0.92)).reshape(h,w,1)
    out=np.clip(out*(1-alpha)+mud_layer*alpha,0,255).astype(np.uint8)
    return out

def sim_mo_kinh(img):
    out=img.copy(); h,w=out.shape[:2]
    k=np.random.choice([13,19,25])
    blurred=cv2.GaussianBlur(out,(k,k),0)
    Y,X=np.ogrid[:h,:w]
    fog=np.exp(-((X-w//2)**2+(Y-h//2)**2)/(2*(min(h,w)*0.35)**2)).reshape(h,w,1).astype(np.float32)
    out=(out.astype(np.float32)*(1-fog)+blurred.astype(np.float32)*fog).astype(np.uint8)
    for _ in range(np.random.randint(1,5)):
        gx,gy=np.random.randint(0,w),np.random.randint(0,h)
        gr=np.random.randint(4,15)
        cv2.circle(out,(gx,gy),gr,(175,190,195),1)
        cv2.circle(out,(gx-gr//4,gy-gr//4),max(1,gr//3),(225,238,242),-1)
    return out

def sim_choi_loa(img):
    out=img.copy().astype(np.float32); h,w=out.shape[:2]
    cx=np.random.randint(w//4,3*w//4); cy=np.random.randint(0,h//2)
    radius=np.random.randint(min(h,w)//3,min(h,w))
    Y,X=np.ogrid[:h,:w]
    dist=np.sqrt((X-cx)**2+(Y-cy)**2)
    glare=(np.exp(-dist/(radius*0.45))*255).clip(0,255).reshape(h,w,1).astype(np.float32)
    out=np.clip(out+glare*np.random.uniform(0.9,1.3),0,255).astype(np.uint8)
    return cv2.GaussianBlur(out,(13,13),0)

def sim_vet_xay(img):
    out=img.copy(); h,w=out.shape[:2]
    for _ in range(np.random.randint(8,22)):
        x1,y1=np.random.randint(0,w),np.random.randint(0,h)
        ang=np.random.uniform(0,np.pi); ll=np.random.randint(10,w//2)
        x2,y2=int(x1+ll*np.cos(ang)),int(y1+ll*np.sin(ang))
        sc=(np.random.randint(155,215),)*3
        cv2.line(out,(x1,y1),(x2,y2),sc,1)
    for _ in range(np.random.randint(3,15)):
        cx,cy=np.random.randint(0,w),np.random.randint(0,h)
        cv2.circle(out,(cx,cy),np.random.randint(1,4),(np.random.randint(20,70),)*3,-1)
    return cv2.GaussianBlur(out,(3,3),0)

def sim_nut_bun(img):
    out=img.copy(); h,w=out.shape[:2]
    for _ in range(np.random.randint(1,3)):
        cx=np.random.randint(w//4,3*w//4); cy=np.random.randint(h//4,3*h//4)
        rb=np.random.randint(8,20)
        bc=(np.random.randint(10,40),)*3
        cv2.circle(out,(cx,cy),rb,bc,-1)
        for i in range(np.random.randint(5,9)):
            ang=2*np.pi*i/8+np.random.uniform(-0.3,0.3)
            ll=np.random.randint(rb+5,rb+20)
            cv2.line(out,(cx,cy),(int(cx+ll*np.cos(ang)),int(cy+ll*np.sin(ang))),bc,1)
    return out

# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------
SCENARIOS = {
    "la_cay":   (sim_la_cay,   "La cay che o digit"),
    "bun_dat":  (sim_bun_dat,  "Bun dat dinh kinh"),
    "mo_kinh":  (sim_mo_kinh,  "Kinh bi mo / suong"),
    "choi_loa": (sim_choi_loa, "Choi loa anh sang"),
    "vet_xay":  (sim_vet_xay,  "Vet xuoc / bui ban"),
    "nut_bun":  (sim_nut_bun,  "Con trung / reu"),
}

# ------------------------------------------------------------------
# Mosaic
# ------------------------------------------------------------------
def _make_mosaic(orig_digit, results):
    cell_w,cell_h,lh=130,130,22
    cells=[]
    # goc
    tmp=np.zeros((cell_h+lh,cell_w,3),dtype=np.uint8)
    tmp[:cell_h,:]=cv2.resize(orig_digit,(cell_w,cell_h))
    cv2.putText(tmp,"[GOC]",(3,cell_h+lh-4),cv2.FONT_HERSHEY_SIMPLEX,0.38,(200,200,200),1)
    cells.append(tmp)
    for key,desc,path,ai_res in results:
        sim=cv2.imread(path)
        if sim is None: continue
        cell=np.zeros((cell_h+lh,cell_w,3),dtype=np.uint8)
        cell[:cell_h,:]=cv2.resize(sim,(cell_w,cell_h))
        color=(60,200,60) if "NaN" in ai_res else (80,80,220)
        cv2.putText(cell,desc[:18],(3,cell_h+lh-4),cv2.FONT_HERSHEY_SIMPLEX,0.34,color,1)
        cells.append(cell)
    n_cols=4
    rows=[]
    for i in range(0,len(cells),n_cols):
        row=cells[i:i+n_cols]
        while len(row)<n_cols: row.append(np.zeros_like(cells[0]))
        rows.append(np.hstack(row))
    mosaic=np.vstack(rows)
    path="test_images/nan_sim/mosaic_tat_ca.jpg"
    cv2.imwrite(path,mosaic)
    print(f"Anh mosaic: {path}")

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def run_test(source_img_path, save_only=False):
    base_img=cv2.imread(source_img_path)
    if base_img is None:
        print(f"[LOI] Khong doc duoc: {source_img_path}"); sys.exit(1)
    print(f"\nAnh goc: {source_img_path}  ({base_img.shape[1]}x{base_img.shape[0]} px)")

    # -- Segmentation: lay o digit dau tien ---
    try:
        from segmentation import segment_meter_digits
        digits=segment_meter_digits(base_img, num_digits=5, margin_ratio=0.22)
        if digits:
            digit_img=digits[0]
            print(f"Phan doan thanh cong: lay o digit dau tien ({digit_img.shape[1]}x{digit_img.shape[0]}px)")
        else:
            raise ValueError("Khong phan doan duoc digit")
    except Exception as e:
        print(f"[CANH BAO] Khong phan doan duoc ({e}). Dung anh crop thu cong.")
        h,w=base_img.shape[:2]
        digit_img=base_img[5:h-5, 5:w//5]

    # Resize ve 100x100 (kich thuoc standard)
    digit_img=cv2.resize(digit_img,(100,100))

    # -- Load model --
    reader=None
    if not save_only:
        try:
            from inference_lib import WaterMeterReader
            reader=WaterMeterReader("water_meter_modern.keras")
            print("Model AI da nap thanh cong.\n")
        except Exception as e:
            print(f"[CANH BAO] Khong tai model: {e}. Che do save-only.")
            save_only=True

    print("="*78)
    print(f"{'Tinh huong':<24}{'File':<35}{'Ket qua AI'}")
    print("="*78)

    results=[]
    for key,(fn,desc) in SCENARIOS.items():
        sim_img=fn(digit_img.copy())
        out_path=f"test_images/nan_sim/{key}.jpg"
        cv2.imwrite(out_path,sim_img)

        ai_result="N/A (save-only)"
        if not save_only and reader is not None:
            try:
                res=reader.predict(sim_img)
                digit=res.get("digit","?"); conf=res.get("confidence",0)
                if digit=="NaN":
                    ai_result=f"OK  -> NaN (phat hien dung, conf={conf:.0%})"
                else:
                    ai_result=f"FAIL-> doc nham='{digit}' (conf={conf:.0%})"
            except Exception as e:
                ai_result=f"[LOI] {e}"

        print(f"{desc:<24}{out_path:<35}{ai_result}")
        results.append((key,desc,out_path,ai_result))

    print("="*78)
    if not save_only:
        nan_ok=sum(1 for _,_,_,r in results if "NaN" in r)
        print(f"\nTong ket: {nan_ok}/{len(results)} tinh huong bi phat hien dung la NaN")
        if nan_ok < len(results):
            print("=> Model can them du lieu NaN de nang cao kha nang phat hien")
        else:
            print("=> Model phat hien tat ca tinh huong NaN chinh xac!")
    print(f"\nAnh gia lap luu tai: test_images/nan_sim/")
    _make_mosaic(digit_img, results)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",default="anh_2.jpg")
    parser.add_argument("--save-only",action="store_true")
    args=parser.parse_args()
    run_test(args.source,args.save_only)
