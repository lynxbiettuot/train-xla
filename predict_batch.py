from stardist.models import StarDist2D

models = [
    "2D_demo",
    "2D_versatile_fluo",
    "2D_versatile_he",
    "2D_paper_dsb2018",
    "2D_paper_junction"
]

for m in models:
    try:
        StarDist2D.from_pretrained(m)
        print(f"✔ Mô hình tồn tại: {m}")
    except:
        print(f"✘ Không tồn tại: {m}")
