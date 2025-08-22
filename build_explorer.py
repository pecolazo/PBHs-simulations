#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, numpy as np, h5py
from pathlib import Path
from scipy.spatial import cKDTree
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# =========================
# Salidas
# =========================
OUTDIR  = Path("docs")
HALODIR = OUTDIR / "halos"
OUTDIR.mkdir(parents=True, exist_ok=True)
HALODIR.mkdir(parents=True, exist_ok=True)

# =========================
# Halos a generar (1..18, sin 4 ni 11)
# =========================
EXCLUDE   = {4, 11}
HALO_IDS  = [i for i in range(1, 19) if i not in EXCLUDE]

# =========================
# Rutas de HALOS (según tu estructura)
#   raíz:   /home/pcolazo/doctorado/Paper_I/Volumenes_corte_cerca_de_halos_halos
#   subdir: cdm | nb2 | fct
#   file :  volumen_<subdir>_<id>.hdf5  (p.ej. volumen_cdm_7.hdf5)
# =========================
HALO_ROOT = "/home/pcolazo/doctorado/Paper_I/Volumenes_corte_cerca_de_halos_halos"
SUBDIR    = {"CDM": "cdm", "NB": "nb2", "FCT": "fct"}
FNPAT     = {"CDM": "volumen_cdm_{hid}.hdf5",
             "NB":  "volumen_nb2_{hid}.hdf5",
             "FCT": "volumen_fct_{hid}.hdf5"}

def resolve_path(model: str, halo_id: int) -> str:
    sub  = SUBDIR[model]
    fn   = FNPAT[model].format(hid=halo_id)
    path = f"{HALO_ROOT}/{sub}/{fn}"
    return path

# Construye [(slug, title, p_cdm, p_nb, p_fct)]
HALOS = []
for hid in HALO_IDS:
    slug  = f"h{hid:03d}"
    title = f"Halo {hid}"
    p_cdm = resolve_path("CDM", hid)
    p_nb  = resolve_path("NB",  hid)
    p_fct = resolve_path("FCT", hid)
    HALOS.append((slug, title, p_cdm, p_nb, p_fct))

# =========================
# Full boxes (CDM/FCT) – tus rutas
# =========================
FULLS = [
    ("full_box", "Full Box — CDM vs FCT",
     "/mnt/projects/biasPBH/simus/viejs/cdm-1024-paper-I_0011.hdf5",
     "/mnt/projects/biasPBH/simus/viejs/fct-1024-paper-I_0011.hdf5"),
]

# =========================
# Muestreo
# =========================
SAMPLE_MODE  = "fixed"    # "ratio" | "fixed"
SAMPLE_RATIO = 0.01       # si ratio: 1%
SAMPLE_FIXED = 120_000    # si fixed
SAMPLE_MIN   = 5_000

POS_CAND = [
  "/PartType1/Coordinates","/particles/pos","/pos",
  "/Coords","/data/pos","PartType1/Coordinates"
]
KNN_K    = 32
RNG_SEED = 12345

# Paletas (según tus colores base)
RGB_CDM = (30, 144, 255)   # dodgerblue
RGB_NB  = (34, 139, 34)    # forest green
RGB_FCT = (220, 20, 60)    # crimson

# =========================
# HDF5 helpers
# =========================
def find_pos(h5):
    for c in POS_CAND:
        try:
            ds = h5[c]
            if getattr(ds, "shape", None) and len(ds.shape)==2 and ds.shape[1]==3:
                return ds
        except KeyError:
            pass
    found = []
    def _visit(n,o):
        if hasattr(o,"shape") and len(o.shape)==2 and o.shape[1]==3:
            found.append(n)
    h5.visititems(_visit)
    if found:
        return h5[found[0]]
    raise KeyError("No se encontró dataset Nx3 de coordenadas")

def dataset_len(p):
    with h5py.File(p,"r") as f:
        return find_pos(f).shape[0]

def load_rows(p, idx):
    idx = np.unique(np.asarray(idx, dtype=np.int64))
    idx.sort()
    if idx.size == 0:
        return np.zeros((0,3), np.float32)
    with h5py.File(p,"r") as f:
        ds = find_pos(f)
        parts = []
        gaps = np.where(np.diff(idx)>1)[0] + 1
        runs = np.split(idx, gaps)
        for run in runs:
            s, e = run[0], run[-1]+1
            arr  = ds[s:e, :]
            if e - s != len(run):
                arr = arr[(run - s), :]
            parts.append(np.asarray(arr, np.float32))
    pos = np.round(np.concatenate(parts, axis=0), 3).astype(np.float32)
    return pos

def pick_target(N):
    if SAMPLE_MODE == "ratio":
        return max(SAMPLE_MIN, min(N, int(N * SAMPLE_RATIO)))
    return min(N, SAMPLE_FIXED)

def downsample(N, target, seed):
    if N <= target:
        return np.arange(N, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.choice(N, size=target, replace=False)

# =========================
# Densidad (log) y estilo
# =========================
def density_knn(pos, k=KNN_K):
    tree = cKDTree(pos)
    rk   = tree.query(pos, k=k)[0][:, -1]
    dens = 1.0 / (rk**3 + 1e-30)
    dens = np.log10(dens + 1e-12)
    p_lo, p_hi = np.percentile(dens, [0.5, 99.5])
    dens = (dens - p_lo) / (p_hi - p_lo + 1e-30)
    dens = np.clip(dens, 0, 1)
    dens = dens**0.8
    return dens

def colorscale(rgb, floor=35):
    r,g,b = rgb
    return [[0.0, f"rgb({floor},{floor},{floor})"], [1.0, f"rgb({r},{g},{b})"]]

def style_scene(s, rx, ry, rz):
    s.update(
        bgcolor="black",
        xaxis=dict(showbackground=False, showgrid=False, zeroline=False,
                   showticklabels=False, ticks="", title="", range=rx),
        yaxis=dict(showbackground=False, showgrid=False, zeroline=False,
                   showticklabels=False, ticks="", title="", range=ry),
        zaxis=dict(showbackground=False, showgrid=False, zeroline=False,
                   showticklabels=False, ticks="", title="", range=rz),
        dragmode="orbit", aspectmode="cube", uirevision="sync"
    )

# =========================
# HTML builders
# =========================
def build_compare_html_triple(title, pos_a, pos_b, pos_c, outpath,
                              rgb_a=RGB_CDM, rgb_b=RGB_NB, rgb_c=RGB_FCT):
    dens_a = density_knn(pos_a); dens_b = density_knn(pos_b); dens_c = density_knn(pos_c)
    xmin = min(pos_a[:,0].min(), pos_b[:,0].min(), pos_c[:,0].min()); xmax = max(pos_a[:,0].max(), pos_b[:,0].max(), pos_c[:,0].max())
    ymin = min(pos_a[:,1].min(), pos_b[:,1].min(), pos_c[:,1].min()); ymax = max(pos_a[:,1].max(), pos_b[:,1].max(), pos_c[:,1].max())
    zmin = min(pos_a[:,2].min(), pos_b[:,2].min(), pos_c[:,2].min()); zmax = max(pos_a[:,2].max(), pos_b[:,2].max(), pos_c[:,2].max())

    fig = make_subplots(rows=1, cols=3,
                        specs=[[{"type":"scene"},{"type":"scene"},{"type":"scene"}]],
                        horizontal_spacing=0.0)

    fig.add_trace(go.Scatter3d(x=pos_a[:,0], y=pos_a[:,1], z=pos_a[:,2],
                               mode="markers",
                               marker=dict(size=1.4, opacity=0.95, color=dens_a,
                                           colorscale=colorscale(rgb_a), cmin=0, cmax=1),
                               hoverinfo="skip", showlegend=False), row=1, col=1)

    fig.add_trace(go.Scatter3d(x=pos_b[:,0], y=pos_b[:,1], z=pos_b[:,2],
                               mode="markers",
                               marker=dict(size=1.4, opacity=0.95, color=dens_b,
                                           colorscale=colorscale(rgb_b), cmin=0, cmax=1),
                               hoverinfo="skip", showlegend=False), row=1, col=2)

    fig.add_trace(go.Scatter3d(x=pos_c[:,0], y=pos_c[:,1], z=pos_c[:,2],
                               mode="markers",
                               marker=dict(size=1.4, opacity=0.95, color=dens_c,
                                           colorscale=colorscale(rgb_c), cmin=0, cmax=1),
                               hoverinfo="skip", showlegend=False), row=1, col=3)

    style_scene(fig.layout.scene,  [xmin,xmax],[ymin,ymax],[zmin,zmax])
    style_scene(fig.layout.scene2, [xmin,xmax],[ymin,ymax],[zmin,zmax])
    style_scene(fig.layout.scene3, [xmin,xmax],[ymin,ymax],[zmin,zmax])

    fig.update_layout(height=600, width=1800, margin=dict(l=0,r=0,t=46,b=0),
                      paper_bgcolor="black", plot_bgcolor="black",
                      font=dict(color="#EAEAEA"),
                      title=dict(text=title, x=0.5, xanchor="center",
                                 font=dict(size=18, color="#F5F5F5")))

    div_id = "cmp_triple"
    post_js = f"""
    (function(){{
      var gd = document.getElementById('{div_id}');
      var syncing = false;
      gd.on('plotly_relayout', function(e){{
        if(syncing) return;
        var cam = e['scene.camera'] || e['scene2.camera'] || e['scene3.camera'];
        if(!cam) return;
        syncing = true;
        Plotly.relayout(gd, {{
          'scene.camera': cam,
          'scene2.camera': cam,
          'scene3.camera': cam
        }}).then(function(){{ syncing=false; }});
      }});
    }})();"""

    pio.write_html(fig, file=str(outpath), include_plotlyjs="cdn", full_html=True,
                   div_id=div_id, post_script=post_js,
                   config={"responsive":True,"displaylogo":False,
                           "scrollZoom":True,"doubleClick":"reset"})

    # PWA + fondo
    extra_head = """
<link rel="manifest" href="../manifest.json">
<meta name="theme-color" content="#0b0f14">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<link rel="apple-touch-icon" href="../icon-192.png">
<meta name="apple-mobile-web-app-capable" content="yes">
<style> body{background-color:#000!important;margin:0;padding:0;} </style>
<script> if("serviceWorker" in navigator){ navigator.serviceWorker.register("../sw.js",{scope:"../"}); } </script>
"""
    html = outpath.read_text(encoding="utf-8")
    if "serviceWorker.register(" not in html:
        html = html.replace("</head>", extra_head + "\n</head>") if "</head>" in html \
               else html.replace("<body>", "<body>\n" + extra_head)
        outpath.write_text(html, encoding="utf-8")

def build_compare_html_double(title, pos_a, pos_b, outpath,
                              rgb_a=RGB_CDM, rgb_b=RGB_FCT):
    dens_a = density_knn(pos_a); dens_b = density_knn(pos_b)
    xmin = min(pos_a[:,0].min(), pos_b[:,0].min()); xmax = max(pos_a[:,0].max(), pos_b[:,0].max())
    ymin = min(pos_a[:,1].min(), pos_b[:,1].min()); ymax = max(pos_a[:,1].max(), pos_b[:,1].max())
    zmin = min(pos_a[:,2].min(), pos_b[:,2].min()); zmax = max(pos_a[:,2].max(), pos_b[:,2].max())

    fig = make_subplots(rows=1, cols=2, specs=[[{"type":"scene"},{"type":"scene"}]], horizontal_spacing=0.0)

    fig.add_trace(go.Scatter3d(x=pos_a[:,0], y=pos_a[:,1], z=pos_a[:,2],
                               mode="markers",
                               marker=dict(size=1.4, opacity=0.95, color=dens_a,
                                           colorscale=colorscale(rgb_a), cmin=0, cmax=1),
                               hoverinfo="skip", showlegend=False), row=1, col=1)

    fig.add_trace(go.Scatter3d(x=pos_b[:,0], y=pos_b[:,1], z=pos_b[:,2],
                               mode="markers",
                               marker=dict(size=1.4, opacity=0.95, color=dens_b,
                                           colorscale=colorscale(rgb_b), cmin=0, cmax=1),
                               hoverinfo="skip", showlegend=False), row=1, col=2)

    style_scene(fig.layout.scene,  [xmin,xmax],[ymin,ymax],[zmin,zmax])
    style_scene(fig.layout.scene2, [xmin,xmax],[ymin,ymax],[zmin,zmax])

    fig.update_layout(height=600, width=1200, margin=dict(l=0,r=0,t=46,b=0),
                      paper_bgcolor="black", plot_bgcolor="black",
                      font=dict(color="#EAEAEA"),
                      title=dict(text=title, x=0.5, xanchor="center",
                                 font=dict(size=18, color="#F5F5F5")))

    div_id = "cmp_double"
    post_js = f"""
    (function(){{
      var gd = document.getElementById('{div_id}');
      var syncing = false;
      gd.on('plotly_relayout', function(e){{
        if(syncing) return;
        var cam = e['scene.camera'] || e['scene2.camera'];
        if(!cam) return;
        syncing = true;
        Plotly.relayout(gd, {{
          'scene.camera': cam,
          'scene2.camera': cam
        }}).then(function(){{ syncing=false; }});
      }});
    }})();"""

    pio.write_html(fig, file=str(outpath), include_plotlyjs="cdn", full_html=True,
                   div_id=div_id, post_script=post_js,
                   config={"responsive":True,"displaylogo":False,
                           "scrollZoom":True,"doubleClick":"reset"})

    extra_head = """
<link rel="manifest" href="../manifest.json">
<meta name="theme-color" content="#0b0f14">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<link rel="apple-touch-icon" href="../icon-192.png">
<meta name="apple-mobile-web-app-capable" content="yes">
<style> body{background-color:#000!important;margin:0;padding:0;} </style>
<script> if("serviceWorker" in navigator){ navigator.serviceWorker.register("../sw.js",{scope:"../"}); } </script>
"""
    html = outpath.read_text(encoding="utf-8")
    if "serviceWorker.register(" not in html:
        html = html.replace("</head>", extra_head + "\n</head>") if "</head>" in html \
               else html.replace("<body>", "<body>\n" + extra_head)
        outpath.write_text(html, encoding="utf-8")

# =========================
# Build
# =========================
def build_all():
    cards = []

    # 0) Filtro de existencia (por si faltan archivos)
    _halos_ok = []
    for (slug, title, p_cdm, p_nb, p_fct) in HALOS:
        if all(os.path.exists(x) for x in (p_cdm, p_nb, p_fct)):
            _halos_ok.append((slug, title, p_cdm, p_nb, p_fct))
        else:
            print(f"[WARN] Falta archivo para {slug}:",
                  *[x for x in (p_cdm, p_nb, p_fct) if not os.path.exists(x)], sep="\n       ")
    halos = _halos_ok

    # 1) FULL BOX PAGES (doble)
    for j,(fid, title, p_cdm, p_fct) in enumerate(FULLS):
        if not (os.path.exists(p_cdm) and os.path.exists(p_fct)):
            print(f"[WARN] full_box faltan archivos: {p_cdm} | {p_fct}")
            continue
        n_cdm = dataset_len(p_cdm); n_fct = dataset_len(p_fct)
        tgt_cdm = pick_target(n_cdm); tgt_fct = pick_target(n_fct)
        print(f"[{fid}] CDM N={n_cdm:,} -> sample {tgt_cdm:,} | FCT N={n_fct:,} -> sample {tgt_fct:,}")

        idx_cdm = downsample(n_cdm, tgt_cdm, seed=RNG_SEED + j*2 + 1000)
        idx_fct = downsample(n_fct, tgt_fct, seed=RNG_SEED + j*2 + 1001)

        pos_cdm = load_rows(p_cdm, idx_cdm)
        pos_fct = load_rows(p_fct, idx_fct)

        out = HALODIR / f"{fid}.html"
        build_compare_html_double(title, pos_cdm, pos_fct, out)
        cards.append((fid, title))

    # 2) HALO PAGES (triple)
    for j,(slug, title, p_cdm, p_nb, p_fct) in enumerate(halos):
        n_cdm = dataset_len(p_cdm)
        n_nb  = dataset_len(p_nb)
        n_fct = dataset_len(p_fct)

        tgt_cdm = pick_target(n_cdm)
        tgt_nb  = pick_target(n_nb)
        tgt_fct = pick_target(n_fct)

        print(f"[{slug}] CDM N={n_cdm:,} -> sample {tgt_cdm:,} | NB N={n_nb:,} -> sample {tgt_nb:,} | FCT N={n_fct:,} -> sample {tgt_fct:,}")

        idx_cdm = downsample(n_cdm, tgt_cdm, seed=RNG_SEED + j*3 + 0)
        idx_nb  = downsample(n_nb,  tgt_nb,  seed=RNG_SEED + j*3 + 1)
        idx_fct = downsample(n_fct, tgt_fct, seed=RNG_SEED + j*3 + 2)

        pos_cdm = load_rows(p_cdm, idx_cdm)
        pos_nb  = load_rows(p_nb,  idx_nb)
        pos_fct = load_rows(p_fct, idx_fct)

        out = HALODIR / f"{slug}.html"
        build_compare_html_triple(f"{title} — CDM vs NB vs FCT",
                                  pos_cdm, pos_nb, pos_fct, out,
                                  rgb_a=RGB_CDM, rgb_b=RGB_NB, rgb_c=RGB_FCT)
        cards.append((slug, title))

    # 3) HUB INDEX
    INDEX = f"""<!doctype html><html><head>
<meta charset="utf-8"><title>PBHs — Explorer</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="manifest" href="./manifest.json">
<meta name="theme-color" content="#0b0f14">
<link rel="apple-touch-icon" href="./icon-192.png">
<meta name="apple-mobile-web-app-capable" content="yes">
<style>
  body{{background:#000;color:#eaeaea;font:16px/1.4 system-ui,Segoe UI,Roboto,sans-serif;margin:0;padding:24px;}}
  h1{{margin:0 0 16px 0;font-size:22px;color:#f5f5f5}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px}}
  .card{{background:#0b0f14;border:1px solid #1d2630;border-radius:16px;padding:14px;text-decoration:none;color:#eaeaea}}
  .card:hover{{outline:1px solid #3a83f6}}
  .tag{{display:inline-block;font-size:12px;opacity:.8;margin-top:4px}}
</style>
<script> if("serviceWorker" in navigator){{ navigator.serviceWorker.register("./sw.js", {{scope:"./"}}); }} </script>
</head><body>
<h1>PBHs — Explorer (Full Boxes + Halos)</h1>
<div class="grid">
{''.join([f'<a class="card" href="./halos/{slug}.html"><div style="font-weight:600">{title}</div><div class="tag">{slug}</div></a>' for slug,title in cards])}
</div>
</body></html>"""
    (OUTDIR/"index.html").write_text(INDEX, encoding="utf-8")
    print(f"Listo: {OUTDIR/'index.html'}  + {len(cards)} páginas en {HALODIR}/")

# =========================
# Main
# =========================
if __name__ == "__main__":
    build_all()
