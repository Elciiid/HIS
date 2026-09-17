"""Self-contained interactive HTML map (no CDN, works offline in any browser).

Generic machinery reused by Part B:
  layers    raster images (PNG, base64) stacked in the map, each toggleable
  markers   clickable points; clicking fills the side panel from the embedded JSON
            (``series`` entries are drawn as small SVG line charts)
  regions   clickable polygons (e.g. intervention footprints), same panel behaviour
Coordinates are given in metres of the model grid; the page maps them to pixels.
"""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import numpy as np

from .. import SYNTHETIC_WATERMARK


def png_b64(rgba: np.ndarray) -> str:
    """(ny, nx, 3|4) float array, row 0 = south -> base64 PNG (north up)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    buf = io.BytesIO()
    mpimg.imsave(buf, np.clip(np.asarray(rgba)[::-1], 0, 1), format="png")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def fig_b64(fig, dpi=110) -> str:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


_CSS = """
:root{--bg:#f6f5f2;--fg:#1d2327;--muted:#5b6770;--line:#d9d6cf;--accent:#1565c0;--warn:#b00020}
@media (prefers-color-scheme: dark){:root{--bg:#15181b;--fg:#e7e9ea;--muted:#9aa4ab;--line:#2c3237;--accent:#64b5f6;--warn:#ff6b81}}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--fg)}
header{padding:14px 20px;border-bottom:1px solid var(--line)}h1{font-size:18px;margin:0 0 4px}
.warn{color:var(--warn);font-weight:700;letter-spacing:.02em}
.wrap{display:flex;flex-wrap:wrap;gap:16px;padding:16px 20px}
.map{position:relative;flex:1 1 640px;max-width:100%;border:1px solid var(--line);background:#000;align-self:flex-start}
.map img.layer{position:absolute;left:0;top:0;width:100%;height:100%;image-rendering:auto}
.map img.base{position:relative;display:block;width:100%;height:auto}
.map svg{position:absolute;left:0;top:0;width:100%;height:100%}
.marker{cursor:pointer}.marker circle{stroke:#fff;stroke-width:2}
.region{cursor:pointer;fill-opacity:.25;stroke-width:2}
.side{flex:0 1 360px;min-width:280px;max-width:100%}
.card{border:1px solid var(--line);border-radius:6px;padding:12px;margin-bottom:12px;background:color-mix(in srgb,var(--bg) 92%,#888 8%)}
.card h2{font-size:14px;margin:0 0 8px}
table{border-collapse:collapse;width:100%;font-size:13px}td{padding:2px 4px;border-bottom:1px solid var(--line);vertical-align:top}
td:first-child{color:var(--muted);white-space:nowrap}
label{display:block;margin:3px 0}
.legend img{max-width:100%}
footer{padding:10px 20px;color:var(--muted);font:12px ui-monospace,monospace;border-top:1px solid var(--line);overflow-wrap:anywhere}
.chart{width:100%;height:140px}
"""

_JS = """
const D = JSON.parse(document.getElementById('payload').textContent);
const panel = document.getElementById('panel');
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function chart(series){
  const W=320,H=140,P=28; let xs=[],ys=[];
  series.lines.forEach(l=>{xs=xs.concat(l.x);ys=ys.concat(l.y);});
  const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(0,...ys),y1=Math.max(...ys)||1;
  const sx=x=>P+(x-x0)/((x1-x0)||1)*(W-P-6), sy=y=>H-P+(-(y-y0)/((y1-y0)||1))*(H-P-8);
  let s=`<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(series.title)}">`;
  s+=`<line x1="${P}" y1="${H-P}" x2="${W-6}" y2="${H-P}" stroke="currentColor" opacity=".4"/>`;
  s+=`<line x1="${P}" y1="8" x2="${P}" y2="${H-P}" stroke="currentColor" opacity=".4"/>`;
  s+=`<text x="${P}" y="${H-8}" font-size="10" fill="currentColor">${esc(series.xlabel)}</text>`;
  s+=`<text x="2" y="12" font-size="10" fill="currentColor">${y1.toPrecision(3)}</text>`;
  series.lines.forEach(l=>{
    const pts=l.x.map((x,i)=>`${sx(x).toFixed(1)},${sy(l.y[i]).toFixed(1)}`).join(' ');
    s+=`<polyline fill="none" stroke="${l.color}" stroke-width="1.8" ${l.dash?'stroke-dasharray="4 3"':''} points="${pts}"/>`;
  });
  let ly=16; series.lines.forEach(l=>{s+=`<text x="${W-6}" y="${ly}" text-anchor="end" font-size="10" fill="${l.color}">${esc(l.label)}</text>`;ly+=12;});
  return s+'</svg>';
}
function show(item){
  let h=`<h2>${esc(item.label)}</h2><table>`;
  for(const [k,v] of Object.entries(item.info||{})) h+=`<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`;
  h+='</table>';
  (item.series||[]).forEach(s=>{h+=`<div style="margin-top:8px"><b>${esc(s.title)}</b>${chart(s)}</div>`;});
  panel.innerHTML=h;
}
const svg=document.getElementById('overlay');
const toPx=(x,y)=>[x/D.extent[0]*D.px[0], (1-y/D.extent[1])*D.px[1]];
svg.setAttribute('viewBox',`0 0 ${D.px[0]} ${D.px[1]}`);
(D.regions||[]).forEach((r,i)=>{
  const p=document.createElementNS('http://www.w3.org/2000/svg','polygon');
  p.setAttribute('points',r.xy.map(q=>toPx(q[0],q[1]).join(',')).join(' '));
  p.setAttribute('class','region'); p.setAttribute('fill',r.color||'#ffd54f'); p.setAttribute('stroke',r.color||'#ffd54f');
  p.addEventListener('click',()=>show(r)); svg.appendChild(p);
});
(D.markers||[]).forEach(m=>{
  const g=document.createElementNS('http://www.w3.org/2000/svg','g'); g.setAttribute('class','marker');
  const [px,py]=toPx(m.x,m.y);
  const c=document.createElementNS('http://www.w3.org/2000/svg','circle');
  c.setAttribute('cx',px);c.setAttribute('cy',py);c.setAttribute('r',Math.max(4,D.px[0]/160));c.setAttribute('fill',m.color||'#e53935');
  const t=document.createElementNS('http://www.w3.org/2000/svg','title'); t.textContent=m.label;
  g.appendChild(c); g.appendChild(t); g.addEventListener('click',()=>show(m)); svg.appendChild(g);
});
document.querySelectorAll('input[data-layer]').forEach(cb=>cb.addEventListener('change',()=>{
  document.getElementById(cb.dataset.layer).hidden=!cb.checked;}));
document.querySelectorAll('input[data-opacity]').forEach(sl=>sl.addEventListener('input',()=>{
  document.getElementById(sl.dataset.opacity).style.opacity=sl.value;}));
"""


def write_html(path: str | Path, title: str, base_png: str, extent_m: tuple[float, float], px: tuple[int, int],
               layers: list[dict], markers: list[dict], regions: list[dict], prov, legend_png: str | None = None,
               summary: dict | None = None, notes: list[str] | None = None) -> Path:
    """layers: [{"id","name","png","visible","opacity"}]; markers: [{"x","y","label","info","series"}]."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"extent": list(extent_m), "px": list(px), "markers": markers, "regions": regions}
    lay_html = "".join(
        f'<img class="layer" id="{html.escape(l["id"])}" alt="{html.escape(l["name"])}" '
        f'src="data:image/png;base64,{l["png"]}" style="opacity:{l.get("opacity", 1)}"'
        f'{"" if l.get("visible", True) else " hidden"}>' for l in layers)
    ctrl = "".join(
        f'<label><input type="checkbox" data-layer="{html.escape(l["id"])}"{" checked" if l.get("visible", True) else ""}> '
        f'{html.escape(l["name"])}</label><input type="range" min="0" max="1" step="0.05" value="{l.get("opacity", 1)}" '
        f'data-opacity="{html.escape(l["id"])}" aria-label="opacity of {html.escape(l["name"])}">' for l in layers)
    summ = ""
    if summary:
        summ = '<div class="card"><h2>Summary</h2><table>' + "".join(
            f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>" for k, v in summary.items()) + "</table></div>"
    note = "".join(f"<li>{html.escape(n)}</li>" for n in (notes or []))
    warn = f'<div class="warn">{html.escape(SYNTHETIC_WATERMARK)}</div>' if prov.synthetic else ""
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>{_CSS}</style></head><body>
<header><h1>{html.escape(title)}</h1>{warn}</header>
<div class="wrap">
 <div class="map"><img class="base" alt="basemap" src="data:image/png;base64,{base_png}">{lay_html}
  <svg id="overlay" aria-label="clickable markers"></svg></div>
 <div class="side">
  <div class="card" id="panel"><h2>Details</h2><p>Click a marker or region on the map.</p></div>
  <div class="card"><h2>Layers</h2>{ctrl}</div>
  {summ}
  {'<div class="card legend"><h2>Legend</h2><img alt="legend" src="data:image/png;base64,' + legend_png + '"></div>' if legend_png else ''}
  {'<div class="card"><h2>Notes</h2><ul>' + note + '</ul></div>' if note else ''}
 </div>
</div>
<footer>{html.escape(prov.line())}</footer>
<script type="application/json" id="payload">{json.dumps(payload).replace("</", "<\\/")}</script>
<script>{_JS}</script></body></html>"""
    path.write_text(doc, encoding="utf-8")
    return path
