"""由 papers.json 生成：① README.md 里的论文一览表（两个标记之间）② docs/guide.pdf 入门指引。
新增一篇论文后运行：python3 tools/build.py     （需要 pip install weasyprint "qrcode[pil]"）"""
import os, json, re, io, base64, datetime
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); URL = "github.com/XYan-Alex/quant-paper-replication"
P = json.load(open(os.path.join(ROOT, "papers.json"), encoding="utf-8")); N = len(P)
# ---------- README ----------
tb = "| # | 论文 | 方向 | 复现结论 |\n|---|---|---|---|\n" + "\n".join(f"| [{p['id']}]({p['dir']}) | {p['title']}（{p['source']}） | {p['topic']} | {p['verdict']} |" for p in P)
tb += "\n| [附](overfitting_check) | PBO、DSR 过拟合检验 | — | 方法与结果见文件夹内说明 |"
tree = "\n".join(f"{p['dir'] + '/':25s}{p['title']}" for p in P)
f = os.path.join(ROOT, "README.md"); s = open(f, encoding="utf-8").read()
s = re.sub(r"(<!-- PAPERS:START -->).*?(<!-- PAPERS:END -->)", lambda m: f"{m.group(1)}\n{tb}\n{m.group(2)}", s, flags=re.S)
s = re.sub(r"(<!-- TREE:START -->).*?(<!-- TREE:END -->)", lambda m: f"{m.group(1)}\n```\n{tree}\noverfitting_check/        PBO / DSR 过拟合检验\nresults/                 每篇的结果文件（JSON / CSV）\ndata/                    数据目录（不进版本库，见 data/README.md）\npapers.json              论文清单（README 一览表和指引 PDF 都由它生成）\ntools/build.py           重新生成一览表和 docs/guide.pdf\n_template/               新增论文时用的 README 模板\n```\n{m.group(2)}", s, flags=re.S)
s = re.sub(r"<!-- N -->\d+<!-- /N -->", f"<!-- N -->{N}<!-- /N -->", s); open(f, "w", encoding="utf-8").write(s); print("README 已更新：", N, "篇")
# ---------- PDF ----------
try:
    import qrcode; from weasyprint import HTML
except ImportError: raise SystemExit("跳过 PDF：缺 weasyprint 或 qrcode")
q = qrcode.QRCode(border=1, box_size=8); q.add_data("https://" + URL); q.make(); b = io.BytesIO(); q.make_image(fill_color="#5A4636", back_color="#F3E6D3").save(b, "PNG"); QR = base64.b64encode(b.getvalue()).decode()
tr = "".join(f"<tr><td class='n'>{p['id']}</td><td class='p'>{p['title']}</td><td class='r'>{p['source']}</td><td>{p['topic']}</td></tr>" for p in P)
day = datetime.date.today().strftime("%Y-%m-%d")
html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><style>
@page{{size:A4;margin:0}} @page body{{margin:18mm 18mm 16mm;background:#FBF6EE}}
*{{box-sizing:border-box}} body{{margin:0;font-family:"Noto Sans CJK SC",sans-serif;color:#2a211c}}
.c1{{width:210mm;height:297mm;padding:22mm 18mm;position:relative;background:#1A1412;color:#F3E6D3;page-break-after:always;overflow:hidden}}
.tag{{font-size:11pt;letter-spacing:.3em;color:#C8643B;font-weight:700}}
h1{{font-size:40pt;line-height:1.2;margin:16mm 0 6mm;font-weight:900}} h1 span{{color:#E07A4F}}
.sub{{font-size:13.5pt;color:#A88B74;line-height:1.75;max-width:155mm}}
.addr{{margin-top:18mm;border-left:2.5pt solid #C8643B;padding:2mm 0 2mm 6mm}} .addr .t{{font-size:10.5pt;color:#A88B74}} .url{{font-family:"DejaVu Sans Mono",monospace;font-size:14pt;font-weight:700;margin-top:2.5mm}}
.kp{{display:flex;gap:6mm;margin-top:18mm}} .kp div{{flex:1;border-top:2pt solid #C8643B;padding-top:4mm}} .kp b{{display:block;font-size:26pt;color:#E07A4F;font-weight:900}} .kp span{{font-size:10.5pt;color:#A88B74;line-height:1.6}}
.ft{{position:absolute;left:18mm;right:18mm;bottom:14mm;display:flex;gap:6mm;align-items:flex-end;border-top:.5pt solid #5A4636;padding-top:4mm}}
.ft .tx{{flex:1;font-size:9pt;color:#7d6a5c;line-height:1.65}} .ft img{{width:17mm;height:17mm;border-radius:1mm;opacity:.9}}
.b{{page:body}} h2{{font-size:19pt;margin:0 0 6mm;font-weight:900;color:#1A1412}} h2 i{{font-style:normal;color:#C8643B}}
.st{{display:flex;gap:5mm;margin-bottom:5mm;page-break-inside:avoid}} .st .k{{flex:0 0 10mm;height:10mm;border-radius:50%;background:#C8643B;color:#fff;font-weight:900;font-size:13pt;text-align:center;line-height:10mm}}
.st .x{{flex:1}} .st .x b.h{{font-size:12.5pt;display:block;margin-bottom:1mm;color:#1A1412}} .st .x span{{font-size:10.5pt;color:#5A4636;line-height:1.65}}
code{{font-family:"DejaVu Sans Mono",monospace;font-size:9pt;background:#EFE4D3;padding:.6mm 1.6mm;border-radius:1mm}}
pre{{font-family:"DejaVu Sans Mono",monospace;font-size:9pt;background:#1A1412;color:#F3E6D3;padding:3.5mm 4mm;border-radius:2mm;margin:2mm 0;line-height:1.6}}
table{{width:100%;border-collapse:collapse;font-size:9.8pt}} thead{{display:table-header-group}} th{{text-align:left;font-size:9pt;color:#A88B74;border-bottom:1pt solid #C8643B;padding:1.5mm 2mm}}
tr{{page-break-inside:avoid}} td{{padding:1.9mm 2mm;border-bottom:.4pt solid #E2D5C2;vertical-align:top;line-height:1.45}} td.n{{color:#C8643B;font-weight:900;width:9mm}} td.p{{font-weight:700;width:58mm}} td.r{{color:#5A4636;width:42mm}}
.note{{font-size:9.5pt;color:#8a7868;margin-top:4mm;line-height:1.65}} .end{{font-size:8.6pt;color:#8a7868;border-top:.5pt solid #D8C9B4;padding-top:2mm;margin-top:3mm;line-height:1.5}}
</style></head><body>
<div class="c1"><div class="tag">量化舵 · 开源仓库</div>
<h1>量化论文<br><span>从数据到代码</span><br>完整复现</h1>
<div class="sub">视频里讲过的每一篇论文，复现用到的全部代码、每一次运行的结果，都放在一个公开的代码仓库里，可以查看、下载、自己跑一遍。仓库持续更新。</div>
<div class="addr"><div class="t">仓库地址（GitHub，公开、免费，仅用于学习交流）</div><div class="url">{URL}</div></div>
<div class="kp"><div><b>{N}</b><span>篇论文已复现<br>持续增加中</span></div><div><b>逐篇</b><span>有说明和结果表<br>含与论文的差异</span></div><div><b>对照</b><span>论文数字与复现数字<br>并排列出</span></div></div>
<div class="ft"><div class="tx">量化方法科普 · 全部为历史模拟回测，非实盘，不代表未来表现，不构成任何投资建议。<br>不荐股、不提供任何交易服务；仓库内只有研究代码与结果。指引更新于 {day}，最新内容以仓库首页为准。</div><img src="data:image/png;base64,{QR}"></div></div>
<div class="b"><h2><i>三步</i>进入仓库</h2>
<div class="st"><div class="k">1</div><div class="x"><b class="h">打开仓库首页</b><span>在浏览器地址栏输入 <code>{URL}</code>（或扫封面右下角的码）。只看代码和结果不需要登录。页面往下拉就是总说明（README），里面有全部论文的一览表。</span></div></div>
<div class="st"><div class="k">2</div><div class="x"><b class="h">点进想看的那一篇</b><span>文件列表里每篇论文一个文件夹，名字以编号开头（如 <code>02_alpha_gpt</code>）。点进去自动显示这一篇的说明：论文在做什么、每个脚本的作用、怎么运行、结果表。点任意 <code>.py</code> 文件即可看代码；结果文件在 <code>results</code> 文件夹。</span></div></div>
<div class="st"><div class="k">3</div><div class="x"><b class="h">想自己跑一遍</b><span>点首页绿色的 Code 按钮 → Download ZIP 下载全部代码；或者用命令行：</span>
<pre>git clone https://{URL}.git
cd quant-paper-replication
pip install -r requirements.txt</pre>
<span>数据不在仓库里（体积大、有版权），下载方法写在 <code>data/README.md</code>。全部代码在普通电脑的 CPU 上就能跑。</span></div></div>
<h2 style="margin-top:5mm">目前<i>有什么</i></h2>
<table><thead><tr><th>#</th><th>论文</th><th>来源</th><th>方向</th></tr></thead><tbody>{tr}</tbody></table>
<div class="note">另有一份过拟合检验（PBO、DSR），在 <code>overfitting_check</code> 文件夹。以上为截至 {day} 的清单；之后新增的论文按编号顺延，以仓库首页的一览表为准。</div>
<div class="end">发现问题可以在仓库的 Issues 里留言。抖音 / B 站：量化舵 · 量化方法科普，不构成任何投资建议。</div></div>
</body></html>"""
os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True); HTML(string=html).write_pdf(os.path.join(ROOT, "docs", "guide.pdf")); print("docs/guide.pdf 已生成")
