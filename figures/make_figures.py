"""V1.5 figure assets from pinned source data (Python/Matplotlib only).

The schematic uses declared rational parameters, not observed eigenvalues.
All numerical plots consume copied JSON without modifying it.  The actual
alpha degree denotes FIRST success over complete blocks, not a monotone tail.
"""
from pathlib import Path
import csv
import base64
import hashlib
import json
import math
import sys
import re
from io import BytesIO
from fractions import Fraction

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from fontTools.ttLib import TTFont
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from matplotlib.ticker import FixedLocator, ScalarFormatter, LogLocator, NullFormatter, NullLocator

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
QA = ROOT / 'qa'
OUT.mkdir(exist_ok=True)
QA.mkdir(exist_ok=True)

# Current user typography instruction overrides Arial/bold personal defaults.
# Computer Modern is the source family of Latin Modern and matches its shapes.
mpl.rcParams.update({
    'font.family': 'cmr10', 'font.size': 9.5, 'mathtext.fontset': 'cm',
    'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False,
    'axes.labelsize': 9.5, 'axes.titlesize': 10., 'axes.labelweight': 'normal',
    'axes.titleweight': 'normal', 'xtick.labelsize': 9., 'ytick.labelsize': 9.,
    'legend.fontsize': 8.5, 'legend.title_fontsize': 8.5, 'legend.frameon': False,
    'axes.linewidth': 1., 'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.width': .8, 'ytick.major.width': .8,
    'xtick.minor.width': .5, 'ytick.minor.width': .5,
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
    'xtick.minor.size': 2., 'ytick.minor.size': 2.,
    'xtick.major.pad': 3., 'ytick.major.pad': 3.,
    'xtick.top': False, 'ytick.right': False,
    'text.color': 'black', 'axes.labelcolor': 'black',
    'xtick.color': 'black', 'ytick.color': 'black',
    'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
})

BLUE = '#1976A3'
ORANGE = '#D55E00'
GREEN = '#16856A'
PURPLE = '#845AA1'
GRAY = '#474747'
SIZE = (5.65, 3.20)
DATA = ROOT / 'inputs'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def row_title(ax, text):
    ax.set_title(text, loc='left', pad=8, fontsize=10, fontweight='normal')


def positive(values):
    a = np.asarray(values, dtype=float)
    assert np.all(np.isfinite(a)) and np.all(a > 0)
    return a


limits = {}
layout_records = {}


def check_panel_alignment(fig, stem, tolerance_pt=1.5):
    """Self-contained check for the equal-size two-panel rows used here.

    Checks final rendered plot-area width, height and shared top/bottom edges;
    it is a geometric self-check, not an independent visual review.
    """
    fig.canvas.draw()
    width,height=fig.get_size_inches()*72
    boxes=[np.array([a.get_position().x0*width,a.get_position().y0*height,
                     a.get_position().x1*width,a.get_position().y1*height])
           for a in fig.axes]
    assert len(boxes)==2, 'This bounded check expects two Cartesian panels.'
    left,right=boxes
    deviations={
        'bottom':abs(left[1]-right[1]),'top':abs(left[3]-right[3]),
        'width':abs((left[2]-left[0])-(right[2]-right[0])),
        'height':abs((left[3]-left[1])-(right[3]-right[1])),
    }
    nonoverlap=left[2]<right[0]
    passed=all(v<=tolerance_pt for v in deviations.values()) and nonoverlap
    record={'backend':'Python/Matplotlib','verdict':'PASS' if passed else 'FAIL',
            'review_kind':'author geometric self-check',
            'tolerance_pt':tolerance_pt,'deviations_pt':deviations,
            'nonoverlap':bool(nonoverlap),
            'figure_size_pt':[float(width),float(height)],
            'exemptions':getattr(fig,'_audit_exemptions',[]),
            'panels':[{'id':label,'bbox_pt':b.tolist()}
                      for label,b in zip(['a','b'],boxes)]}
    (QA/f'{stem}.alignment.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    assert passed, record
    return record


def save(fig, stem, *, bound_axes=()):
    # Verify actual final axes and all four 1-pt frame sides after layout.
    all_axes=list(fig.axes)+[child for ax in fig.axes for child in ax.child_axes]
    for ax in all_axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.)
        ax.tick_params(top=False, right=False)
    fig.canvas.draw()
    alignment = check_panel_alignment(fig,stem)
    layout_records[stem] = alignment
    for n, ax in enumerate(all_axes):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        series = []
        for line in ax.lines:
            if line.get_gid() == 'schematic-boundary':
                continue
            x, y = np.asarray(line.get_xdata(), dtype=float), np.asarray(line.get_ydata(), dtype=float)
            if len(x) and len(y):
                if n in bound_axes:
                    assert np.min(x) >= xmin and np.max(x) <= xmax
                    assert np.min(y) > ymin and np.max(y) < ymax, (stem, n, line.get_label(), ymin, ymax, min(y), max(y))
                series.append({'label': line.get_label(), 'n': len(x),
                               'x_extent': [float(min(x)), float(max(x))],
                               'y_extent': [float(min(y)), float(max(y))]})
        limits[f'{stem}:{n}'] = {'xlim': [xmin, xmax], 'ylim': [ymin, ymax], 'series': series}
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(OUT / f'{stem}.{ext}', dpi=600)
    # Embed the live-text font files so that editable SVGs render without
    # relying on installation of Computer Modern on another workstation.
    svgpath=OUT/f'{stem}.svg'
    svg=svgpath.read_text(encoding='utf-8')
    # Mathtext's legacy CM encoding places capital Omega at soft-hyphen,
    # which a browser suppresses. Remap that live glyph to Unicode Omega
    # and add the equivalent cmap entry in the EMBEDDED font only.
    # The font outline, geometry and all supplied font files stay unchanged.
    svg=svg.replace('>\u00ad</tspan>','>\u03a9</tspan>')
    used=sorted(set(re.findall(r"font-family:\s*'([^']+)'",svg)))
    font_rules=[]
    for family in used:
        fontpath=Path(font_manager.findfont(font_manager.FontProperties(family=family),fallback_to_default=False))
        fontbytes=fontpath.read_bytes()
        if family=='cmr10':
            font=TTFont(BytesIO(fontbytes))
            for table in font['cmap'].tables:
                if table.isUnicode() and 0xad in table.cmap:
                    table.cmap[0x03a9]=table.cmap[0xad]
            memory=BytesIO();font.save(memory);fontbytes=memory.getvalue()
        encoded=base64.b64encode(fontbytes).decode('ascii')
        font_rules.append("@font-face{font-family:'"+family+"';src:url(data:font/ttf;base64,"+encoded+") format('truetype');font-weight:normal;font-style:normal;}")
    svg=svg.replace('<defs>', '<defs>\n<style type="text/css"><![CDATA['+''.join(font_rules)+']]></style>',1)
    svgpath.write_text(svg,encoding='utf-8')
    # A selected-backend final-size reading proof, without modifying final PNG.
    fig.savefig(QA / f'{stem}.reading.png', dpi=300)
    plt.close(fig)


def geometry():
    m, chi, S, J = 1/8, 1., 3/4, 3
    shifts = m * 2.**np.arange(J)
    fig, axs = plt.subplots(1, 2, figsize=(5.65, 3.35))
    fig.subplots_adjust(left=.09, right=.98, top=.88, bottom=.28, wspace=.29)
    theta = np.linspace(0, 2*np.pi, 800)
    a = axs[0]
    a.plot(chi*np.cos(theta), chi*np.sin(theta), color=GRAY, lw=1.15,
           label=r'$|z|\leq\chi$', gid='schematic-boundary')
    a.add_patch(Circle((chi/2, 0), chi/2, fill=False, ec=ORANGE, ls='--', lw=1.1,
                       label=r'$|z-\chi/2|\leq\chi/2$'))
    xx = np.linspace(m, chi, 500)
    yy = np.minimum(S*xx, np.sqrt(np.maximum(0, chi**2-xx**2)))
    a.fill_between(xx, -yy, yy, color='#CFE5EE', zorder=0)
    a.plot([0, chi], [0, S*chi], color=BLUE, lw=.8, ls=':', gid='schematic-boundary')
    a.plot([0, chi], [0, -S*chi], color=BLUE, lw=.8, ls=':', gid='schematic-boundary')
    a.axvline(m, color=GRAY, ls='-.', lw=.8, label=r'$\mathrm{Re}\,z=m_\tau$')
    a.axhline(0, color='#999999', lw=.5, gid='schematic-boundary')
    a.axvline(0, color='#999999', lw=.5, gid='schematic-boundary')
    a.text(.61, .09, r'$\Omega_\tau$', fontsize=11)
    a.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1),
          xlabel=r'$\mathrm{Re}\,z$', ylabel=r'$\mathrm{Im}\,z$')
    a.set_aspect('equal', adjustable='box')
    a.set_xticks([-1, 0, 1]); a.set_yticks([-1, 0, 1])
    row_title(a, '(a) Cayley enclosure')
    b = axs[1]
    # Radial shells intersect the SAME sector/disk/vertical enclosure.
    # Polar polygon rendering preserves the exact radius boundaries.
    angle = np.linspace(-math.atan(S), math.atan(S), 401)
    fills = ['#DFEDF2', '#BFDCE7', '#91C1D4']
    for j, (s, fill) in enumerate(zip(shifts, fills)):
        rlo = np.maximum(s, m / np.cos(angle))
        rhi = np.full_like(angle, min(2*s, chi))
        ok = rhi >= rlo
        th = angle[ok]; lo = rlo[ok]; hi = rhi[ok]
        px = np.r_[hi*np.cos(th), (lo*np.cos(th))[::-1]]
        py = np.r_[hi*np.sin(th), (lo*np.sin(th))[::-1]]
        b.fill(px, py, color=fill, ec='none')
    for r in [m, 2*m, 4*m, chi]:
        th = angle[np.cos(angle)*r >= m-1e-12]
        b.plot(r*np.cos(th), r*np.sin(th), color=BLUE, lw=.75, gid='schematic-boundary')
    b.plot([m,m], [-S*m,S*m], color=GRAY, lw=.8, ls='-.', gid='schematic-boundary')
    b.plot([m,chi/np.sqrt(1+S*S)], [S*m,S*chi/np.sqrt(1+S*S)], color=BLUE, lw=.8, gid='schematic-boundary')
    b.plot([m,chi/np.sqrt(1+S*S)], [-S*m,-S*chi/np.sqrt(1+S*S)], color=BLUE, lw=.8, gid='schematic-boundary')
    b.axhline(0, color='#888888', lw=.6, gid='schematic-boundary')
    b.plot(shifts, np.zeros(J), 's', ms=4.2, color=ORANGE, label=r'Cayley shifts $s_j$')
    b.text(.01,.54,r'$s_j=m_\tau 2^j$',fontsize=10.)
    b.set(xlim=(-.15,1.15), ylim=(-.65,.65),
          xlabel=r'$\mathrm{Re}\,z$', ylabel=r'$\mathrm{Im}\,z$')
    b.set_aspect('equal', adjustable='box')
    b.set_xticks([0,.5,1]); b.set_yticks([-.5,0,.5])
    row_title(b, '(b) Dyadic shells')
    handles = [Patch(fc='#CFE5EE', ec='none', label=r'$\Omega_\tau$'),
               Line2D([],[],color=GRAY,lw=1.15,label=r'$|z|\leq\chi$'),
               Line2D([],[],color=ORANGE,lw=1.1,ls='--',label=r'$|z-\chi/2|\leq\chi/2$'),
               Line2D([],[],color=GRAY,lw=.8,ls='-.',label=r'$\mathrm{Re}\,z=m_\tau$')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53,.005), ncol=2,
               columnspacing=2.0, handlelength=2.0, labelspacing=.5)
    save(fig, 'F01_cayley_geometry')
    return {'m_tau':'1/8','chi':'1','S':'3/4','J':J,'shifts':shifts.tolist(),
            'interpretation':'Schematic only; no sample spectrum or numerical range is plotted.'}


def time_scaling(time):
    fig, axs = plt.subplots(1,2,figsize=(5.65,3.45))
    fig.subplots_adjust(left=.115, right=.98, bottom=.27, top=.88, wspace=.41)
    colors = [BLUE, ORANGE, GREEN, PURPLE, GRAY]
    records = []
    for row, color in zip(time['rows'], colors):
        c = row['constants']
        d = np.arange(len(row['F']))
        assert len(d) == len(row['G'])
        obs = positive(np.maximum(row['F'], row['G']))
        bound = positive(c['preBoth'] * c['q']**(d//c['J']))
        axs[0].plot(d, obs, color=color, lw=1.2, label=f"1/{row['denominator']}")
        axs[0].plot(d, bound, color=color, lw=.9, ls='--')
        records.append({'denominator':row['denominator'],'n':len(d),
                        'd':d.tolist(),'observed':obs.tolist(),'sufficient_factor':bound.tolist()})
    axs[0].set(xlim=(-1,86), ylim=(5e-14,8), yscale='log',
               xlabel=r'Temporal degree $d$', ylabel='Endpoint operator factor')
    axs[0].set_yticks([1e-12,1e-8,1e-4,1])
    axs[0].set_xticks([0,20,40,60,80])
    axs[0].yaxis.set_minor_locator(NullLocator())
    # A detail window occupies a data-free interior area: all observed
    # factors here lie below it and the full-size bounds lie above it.
    detail=axs[0].inset_axes([.62,.57,.35,.23])
    for row,color in zip(time['rows'],colors):
        c=row['constants'];d=np.arange(len(row['F']))
        detail.plot(d,c['preBoth']*c['q']**(d//c['J']),color=color,lw=.75,ls='--')
    detail.set(xlim=(-1,86),ylim=(.075,4.5),yscale='log',xlabel=r'$d$')
    detail.set_xticks([0,40,80]);detail.set_yticks([.1,1])
    detail.yaxis.set_major_formatter(ScalarFormatter())
    detail.yaxis.set_minor_locator(NullLocator())
    detail.tick_params(labelsize=7.5,length=2.,width=.6,pad=1.5)
    detail.xaxis.label.set_size(7.5);detail.xaxis.labelpad=1.
    detail.set_title('Bound detail',fontsize=8.,pad=3.,fontweight='normal')
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    label_artists=[detail.title,detail.xaxis.label,*detail.get_xticklabels(),*detail.get_yticklabels()]
    occupied=[detail.get_window_extent().padded(1.)]+[
        artist.get_window_extent(renderer).padded(1.) for artist in label_artists if artist.get_text()]
    clearances=[all(not line.get_path().transformed(line.get_transform()).intersects_bbox(box,filled=False)
                    for box in occupied) for line in axs[0].lines]
    (QA/'F02_inset_data_clearance.json').write_text(json.dumps({
        'all_parent_curves_clear':all(clearances),'curves_checked':len(clearances),
        'clearances':clearances,
        'hits':[[i,j] for i,line in enumerate(axs[0].lines) for j,box in enumerate(occupied)
                if line.get_path().transformed(line.get_transform()).intersects_bbox(box,filled=False)],
        'occupied_boxes_plus_one_pixel':[b.extents.tolist() for b in occupied],
        'check':'Every main data path tested separately against the inset frame, title, axis labels and tick-label boxes.'
    },indent=2),encoding='utf-8')
    assert all(clearances), 'The detail inset must not cover any main data curve.'
    fig._audit_exemptions=[{'id':'bound-detail','checks':['main-panel-row-comparison'],
                           'reason':'Sufficient-factor detail inset has its own d and factor scales; it is not an equal-size evidence panel.'}]
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,title=r'Time step $\tau$',loc='lower center',
               bbox_to_anchor=(.54,.005),ncol=5,handlelength=1.5,
               columnspacing=1.25,handletextpad=.45,labelspacing=.4)
    row_title(axs[0], '(a) Operator factors')
    x = np.log2([r['denominator'] for r in time['rows']])
    for key, color, ls, marker, label in [
        ('proved_both',BLUE,'-','o','Sufficient (Theorem)'),
        ('observed_both',ORANGE,'-','^','Observed'),
        ('necessary_degree_real',GREEN,':','d','Necessary (class)')]:
        y = positive([r['degrees'][0][key] for r in time['rows']])
        axs[1].plot(x,y,color=color,lw=1.2,ls=ls,marker=marker,ms=4.,label=label)
    axs[1].set(yscale='log', ylim=(2,200), xlim=(3.8,12.8),
               xlabel=r'$\log_2(1/\tau)$', ylabel=r'Degree at accuracy $10^{-2}$')
    axs[1].set_xticks([4,6,8,10,12])
    axs[1].set_yticks([3,10,30,100]); axs[1].yaxis.set_major_formatter(ScalarFormatter())
    axs[1].yaxis.set_minor_locator(NullLocator())
    axs[1].legend(loc='center left', bbox_to_anchor=(.0,.58),handlelength=1.6,
                  labelspacing=.45, borderaxespad=.4)
    row_title(axs[1], '(b) Time-step complexity')
    save(fig,'F02_time_scaling',bound_axes=(0,1,2))
    return records


def alpha_degrees(alpha):
    rows = alpha['rows']
    # Increasing inverse-alpha order makes the scaling guide transparent.
    x = positive([1/float(r['alpha']) for r in rows])
    actual = positive([r['actual_degree'] for r in rows])
    sufficient = positive([r['constants']['sufficient_degree'] for r in rows])
    # The exponent-1/2 guide is normalized at alpha=10^-2 (actual degree=21).
    # It is NOT the theorem's asymptotic class lower bound.
    guide = actual[0] * np.sqrt(x / x[0])
    assert np.all(np.diff(x) > 0)
    assert all(p['integer_degrees_identical'] for p in alpha['precision_comparisons'])
    assert all(float(r['norm_F_at_actual']) <= .01 and float(r['norm_G_at_actual']) <= .01 for r in rows)
    fig, axs = plt.subplots(1,2,figsize=SIZE)
    fig.subplots_adjust(left=.105,right=.98,bottom=.18,top=.87,wspace=.38)
    for ax in axs:
        ax.set(xscale='log',yscale='log',xlim=(50,2e8),
               xlabel=r'Inverse control floor $1/\alpha_-$', ylabel=r'Degree $d$')
        ax.set_xticks([1e2,1e4,1e6,1e8])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
    axs[0].plot(x,sufficient,color=BLUE,lw=1.2,marker='o',ms=4.,label='Sufficient (Theorem)')
    axs[0].plot(x,actual,color=ORANGE,lw=1.2,marker='s',ms=4.,label='First successful block')
    axs[0].plot(x,guide,color=GRAY,lw=.8,ls='--',label='Slope 1/2 guide')
    axs[0].set_ylim(8,2e12)
    axs[0].set_yticks([1e1,1e4,1e7,1e10,1e12])
    axs[0].legend(loc='upper left',handlelength=1.5,labelspacing=.35, borderaxespad=.4)
    row_title(axs[0], '(a) Observed and sufficient')
    axs[1].plot(x,actual,color=ORANGE,lw=1.2,marker='s',ms=4.,label='First successful block')
    axs[1].plot(x,guide,color=GRAY,lw=.8,ls='--',label='Slope 1/2 guide')
    axs[1].set_ylim(10,5e4)
    axs[1].set_yticks([1e1,1e2,1e3,1e4])
    axs[1].legend(loc='upper left',handlelength=1.5,labelspacing=.4,borderaxespad=.4)
    row_title(axs[1], '(b) Observed-degree detail')
    save(fig,'F03_alpha_degrees',bound_axes=(0,1))
    c = (math.acos(199/200)-math.acos(9999/10000))/(44*math.pi)
    return [{'alpha':float(r['alpha']),'inverse_alpha':float(xx),
             'first_successful_degree':int(y),'sufficient_degree':int(s),
             'guide_degree':float(g),'J':r['constants']['J']}
            for r,xx,y,s,g in zip(rows,x,actual,sufficient,guide)], {
                'guide':'21 * (alpha_-/0.01)^(-1/2)',
                'guide_anchor':{'alpha':.01,'actual_degree':21},
                'asymptotic_class_c':c,
                'theorem_reference_not_plotted':'c * alpha_-^(-1/2) * log(100)',
                'finite_alpha_caveat':'The validity threshold a0 is existential; the theorem reference is not a certified finite-alpha pointwise lower bound on this fixed family.'}


def prior_continuation(prior):
    wins = [w['n'] for w in prior['windows']]
    fig, axs = plt.subplots(1,2,figsize=(5.65,3.50))
    fig.subplots_adjust(left=.115,right=.98,bottom=.27,top=.88,wspace=.40)
    export = []
    for ax, level, title in zip(axs,[0.,.001],['(a) Noiseless observations',r'(b) Noise budget $10^{-3}A_{\mathrm{cls}}$']):
        sub = [r for r in prior['rows'] if r['noise_fraction']==level]
        assert len(sub) == 48*len(wins)
        for key,color,ls,marker,label in [
            ('rational_bound',ORANGE,'-','o','Rational certificate'),
            ('prefix_bound',BLUE,'-','s','Prefix certificate'),
            ('rational_error',ORANGE,':','o','Rational error'),
            ('prefix_error',BLUE,':','s','Prefix error')]:
            values=[]
            for n in wins:
                cases=[r for r in sub if r['n']==n]
                assert len(cases)==48
                val=max(r[key] for r in cases)
                values.append(val)
                export.append({'noise_fraction':level,'n':n,'quantity':key,'finite_set_max':val,'n_cases':48})
            positive(values)
            ax.plot(wins,values,color=color,lw=1.15,ls=ls,marker=marker,ms=4.,
                    label=label,markerfacecolor=color if 'bound' in key else 'white')
        ax.set(yscale='log',ylim=(1e-7,1),xlim=(280,1040),
               xlabel=r'Observed time coefficients $n$',
               ylabel=r'Error / class amplitude $A_{\mathrm{cls}}$')
        ax.set_xticks([320,640,1000])
        ax.set_yticks([1e-6,1e-4,1e-2,1])
        ax.yaxis.set_minor_locator(NullLocator())
        row_title(ax,title)
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.54,.005),
               ncol=2,handlelength=1.65,columnspacing=2.0,labelspacing=.4)
    save(fig,'S01_prior_continuation',bound_axes=(0,1))
    return export


def csv_export(name, rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__ == '__main__':
    input_checks=[]
    tpath=DATA/'robin'/'time_step_results.json'
    apath=DATA/'boundary_zero'/'results.json'
    ppath=DATA/'robin'/'prior_results.json'
    geom=geometry()
    timedata=time_scaling(load(tpath))
    alphadata, alphainfo=alpha_degrees(load(apath))
    priordata=prior_continuation(load(ppath))
    csv_export('F03_alpha_degrees_data.csv',alphadata)
    csv_export('S01_prior_continuation_data.csv',priordata)
    (OUT/'F02_time_scaling_data.json').write_text(json.dumps(timedata,indent=2),encoding='utf-8')
    # Exact rational counterexample demonstrates why the two disks differ.
    x,y,m,S=Fraction(4,5),Fraction(3,5),Fraction(1,8),Fraction(3,4)
    symbolic={'point':'4/5 + (3/5)i','Re_ge_m':x>=m,'sector':abs(y)<=S*x,
              'origin_disk':x*x+y*y<=1,
              'outside_shifted_disk':(x-Fraction(1,2))**2+y*y>Fraction(1,4),
              'squared_distance_to_shifted_center':str((x-Fraction(1,2))**2+y*y),
              'shell_union':'[1/8,1/4] union [1/4,1/2] union [1/2,1] = [m_tau,chi]',
              'shell_count':3,'J_formula':math.ceil(math.log2(1/(1/8)))==3,
              'shifted_disk_subset_origin_disk':'|z| <= |z-chi/2| + chi/2 <= chi (triangle inequality)'}
    assert all(symbolic[k] for k in ['Re_ge_m','sector','origin_disk','outside_shifted_disk','J_formula'])
    provenance={'backend':'Python/Matplotlib','python':sys.version,'matplotlib':mpl.__version__,
                'print_width_in':5.65,'dpi':600,'font':'Computer Modern Roman regular (cmr10) with CM math',
                'inputs':input_checks,'geometry':geom,'geometry_symbolic_checks':symbolic,
                'alpha':alphainfo,'source_mapping':{
                    'F02':'time_step_results.json: maximum F/G at EVERY stored degree; preBoth*q**floor(d/J); degrees[0] at epsilon=.01',
                    'F03':'boundary_numerics/results.json: rows.actual_degree and rows.constants.sufficient_degree; exact generated table cross-check follows in selfcheck',
                    'S01':'prior_results.json: finite-set maximum of each of four quantities over same 48 cases, at noise_fraction=0 and .001; all five windows'},
                'uncertainty':'None: deterministic computations and finite-set maxima. No invented intervals.',
                'dependencies':{'required':['numpy','matplotlib','fonttools'],
                                'optional_verification':['Pillow','pymupdf']},
                'alignment_check':'Self-contained rendered two-panel bounding-box check at 1.5 pt tolerance.'}
    (ROOT/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    (QA/'plot_extents.json').write_text(json.dumps(limits,indent=2),encoding='utf-8')
    print('Rendered four same-stem PDF/SVG/600-dpi PNG triples. Generated from the supplied relative input snapshot.')
