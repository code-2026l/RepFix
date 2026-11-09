"""Layered vector planes for explanatory feature diagrams; never measurements."""
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Rectangle,FancyBboxPatch,FancyArrowPatch,Circle

from figure_style import COLORS as C

FONT='Times New Roman'


def text(ax,x,y,value,size=8,color='#17212A',bold=False,ha='center',va='center',italic=False):
    return ax.text(x,y,value,fontsize=size,fontfamily=FONT,fontweight='bold' if bold else 'normal',
                   fontstyle='italic' if italic else 'normal',color=color,ha=ha,va=va,zorder=10)


def line_arrow(ax,points,color='#27333C',head=6,lw=.65):
    for p,q in zip(points[:-2],points[1:-1]):
        ax.plot([p[0],q[0]],[p[1],q[1]],color=color,lw=lw,zorder=6)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=head,
                                color=color,lw=lw,shrinkA=0,shrinkB=0,zorder=6))


def plane(ax,x,y,w,h,color,depth=4,slant=21,dx=11,dy=8,front_alpha=.055):
    for i in reversed(range(depth)):
        ox,oy=i*dx,i*dy
        p=[(x+ox,y+oy),(x+w+ox,y+oy),(x+w+slant+ox,y+h+oy),(x+slant+ox,y+h+oy)]
        ax.add_patch(Polygon(p,closed=True,facecolor=color,alpha=front_alpha if i==0 else .045,
                             edgecolor='none',zorder=2+i*.01))
        ax.add_patch(Polygon(p,closed=True,fill=False,edgecolor=color,alpha=.70 if i==0 else .22,
                             lw=.65 if i==0 else .55,zorder=4))
    return x+w/2+slant/2,y+h/2


def small_head(ax,x,y,w,h,label,color):
    patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=3',
        facecolor='white',edgecolor=color,lw=.65,zorder=8)
    ax.add_patch(patch)
    label=text(ax,x+w/2,y+h/2,label,8.2,color)
    if not hasattr(ax,'_diagram_boxes'):ax._diagram_boxes=[]
    ax._diagram_boxes.append((patch,label))


def consumer_planes(ax):
    ax.set(xlim=(0,1000),ylim=(0,365));ax.set_axis_off()
    ax.add_patch(Rectangle((13,48),190,257,facecolor='#FFF8E9',edgecolor='none',zorder=0))
    text(ax,108,335,'1. Shared update',8.6,bold=True)
    text(ax,108,292,'$x$',8.5)
    line_arrow(ax,[(108,280),(108,269)])
    small_head(ax,39,226,140,42,'Encoder $f_\\theta$',C['current'])
    text(ax,108,177,'$L_p+\\alpha L_a$',8.5)
    text(ax,108,112,'Cloned model\n+ Adam state',7.5,color='#59636C',italic=True)
    line_arrow(ax,[(180,247),(223,247),(223,88),(269,88)])

    # Feature vectors are schematic coordinates, not measured values or depth.
    plane(ax,260,54,420,68,C['current'],depth=4,slant=22)
    text(ax,468,105,'2. Shared representation',8.5,bold=True)
    for x,label in [(309,'$h_1$'),(425,'$h_2$'),(555,'\\ldots'),(651,'$h_B$')]:
        text(ax,x,75,label if label.startswith('$') else '$'+label+'$',7.9)
    # Soft projection sheets support the forward arrows without adding modules.
    ax.add_patch(Polygon([(304,123),(356,123),(434,260),(376,260)],facecolor=C['frozen'],
                         edgecolor='none',alpha=.045,zorder=1))
    ax.add_patch(Polygon([(440,123),(502,123),(513,185),(451,185)],facecolor=C['repair'],
                         edgecolor='none',alpha=.055,zorder=1))
    line_arrow(ax,[(319,123),(319,280),(371,280)])
    line_arrow(ax,[(469,123),(469,183)])
    line_arrow(ax,[(694,88),(808,88)])

    text(ax,555,344,'3. Retained consumer coordinates',8.7,bold=True)
    plane(ax,360,260,365,40,C['purple'],depth=3,slant=22,front_alpha=.045)
    text(ax,551,280,'$\\mathcal{N}_0(H_\\theta)$: fixed statistics',8.1)
    plane(ax,360,185,365,40,C['repair'],depth=3,slant=22,front_alpha=.045)
    text(ax,551,205,'$\\mathcal{T}(H_\\theta)$: input repair',8.1)
    line_arrow(ax,[(738,280),(808,280)],C['frozen'])
    line_arrow(ax,[(738,205),(808,205)],C['repair'])
    small_head(ax,810,260,135,40,'$W_0,\\ b_0$',C['frozen'])
    small_head(ax,810,185,135,40,'$W_0,\\ b_0$',C['repair'])
    small_head(ax,810,67,135,42,'$W_\\theta,\\ b_\\theta$',C['current'])
    text(ax,878,320,'Frozen',7.8,C['frozen'],italic=True)
    text(ax,878,242,'Repaired',7.8,C['repair'],italic=True)
    text(ax,878,127,'Current',7.8,C['current'],italic=True)
    ax.plot([877,877],[228,257],color='#8F979F',lw=.55,ls=(0,(3,3)),zorder=3)
    for y,label,color in [(280,'$\\ell_0$',C['frozen']),(205,'$\\ell_0^{\\mathcal{T}}$',C['repair']),
                           (88,'$\\ell_\\theta$',C['current'])]:
        line_arrow(ax,[(946,y),(965,y)],color,head=4.5)
        text(ax,983,y,label,7.8,color)
    text(ax,500,18,'Same healthy head; repaired coordinates. Layered planes are schematic.',
         7.2,color='#58616A',italic=True)


def nullspace_planes(ax):
    ax.set(xlim=(0,1000),ylim=(0,440));ax.set_axis_off()
    text(ax,171,405,'1. Reference features',8.7,bold=True)
    text(ax,170,190,'2. Trainable features',8.7,bold=True)
    text(ax,600,405,'3. Preserve the readout',8.7,bold=True)
    ax.add_patch(Rectangle((15,49),187,314,facecolor='#FFF8E9',edgecolor='none',zorder=0))
    text(ax,36,218,'$x$',8.7)
    line_arrow(ax,[(53,218),(76,218),(76,307),(103,307)])
    line_arrow(ax,[(76,217),(76,137),(103,137)])
    small_head(ax,105,284,102,46,'$f_{\\theta_0}$',C['current'])
    small_head(ax,105,114,102,46,'$f_\\theta$',C['purple'])
    text(ax,155,260,'Frozen',7.5,C['current'],italic=True)
    text(ax,155,91,'Learned',7.5,C['purple'],italic=True)
    line_arrow(ax,[(208,307),(259,307)])
    line_arrow(ax,[(208,137),(259,137)])
    plane(ax,254,283,208,47,C['current'],depth=4,slant=20)
    plane(ax,254,113,208,47,C['purple'],depth=4,slant=20)
    text(ax,366,307,'$H_0$',8.8,C['current'])
    text(ax,366,137,'$H_\\theta$',8.8,C['purple'])
    line_arrow(ax,[(477,307),(518,307),(518,151)])
    line_arrow(ax,[(477,307),(496,307),(496,245),(610,245)])
    line_arrow(ax,[(477,137),(504,137)])
    for x,y,label in [(518,137,'−'),(625,245,'+')]:
        ax.add_patch(Circle((x,y),13,facecolor='white',edgecolor='#303841',lw=.65,zorder=8))
        text(ax,x,y,label,9)
    text(ax,521,99,'$\\Delta H$',7.8,C['purple'])
    line_arrow(ax,[(532,137),(559,137)])
    small_head(ax,561,114,101,46,'$P_\\perp$',C['purple'])
    text(ax,612,83,'Readout-nullspace',7.3,C['purple'],italic=True)
    line_arrow(ax,[(663,137),(705,137),(705,245),(640,245)])
    line_arrow(ax,[(625,259),(625,294)])
    plane(ax,562,296,177,50,C['repair'],depth=4,slant=20)
    text(ax,661,322,'$H_{\\rm guard}$',8.5,C['repair'])
    text(ax,610,375,'$H_{\\rm guard}=H_0+\\Delta H P_\\perp$',8.4)
    line_arrow(ax,[(754,322),(791,322),(791,356),(820,356)])
    line_arrow(ax,[(791,322),(791,272),(820,272)])
    small_head(ax,822,335,165,42,'Current head',C['current'])
    small_head(ax,822,251,165,42,"$A,\\ b'_0$",C['repair'])
    text(ax,904,395,'Task utility',7.6,C['current'],italic=True)
    text(ax,905,230,'Same logits',7.6,C['repair'],italic=True)
    ax.plot([801,801],[49,414],ls=(0,(3,3)),lw=.5,color='#B8BEC5',zorder=0)
    ax.add_patch(Rectangle((818,56),174,127,facecolor='#FAF2F6',edgecolor='none',zorder=0))
    text(ax,905,165,'Readout span',7.8,bold=True)
    for i in range(4):ax.plot([846,963],[103+i*13]*2,color='#7D8794',lw=.5,zorder=2)
    for j in range(7):ax.plot([846+j*19.5]*2,[103,142],color='#7D8794',lw=.5,zorder=2)
    text(ax,905,78,'$A P_\\perp=0$',8.4)
    text(ax,906,28,'$\\ell_{\\rm guard}=\\ell_0$',8.3,C['repair'])
    text(ax,390,27,'Two encoder evaluations at inference; feature planes are schematic.',
         7.2,color='#58616A',italic=True)


def validate_layout(fig):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();checked=0
    for ax in fig.axes:
        for patch,label in getattr(ax,'_diagram_boxes',[]):
            p,b=patch.get_window_extent(renderer),label.get_window_extent(renderer)
            assert b.x0>=p.x0+1 and b.x1<=p.x1-1,('Box text too wide',label.get_text())
            assert b.y0>=p.y0+.5 and b.y1<=p.y1-.5,('Box text too tall',label.get_text())
            checked+=1
    return checked
