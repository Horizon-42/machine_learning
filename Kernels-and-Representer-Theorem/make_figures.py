"""
Teaching figures for the rewritten Kernel Theory + Representer Theorem tutorial.
All figures generated programmatically (matplotlib) for accuracy.
"""
import numpy as np, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from numpy.linalg import solve

rng = np.random.default_rng(7)

BLUE="#1f3a68"; BLUE2="#3f6fb5"; ORANGE="#e8833a"; GREEN="#2e8b57"
RED="#b5384d"; GREY="#6b7280"; PURPLE="#6a4c93"
plt.rcParams.update({
    "figure.dpi":120,"savefig.dpi":120,"font.size":13,
    "axes.edgecolor":"#444","axes.linewidth":1.0,"axes.titlesize":14,
    "axes.titleweight":"bold","axes.labelsize":13,"axes.grid":True,
    "grid.color":"#e8e8e8","grid.linewidth":0.9,"legend.frameon":False,
    "font.family":"DejaVu Sans",
})
OUT="/root/kt2/figures/"
def save(fig,name):
    fig.tight_layout(); fig.savefig(OUT+name,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("wrote",name)

def rbf(a,b,s=0.6):
    return np.exp(-(a[:,None]-b[None,:])**2/(2*s*s))

# ============================================================ 1. story arc
def fig_arc():
    fig,ax=plt.subplots(figsize=(11,2.5)); ax.axis("off")
    steps=[(0.5,"1. THE WALL\nlinear models\ncan't bend",RED),
           (2.5,"2. THE TRICK\nkernels = cheap\ndot products",ORANGE),
           (4.5,"3. THE SPACE\nRKHS hiding\ninside a kernel",BLUE),
           (6.5,"4. THE PAYOFF\nrepresenter thm\n= n numbers",GREEN)]
    for x,txt,c in steps:
        ax.add_patch(plt.Rectangle((x-0.75,0.2),1.5,1.2,fill=True,
            facecolor="white",edgecolor=c,lw=2.6))
        ax.text(x,0.8,txt,ha="center",va="center",fontsize=11,color=c,weight="bold")
    for x in [1.35,3.35,5.35]:
        ax.annotate("",xy=(x+0.35,0.8),xytext=(x,0.8),
                    arrowprops=dict(arrowstyle="->",color=GREY,lw=2.2))
    ax.set_xlim(-0.5,7.5); ax.set_ylim(0,1.7)
    save(fig,"story_arc.png")

# ============================================================ 2. the wall + lift
def fig_lift():
    n=110
    r_in=0.38*rng.random(n)**0.5; t=2*np.pi*rng.random(n)
    Xin=np.c_[r_in*np.cos(t),r_in*np.sin(t)]
    r_out=0.62+0.32*rng.random(n); t2=2*np.pi*rng.random(n)
    Xout=np.c_[r_out*np.cos(t2),r_out*np.sin(t2)]
    fig=plt.figure(figsize=(11,4.6))
    ax1=fig.add_subplot(1,2,1)
    ax1.scatter(*Xin.T,s=18,c=BLUE,label="class A")
    ax1.scatter(*Xout.T,s=18,c=ORANGE,label="class B")
    xs=np.linspace(-1.1,1.1,10)
    for b in (-0.1,0.25,0.55):
        ax1.plot(xs,0.5*xs+b,"--",c=RED,lw=1.3,alpha=.7)
    ax1.set_title("A straight line is helpless here")
    ax1.set_xlabel("$x_1$");ax1.set_ylabel("$x_2$");ax1.set_aspect("equal")
    ax1.legend(loc="upper right",fontsize=10);ax1.set_xlim(-1.15,1.15);ax1.set_ylim(-1.15,1.15)
    def phi(X): return np.c_[X[:,0]**2,np.sqrt(2)*X[:,0]*X[:,1],X[:,1]**2]
    Fi,Fo=phi(Xin),phi(Xout)
    ax2=fig.add_subplot(1,2,2,projection="3d")
    ax2.scatter(Fi[:,0],Fi[:,1],Fi[:,2],s=16,c=BLUE)
    ax2.scatter(Fo[:,0],Fo[:,1],Fo[:,2],s=16,c=ORANGE)
    gx,gy=np.meshgrid(np.linspace(0,1,6),np.linspace(-0.8,0.8,6))
    ax2.plot_surface(gx,gy,0.42-gx,alpha=0.25,color=GREEN)
    ax2.set_title("Lift with $\\phi(x)=(x_1^2,\\sqrt{2}x_1x_2,x_2^2)$: a plane works")
    ax2.set_xlabel("$x_1^2$");ax2.set_ylabel("$\\sqrt{2}x_1x_2$");ax2.set_zlabel("$x_2^2$")
    ax2.view_init(elev=18,azim=-62)
    save(fig,"lift_circle.png")

# ============================================================ 3. feature explosion
def fig_feature_explosion():
    ps=np.arange(1,9)
    fig,ax=plt.subplots(figsize=(7.4,4.7))
    for d,c in [(2,GREEN),(10,ORANGE),(100,RED)]:
        feats=[math.comb(d+p,p) for p in ps]
        ax.plot(ps,feats,marker="o",lw=2.4,c=c,label=f"$d={d}$ inputs")
    ax.set_yscale("log")
    ax.set_xlabel("polynomial degree $p$")
    ax.set_ylabel("number of explicit features  (log scale)")
    ax.set_title("Why you can't just write the features down")
    ax.axhline(1e6,color=GREY,ls=":",lw=1)
    ax.text(1.1,1.4e6,"a million features",color=GREY,fontsize=10)
    ax.legend(fontsize=10.5,loc="lower right")
    ax.text(2.2,2e9,"RBF kernel lives in an\n$\\infty$-dimensional space\n— off this chart entirely",
            fontsize=10,color=BLUE,ha="center",va="center",
            bbox=dict(boxstyle="round",fc="#eef2f8",ec=BLUE,lw=1))
    save(fig,"feature_explosion.png")

# ============================================================ 4. kernel = similarity + bumps
def fig_similarity_bumps():
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    d=np.linspace(-3,3,400)
    for s,c in [(0.3,BLUE),(0.7,ORANGE),(1.5,GREEN)]:
        axes[0].plot(d,np.exp(-d**2/(2*s*s)),lw=2.4,c=c,label=f"$\\sigma={s}$")
    axes[0].set_title("A kernel is a similarity score")
    axes[0].set_xlabel("distance $x-x'$");axes[0].set_ylabel("$k(x,x')$")
    axes[0].legend(fontsize=10.5,title="RBF bandwidth")
    centers=np.array([-2.0,-0.7,0.5,1.8]); al=np.array([0.8,-1.1,1.3,0.6])
    x=np.linspace(-3.2,3.2,500); tot=np.zeros_like(x)
    for a,xi in zip(al,centers):
        b=a*np.exp(-(x-xi)**2/(2*0.6**2)); axes[1].plot(x,b,lw=1.3,ls="--",c=GREY,alpha=.9); tot+=b
    axes[1].plot(x,tot,lw=3,c=BLUE,label="$f=\\sum_i\\alpha_i\\,k(\\cdot,x_i)$")
    axes[1].scatter(centers,0*centers,c=ORANGE,s=55,zorder=5,label="data points $x_i$")
    axes[1].axhline(0,color="#ccc",lw=1)
    axes[1].set_title("Every function the model builds is a sum of bumps")
    axes[1].set_xlabel("$x$");axes[1].set_ylabel("$f(x)$");axes[1].legend(fontsize=10)
    save(fig,"similarity_bumps.png")

# ============================================================ 5. eval not continuous in L2
def fig_eval_not_continuous():
    fig,ax=plt.subplots(figsize=(7.6,4.7))
    x=np.linspace(-1.4,1.4,1000)
    for w,c in [(1.0,GREEN),(0.5,ORANGE),(0.22,RED)]:
        f=np.clip(1-np.abs(x)/w,0,None)
        norm2=2*w/3.0
        ax.plot(x,f,lw=2.4,c=c,label=f"width $w={w}$:  $\\|f\\|_2^2={norm2:.2f}$")
    ax.scatter([0],[1],c="k",zorder=6)
    ax.annotate("every spike has $f(0)=1$",xy=(0,1),xytext=(-1.35,0.62),
                fontsize=11,arrowprops=dict(arrowstyle="->",color="k"))
    ax.set_title("In $L^2$, size $\\to 0$ but the value at a point does NOT")
    ax.set_xlabel("$x$");ax.set_ylabel("$f(x)$");ax.set_ylim(-0.05,1.3)
    ax.legend(fontsize=10.5,loc="upper right")
    save(fig,"eval_not_continuous.png")

# ============================================================ 6. norm = complexity
def fig_norm_complexity():
    x=np.linspace(0,2*np.pi,600)
    fig,ax=plt.subplots(figsize=(7.8,4.7))
    for k,c in [(1,GREEN),(2,ORANGE),(5,RED)]:
        ax.plot(x,np.sin(k*x),lw=2.2,c=c,
                label=f"$\\sin({k}x)$:  size $\\|f\\|_2^2=\\pi$,  roughness $={k*k}\\pi$")
    ax.set_title("The RKHS norm charges for wiggles, not for size")
    ax.set_xlabel("$x$");ax.set_ylabel("$f(x)$");ax.set_ylim(-1.5,1.6)
    ax.legend(fontsize=10,loc="upper right")
    save(fig,"norm_complexity.png")

# ============================================================ 7. orthogonal decomposition
def fig_orthogonal():
    fig=plt.figure(figsize=(7.4,6)); ax=fig.add_subplot(111,projection="3d")
    e1=np.array([1.0,0.2,0.0]); e2=np.array([0.15,1.0,0.0])
    gs=np.linspace(-1.2,1.2,8); G1,G2=np.meshgrid(gs,gs)
    P=(G1[...,None]*e1+G2[...,None]*e2)
    ax.plot_surface(P[...,0],P[...,1],P[...,2],alpha=0.18,color=BLUE2)
    u=0.9*e1+0.5*e2; v=np.array([0,0,1.1]); f=u+v
    def arr(vec,c,lab):
        ax.quiver(0,0,0,*vec,color=c,lw=3,arrow_length_ratio=0.12)
        ax.text(vec[0]*1.05,vec[1]*1.05,vec[2]*1.05,lab,color=c,fontsize=15,weight="bold")
    arr(u,GREEN,"$u$ (in data-span)"); arr(v,ORANGE,"$v$ (leftover)"); arr(f,BLUE,"$f=u+v$")
    ax.plot([u[0],f[0]],[u[1],f[1]],[u[2],f[2]],ls=":",c=GREY)
    ax.text(-1.15,-1.0,0.0,"$S=\\mathrm{span}\\{k(\\cdot,x_1),\\dots,k(\\cdot,x_n)\\}$",color=BLUE2,fontsize=10.5)
    ax.set_xlim(-1.2,1.2);ax.set_ylim(-1.2,1.2);ax.set_zlim(-0.2,1.3)
    ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([])
    ax.set_title("The one picture behind the representer theorem")
    ax.view_init(elev=16,azim=-58)
    save(fig,"orthogonal_decomposition.png")

# ============================================================ 8. kernel ridge payoff
def fig_kernel_ridge():
    n=22; xtr=np.sort(rng.uniform(-3,3,n))
    ytr=np.sin(1.3*xtr)+0.35*xtr+0.25*rng.standard_normal(n)
    s=0.7; lam=0.15; K=rbf(xtr,xtr,s)
    alpha=solve(K+n*lam*np.eye(n),ytr)
    xt=np.linspace(-3.4,3.4,400)
    Kt=np.exp(-(xt[:,None]-xtr[None,:])**2/(2*s*s)); yhat=Kt@alpha
    fig,ax=plt.subplots(figsize=(7.6,4.7))
    for a,xi in zip(alpha,xtr):
        ax.plot(xt,a*np.exp(-(xt-xi)**2/(2*s*s)),lw=0.8,c=GREY,alpha=.5)
    ax.scatter(xtr,ytr,c=ORANGE,s=42,zorder=5,label="training data")
    ax.plot(xt,yhat,c=BLUE,lw=3,label="$f^*(x)=\\sum_i\\alpha_i^*\\,k(x,x_i)$")
    ax.axhline(0,color="#ddd",lw=1)
    ax.set_title("Kernel ridge regression: the whole story, computed")
    ax.set_xlabel("$x$");ax.set_ylabel("$y$");ax.legend(fontsize=10.5,loc="upper left")
    save(fig,"kernel_ridge_fit.png")

# ============================================================ 9. lambda effect
def fig_lambda():
    n=20; xtr=np.sort(rng.uniform(-3,3,n)); ytr=np.sin(1.4*xtr)+0.25*rng.standard_normal(n)
    s=0.5; xt=np.linspace(-3.3,3.3,400)
    K=rbf(xtr,xtr,s); Kt=np.exp(-(xt[:,None]-xtr[None,:])**2/(2*s*s))
    fig,axes=plt.subplots(1,3,figsize=(12.5,4.1),sharey=True)
    for ax,lam,t in zip(axes,[1e-4,0.2,8.0],
        ["$\\lambda$ tiny: memorises noise","$\\lambda$ balanced: learns the shape","$\\lambda$ huge: forgets everything"]):
        a=solve(K+n*lam*np.eye(n),ytr)
        ax.plot(xt,Kt@a,c=BLUE,lw=2.6); ax.scatter(xtr,ytr,c=ORANGE,s=32,zorder=5)
        ax.set_title(t,fontsize=12);ax.set_xlabel("$x$");ax.set_ylim(-2,2)
    axes[0].set_ylabel("$y$")
    save(fig,"lambda_effect.png")

if __name__=="__main__":
    fig_arc(); fig_lift(); fig_feature_explosion(); fig_similarity_bumps()
    fig_eval_not_continuous(); fig_norm_complexity(); fig_orthogonal()
    fig_kernel_ridge(); fig_lambda()
    print("ALL DONE")
