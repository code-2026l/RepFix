// Fused CPU reference/current evaluation; all weights are prepacked float32.
// C++ compilation is numerical runtime setup, not manuscript compilation.
#include <algorithm>
#include <cstddef>
#ifndef DIM
#define DIM 256
#endif
#ifndef CLASSES
#define CLASSES 11
#endif
#ifndef FIRST_RANK
#define FIRST_RANK 16
#endif
#ifndef SECOND_RANK
#define SECOND_RANK 12
#endif
#ifndef SPARSE
#define SPARSE 0
#endif
constexpr int M=64,D=DIM,C=CLASSES,S=FIRST_RANK,R=SECOND_RANK;

static inline void mv(const float* x,const float* w,const float* b,float* y,int k,int n){
    for(int j=0;j<n;++j)y[j]=b?b[j]:0.f;
    for(int i=0;i<k;++i){const float z=x[i];
        #pragma GCC ivdep
        for(int j=0;j<n;++j)y[j]+=z*w[i*n+j];
    }
}
static inline void relu(float* x,int n){for(int i=0;i<n;++i)x[i]=std::max(x[i],0.f);}
static inline void current_one(const float*x,const float*const*p,float*y){
    alignas(64) float h1[M],h2[M];mv(x,p[0],p[1],h1,D,M);relu(h1,M);
    mv(h1,p[2],p[3],h2,M,M);relu(h2,M);mv(h2,p[4],p[5],y,M,C);
}
static inline void healthy_one(const float*x,const float*const*p,float*y){
    alignas(64) float h1[M],h2[M];mv(x,p[6],p[7],h1,D,M);relu(h1,M);
    mv(h1,p[8],p[9],h2,M,M);relu(h2,M);mv(h2,p[10],p[11],y,M,C);
}
extern "C" void rf_current(const float*x,const float*const*p,float*y,int batch){
    for(int n=0;n<batch;++n)current_one(x+n*D,p,y+n*C);
}
extern "C" void rf_independent(const float*x,const float*const*p,float*y,int batch){
    for(int n=0;n<batch;++n){current_one(x+n*D,p,y+n*2*C);healthy_one(x+n*D,p,y+n*2*C+C);}
}
extern "C" void rf_empty(const float*x,const float*const*,float*y,int){y[0]=x[0];}
extern "C" void rf_replay(const float*x,const float*const*p,float*y,int batch){
    for(int n=0;n<batch;++n){
        alignas(64) float pre[M+S],old1[M],cur1[M],old2[M],cur2[M],small[R],change[M];
        mv(x+n*D,p[12],p[13],pre,D,M+S);
        for(int i=0;i<M;++i)old1[i]=cur1[i]=std::max(pre[i],0.f);
        #if SPARSE
        for(int j=0;j<S;++j)cur1[static_cast<int>(p[19][j])]=std::max(pre[M+j],0.f);
        mv(old1,p[8],p[9],old2,M,M);
        for(int j=0;j<M;++j)cur2[j]=old2[j]+p[18][j];
        for(int i=0;i<S;++i){int index=static_cast<int>(p[19][i]);float delta=cur1[index]-old1[index];
            #pragma GCC ivdep
            for(int j=0;j<M;++j)cur2[j]+=delta*p[8][index*M+j];
        }
        #else
        mv(pre+M,p[14],p[15],change,S,M);
        for(int j=0;j<M;++j)cur1[j]=std::max(pre[j]+change[j],0.f);
        for(int j=0;j<M;++j){old2[j]=p[9][j];cur2[j]=p[9][j]+p[18][j];}
        // Reuse each healthy second-map weight for both activation versions.
        for(int i=0;i<M;++i){float old=old1[i],cur=cur1[i];
            #pragma GCC ivdep
            for(int j=0;j<M;++j){float weight=p[8][i*M+j];old2[j]+=old*weight;cur2[j]+=cur*weight;}
        }
        #endif
        mv(cur1,p[16],nullptr,small,M,R);mv(small,p[17],nullptr,change,R,M);
        for(int j=0;j<M;++j)cur2[j]=std::max(cur2[j]+change[j],0.f);
        relu(old2,M);mv(cur2,p[4],p[5],y+n*2*C,M,C);mv(old2,p[10],p[11],y+n*2*C+C,M,C);
    }
}
extern "C" void rf_repeat(const float*x,const float*const*p,float*y,int batch,int repetitions,int mode){
    for(int i=0;i<repetitions;++i){
        if(mode==0)rf_current(x,p,y,batch);else if(mode==1)rf_replay(x,p,y,batch);else if(mode==2)rf_independent(x,p,y,batch);else rf_empty(x,p,y,batch);
        asm volatile("" ::: "memory");
    }
}
