/* Parameter-bound DT execution. Generated weights.h supplies the trained state.
 * No external runtime: Vox lifts this complete static image into IMASM.
 */
#include "weights.h"

static double ex(double x) {
    if(x>60)x=60; if(x< -60)x= -60;
    int n=(int)(x/0.6931471805599453094);
    double r=x-n*0.6931471805599453094, term=1, sum=1;
    for(int k=1;k<=24;k++){term*=r/k;sum+=term;}
    while(n>0){sum*=2;n--;} while(n<0){sum*=.5;n++;}
    return sum;
}
static double ln(double x) {
    union {double d; unsigned long long u;} v={.d=x};
    int e=(int)((v.u>>52)&2047)-1023;
    v.u=(v.u&0x000fffffffffffffULL)|(1023ULL<<52);
    double z=(v.d-1)/(v.d+1),term=z,sum=z;
    for(int k=3;k<=61;k+=2){term*=z*z;sum+=term/k;}
    return 2*sum+e*0.6931471805599453094;
}
static double sq(double x) {
    if(x==0)return 0;
    union {double d; unsigned long long u;} v={.d=x};
    v.u=(v.u>>1)+0x1ff8000000000000ULL;
    double a=v.d;for(int k=0;k<8;k++)a=.5*(a+x/a);return a;
}
static double sig(double x){return 1/(1+ex(-x));}
static double lg(double x){if(x<1e-6)x=1e-6;if(x>1-1e-6)x=1-1e-6;return ln(x/(1-x));}

typedef struct {double t[TOKENS][D],f[TOKENS][D];} Evidence;
static double out_t[TOKENS][V],out_f[TOKENS][V];
static double feature_t[TOKENS][D],feature_f[TOKENS][D];
static int settlement_kind;
static unsigned long state_revision,output_revision;
static int output_mode=-1;

static void linear(const double *a,const double *b,const double *bias,
                   const double *t,const double *f,int ni,int no,double *ot,double *of){
    for(int i=0;i<no;i++){
        double x=bias[i],y=bias[i];
        for(int j=0;j<ni;j++){x+=a[i*ni+j]*t[j]+b[i*ni+j]*f[j];
                              y+=a[i*ni+j]*f[j]+b[i*ni+j]*t[j];}
        ot[i]=x;of[i]=y;
    }
}
static void norm(Evidence *x,const double *gamma,Evidence *y){
    for(int i=0;i<TOKENS;i++){
        double sum=0;
        for(int c=0;c<D;c++){double a=lg(x->t[i][c]),b=lg(x->f[i][c]);sum+=a*a+b*b;}
        double inv=1/sq(sum/(2*D)+1e-8);
        for(int c=0;c<D;c++){y->t[i][c]=sig(gamma[c]*lg(x->t[i][c])*inv);
                            y->f[i][c]=sig(gamma[c]*lg(x->f[i][c])*inv);}
    }
}
static void residual(Evidence *x,Evidence *y,const double *gate){
    for(int i=0;i<TOKENS;i++)for(int c=0;c<D;c++){
        double g=sig(gate[c]);
        if(BLEND){x->t[i][c]=(1-g)*x->t[i][c]+g*y->t[i][c];
                  x->f[i][c]=(1-g)*x->f[i][c]+g*y->f[i][c];}
        else {double a=2*y->t[i][c]-1,b=2*y->f[i][c]-1;
              if(a<0)a=0;if(b<0)b=0;
              x->t[i][c]=1-(1-x->t[i][c])*(1-g*a);
              x->f[i][c]=1-(1-x->f[i][c])*(1-g*b);}
    }
}
static void block(Evidence *x,int layer){
    const BlockWeights *w=&blocks[layer];
    Evidence n,y;
    double qt[TOKENS][D],qf[TOKENS][D],kt[TOKENS][D],kf[TOKENS][D];
    double vt[TOKENS][D],vf[TOKENS][D];
    norm(x,w->norm1,&n);
    for(int i=0;i<TOKENS;i++){
        double t[D],f[D];for(int c=0;c<D;c++){t[c]=lg(n.t[i][c]);f[c]=lg(n.f[i][c]);}
        linear(w->qA,w->qB,w->qb,t,f,D,D,qt[i],qf[i]);
        linear(w->kA,w->kB,w->kb,t,f,D,D,kt[i],kf[i]);
        linear(w->vA,w->vB,w->vb,t,f,D,D,vt[i],vf[i]);
    }
    for(int i=0;i<TOKENS;i++){
        double at[D],af[D],ot[D],of[D];
        for(int h=0;h<H;h++){
            double scores[TOKENS],mass=0,max=-1e300;
            int end=CAUSAL?i+1:TOKENS;
            for(int j=0;j<end;j++){
                double a=0,b=0;
                for(int c=h*(D/H);c<(h+1)*(D/H);c++){
                    a+=qt[i][c]*kt[j][c]+qf[i][c]*kf[j][c];
                    b+=qt[i][c]*kf[j][c]+qf[i][c]*kt[j][c];}
                scores[j]=(a+w->lam[h]*b)/sq(D/H);if(scores[j]>max)max=scores[j];
            }
            for(int j=0;j<end;j++){scores[j]=ex(scores[j]-max);mass+=scores[j];}
            for(int c=h*(D/H);c<(h+1)*(D/H);c++){
                double t=0,f=0;
                for(int j=0;j<end;j++){
                    double tv=sig((h%2==0)?-vt[j][c]:vt[j][c]);
                    double fv=sig((h%2==0)?-vf[j][c]:vf[j][c]);
                    if(tv<1e-6)tv=1e-6;if(fv<1e-6)fv=1e-6;
                    t+=scores[j]/mass*ln(tv);f+=scores[j]/mass*ln(fv);
                }
                at[c]=lg((h%2==0)?1-ex(2*t):ex(t));
                af[c]=lg((h%2==0)?1-ex(2*f):ex(f));
            }
        }
        linear(w->oA,w->oB,w->ob,at,af,D,D,ot,of);
        for(int c=0;c<D;c++){y.t[i][c]=sig(ot[c]);y.f[i][c]=sig(of[c]);}
    }
    residual(x,&y,w->gate1);norm(x,w->norm2,&n);
    for(int i=0;i<TOKENS;i++){
        double t[D],f[D],ut[FF],uf[FF],dt[D],df[D];
        for(int c=0;c<D;c++){t[c]=lg(n.t[i][c]);f[c]=lg(n.f[i][c]);}
        linear(w->upA,w->upB,w->upb,t,f,D,FF,ut,uf);
        for(int c=0;c<FF;c++){ut[c]*=sig(1.702*ut[c]);uf[c]*=sig(1.702*uf[c]);}
        linear(w->downA,w->downB,w->downb,ut,uf,FF,D,dt,df);
        for(int c=0;c<D;c++){y.t[i][c]=sig(dt[c]);y.f[i][c]=sig(df[c]);}
    }
    residual(x,&y,w->gate2);
}
static void settle(Evidence *x){
    unsigned char history[32][TOKENS][D],state[TOKENS][D];
    for(int i=0;i<TOKENS;i++)for(int c=0;c<D;c++)state[i][c]=(x->t[i][c]>.5)|((x->f[i][c]>.5)<<1);
    settlement_kind=0;
    for(int step=0;step<32;step++){
        for(int prev=0;prev<step;prev++){
            int equal=1;for(int i=0;i<TOKENS;i++)for(int c=0;c<D;c++)if(history[prev][i][c]!=state[i][c])equal=0;
            if(equal){
                settlement_kind=(step-prev==1)?1:2;
                for(int i=0;i<TOKENS;i++)for(int c=0;c<D;c++){
                    int joined=0;for(int k=prev;k<step;k++)joined|=history[k][i][c];
                    x->t[i][c]=(joined&1)?.95:.05;x->f[i][c]=(joined&2)?.95:.05;
                }return;
            }
        }
        Evidence y;
        for(int i=0;i<TOKENS;i++)for(int c=0;c<D;c++){
            history[step][i][c]=state[i][c];
            y.t[i][c]=(state[i][c]&1)?.95:.05;y.f[i][c]=(state[i][c]&2)?.95:.05;
        }
        block(&y,LAYERS-1);
        for(int i=0;i<TOKENS;i++)for(int c=0;c<D;c++)state[i][c]=(y.t[i][c]>.5)|((y.f[i][c]>.5)<<1);
    }
}
static void forward(int reentry){
    Evidence x,n;
    for(int i=0;i<TOKENS;i++)for(int c=0;c<D;c++){
        x.t[i][c]=sig(embedding_t[tokens[i]*D+c]+.5*position[i*D+c]);
        x.f[i][c]=sig(embedding_f[tokens[i]*D+c]+.5*position[((i+7)%MAXLEN)*D+c]);
    }
    for(int l=0;l<LAYERS;l++)block(&x,l);
    if(reentry)settle(&x);
    norm(&x,final_norm,&n);
    for(int i=0;i<TOKENS;i++){
        for(int c=0;c<D;c++){feature_t[i][c]=lg(n.t[i][c]);feature_f[i][c]=lg(n.f[i][c]);}
        linear(headA,headB,headb,feature_t[i],feature_f[i],D,V,out_t[i],out_f[i]);
    }
    output_revision=state_revision;output_mode=reentry;
}
/* Individual fixed-point readout, for scalar VM entry arguments. */
long vgi_component(long component,long reentry){
    if(component<0||component>=2*TOKENS*V)return -999999999;
    forward((int)reentry);
    int channel=(int)(component/(TOKENS*V)),index=(int)(component%(TOKENS*V));
    double value=channel?out_f[index/V][index%V]:out_t[index/V][index%V];
    return (long)(value*1000000);
}
/* Digest covers both evidence channels at every token and vocabulary position. */
long vgi_eval(long reentry){
    forward((int)reentry);unsigned long hash=2166136261;
    for(int channel=0;channel<2;channel++)for(int i=0;i<TOKENS;i++)for(int c=0;c<V;c++){
        long v=(long)((channel?out_f[i][c]:out_t[i][c])*1000000);
        hash=((hash^(unsigned long)v)*16777619)&0x7fffffff;
    }return (long)hash;
}
/* Shared-library diagnostic retains unquantized values for full parity checks. */
double vgi_value(long component,long reentry){
    if(output_revision!=state_revision||output_mode!=(int)reentry)forward((int)reentry);
    int channel=(int)(component/(TOKENS*V)),index=(int)(component%(TOKENS*V));
    return channel?out_f[index/V][index%V]:out_t[index/V][index%V];
}
long vgi_settlement(void){forward(1);return settlement_kind;}

/* Eight token IDs, nine bits each, supplied through scalar call arguments. */
static int input_words(unsigned long first,unsigned long second){
    if(TOKENS!=8)return 0;
    for(int i=0;i<8;i++){
        int token=(int)(((i<4?first:second)>>(9*(i%4)))&511);
        if(token>=V)return 0;
    }
    for(int i=0;i<8;i++)tokens[i]=(int)(((i<4?first:second)>>(9*(i%4)))&511);
    state_revision++;
    return 1;
}
long vgi_input_eval(unsigned long first,unsigned long second,long reentry){
    if(!input_words(first,second))return -999999999;
    return vgi_eval(reentry);
}

static double original_A[V*D],original_B[V*D],original_b[V];
static int saved_head;
static void reset_head(void){
    if(!saved_head){
        for(int i=0;i<V*D;i++){original_A[i]=headA[i];original_B[i]=headB[i];}
        for(int i=0;i<V;i++)original_b[i]=headb[i];saved_head=1;
    }else{
        for(int i=0;i<V*D;i++){headA[i]=original_A[i];headB[i]=original_B[i];}
        for(int i=0;i<V;i++)headb[i]=original_b[i];
    }
}
static void refresh_head(void){
    /* Only these two rows enter the next gradient. A full forward refreshes
     * every output when an output digest is requested. */
    for(int row=259;row<=260;row++){
        double t=headb[row],f=headb[row];
        for(int c=0;c<D;c++){
            double a=feature_t[TOKENS-1][c],b=feature_f[TOKENS-1][c];
            t+=headA[row*D+c]*a+headB[row*D+c]*b;
            f+=headA[row*D+c]*b+headB[row*D+c]*a;
        }
        out_t[TOKENS-1][row]=t;out_f[TOKENS-1][row]=f;
    }
}
static double carrier_loss(long bits){
    double loss=0;
    for(int k=0;k<4;k++){
        int row=259+k/2;
        double p=sig((k%2)?out_f[TOKENS-1][row]:out_t[TOKENS-1][row]);
        if(p<1e-12)p=1e-12;if(p>1-1e-12)p=1-1e-12;
        loss-=((bits>>k)&1)?ln(p):ln(1-p);
    }return loss/4;
}
static long head_digest(void){
    unsigned long hash=2166136261;
    for(int i=0;i<V*D;i++){
        hash=((hash^(unsigned long)(long)(headA[i]*1000000))*16777619)&0x7fffffff;
        hash=((hash^(unsigned long)(long)(headB[i]*1000000))*16777619)&0x7fffffff;
    }
    for(int i=0;i<V;i++)hash=((hash^(unsigned long)(long)(headb[i]*1000000))*16777619)&0x7fffffff;
    return (long)hash;
}
/* Analytic SGD on the shared paired-evidence head. The trunk is retained.
 * Targets are four independent support bits. Contradiction carries no penalty.
 * Readouts: 0 loss before, 1 loss after, 2 parameter digest, 3 output digest,
 * 4 resulting four-bit prediction, 5 pre-update parameter digest.
 */
long vgi_learn(unsigned long first,unsigned long second,long bits,long steps,long readout){
    if(V<=260||bits<0||bits>15||steps<0||steps>10000||!input_words(first,second))return -999999999;
    reset_head();forward(0);
    double before=carrier_loss(bits);long initial_digest=head_digest();
    for(long step=0;step<steps;step++){
        for(int layer=0;layer<2;layer++){
            int row=259+layer;
            double dt=(sig(out_t[TOKENS-1][row])-((bits>>(2*layer))&1))/4;
            double df=(sig(out_f[TOKENS-1][row])-((bits>>(2*layer+1))&1))/4;
            for(int c=0;c<D;c++){
                double t=feature_t[TOKENS-1][c],f=feature_f[TOKENS-1][c];
                headA[row*D+c]-=.05*(dt*t+df*f);
                headB[row*D+c]-=.05*(dt*f+df*t);
            }
            headb[row]-=.05*(dt+df);
        }
        refresh_head();
    }
    state_revision++;
    if(readout==0)return (long)(before*1000000);
    if(readout==1)return (long)(carrier_loss(bits)*1000000);
    if(readout==2)return head_digest();
    if(readout==3)return vgi_eval(0);
    if(readout==4){
        long result=0;
        for(int k=0;k<4;k++)if(((k%2)?out_f[TOKENS-1][259+k/2]:out_t[TOKENS-1][259+k/2])>0)result|=1L<<k;
        return result;
    }
    if(readout==5)return initial_digest;
    return -999999999;
}
double vgi_parameter(long index){
    if(index<0||index>=2*V*D+V)return 0;
    if(index<V*D)return headA[index];
    if(index<2*V*D)return headB[index-V*D];
    return headb[index-2*V*D];
}
static long checkpoint_write(const void *buffer,unsigned long length){
    unsigned long total=0;
    while(total<length){
        long result;
        const unsigned char *position=(const unsigned char*)buffer+total;
        unsigned long remaining=length-total;
        __asm__ volatile("syscall" : "=a"(result) : "a"(1L),"D"(1L),"S"(position),"d"(remaining) : "rcx","r11","memory");
        if(result<=0)return -1;
        total+=(unsigned long)result;
    }return (long)total;
}
/* The full checkpoint returns through the executing module's stdout boundary. */
long vgi_learn_export(unsigned long first,unsigned long second,long bits,long steps){
    if(vgi_learn(first,second,bits,steps,2)==-999999999)return -1;
    return emit_checkpoint();
}

/* Learned choice over direct, two-arm, three-arm, and released presentations.
 * Targets/criteria are input tokens. No reference answer filters the choice.
 */
static long policy_trace[13],policy_history[2][13];
static double policy_logits[2][2][4];
static int policy_input(void){
    if(V<299||TOKENS!=8||tokens[0]!=256||tokens[7]!=257)return 0;
    if(tokens[1]<266||tokens[1]>281||tokens[2]<282||tokens[2]>289)return 0;
    if(tokens[3]<266||tokens[3]>281)return 0;
    if(tokens[4]!=298&&(tokens[4]<266||tokens[4]>281))return 0;
    return tokens[5]>=290&&tokens[5]<=293&&tokens[6]>=294&&tokens[6]<=297;
}
static long choose_policy(void){
    if(!policy_input())return -2;
    forward(0);long choice=-1;double best=0;
    for(int action=0;action<4;action++)if(out_t[TOKENS-1][261+action]>best){
        best=out_t[TOKENS-1][261+action];choice=action;
    }return choice;
}
static int policy_gate(int work,int state){
    if(work==1)return ((state&1)<<1)|((state&2)>>1)|((state&4)<<1)|((state&8)>>1);
    if(work==3)return state&5;
    if(work==5)return state&10;
    if(work==6)return state|12;
    return state;
}
static long policy_evidence(void){
    long packed=0;
    for(int a=0;a<4;a++){
        long shape=(out_t[TOKENS-1][261+a]>0)|((out_f[TOKENS-1][261+a]>0)<<1);
        packed|=shape<<(2*a);
    }return packed;
}
static long perform_policy(long choice){
    int seed=tokens[1]-266,work=tokens[2]-282,arity=choice==1?2:choice==2?3:0;
    for(int i=0;i<13;i++)policy_trace[i]=0;
    policy_trace[0]=seed;policy_trace[1]=choice;policy_trace[2]=arity;
    policy_trace[10]=tokens[5]-290;policy_trace[11]=arity;policy_trace[12]=choice<0;
    int result=seed;
    if(choice==0)result=policy_gate(work,seed);
    else if(arity){
        int arms[3];
        if(arity==2){arms[0]=seed&5;arms[1]=seed&10;arms[2]=0;}
        else{arms[0]=seed&1;arms[1]=seed&2;arms[2]=seed&12;}
        for(int a=0;a<3;a++)policy_trace[3+a]=arms[a];
        arms[0]=policy_gate(work,arms[0]);result=0;
        for(int a=0;a<arity;a++)result|=arms[a];
        for(int a=0;a<3;a++)policy_trace[6+a]=arms[a];
    }
    policy_trace[9]=result;return result;
}
long vgi_policy_act(unsigned long first,unsigned long second,long readout){
    if(!input_words(first,second))return -999999999;
    long choice=choose_policy();if(choice== -2)return -999999999;
    long result=perform_policy(choice);
    if(readout==0)return choice;
    if(readout==1)return result;
    if(readout==2)return policy_evidence();
    if(readout==3)return choice>=0&&choice!=tokens[5]-290;
    if(readout==4)return choice==3&&(tokens[5]==291||tokens[5]==292);
    if(readout==5){
        long revised=choice>=0&&choice!=tokens[5]-290;
        long released=choice==3&&(tokens[5]==291||tokens[5]==292);
        return (choice+1)|(result<<3)|(policy_evidence()<<7)|(revised<<15)|(released<<16);
    }
    if(readout>=10&&readout<23)return policy_trace[readout-10];
    return -999999999;
}
/* The second query re-enters the first returned carrier and selected view.
 * Both traces and independent support/refutation observations remain resident.
 */
long vgi_policy_pair(unsigned long first,unsigned long second,
                     unsigned long next_first,unsigned long next_second,long readout){
    if(!input_words(first,second))return -999999999;
    long a=choose_policy();if(a== -2)return -999999999;
    long value=perform_policy(a),view=a>=0?a:tokens[5]-290,first_evidence=policy_evidence();
    for(int i=0;i<13;i++)policy_history[0][i]=policy_trace[i];
    for(int i=0;i<4;i++){policy_logits[0][0][i]=out_t[TOKENS-1][261+i];policy_logits[0][1][i]=out_f[TOKENS-1][261+i];}
    if(!input_words(next_first,next_second))return -999999999;
    tokens[1]=266+(int)value;tokens[5]=290+(int)view;state_revision++;
    long b=choose_policy();if(b== -2)return -999999999;
    long final=perform_policy(b);
    for(int i=0;i<13;i++)policy_history[1][i]=policy_trace[i];
    for(int i=0;i<4;i++){policy_logits[1][0][i]=out_t[TOKENS-1][261+i];policy_logits[1][1][i]=out_f[TOKENS-1][261+i];}
    if(readout==0)return b;
    if(readout==1)return final;
    if(readout==2)return a;
    if(readout==3)return value;
    if(readout==4)return first_evidence|(policy_evidence()<<8);
    if(readout>=10&&readout<36)return policy_history[(readout-10)/13][(readout-10)%13];
    if(readout>=100&&readout<116){
        int k=(int)readout-100;return (long)(policy_logits[k/8][(k%8)/4][k%4]*1000000);
    }
    return -999999999;
}
