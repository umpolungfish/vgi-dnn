/* Scalar reverse accumulation, carried in the same static executable as DT.
 * Constants have id -1. Leaves and operations retain local pullbacks.
 */
struct Node {int left,right;double dl,dr,grad;};
#ifndef VGI_TAPE_CAPACITY
#define VGI_TAPE_CAPACITY 1000000
#endif
static Node tape[VGI_TAPE_CAPACITY];
static int used,overflowed;
struct Var {
    double v;int id;
    constexpr Var(double value=0):v(value),id(-1){}
};
static Var node(double value,int left,int right,double dl,double dr){
    Var result(value);
    if(left<0&&right<0)return result;
    if(used>=VGI_TAPE_CAPACITY){overflowed=1;return result;}
    int i=used++;tape[i]={left,right,dl,dr,0};result.id=i;return result;
}
static Var leaf(double value){
    Var result(value);
    if(used>=VGI_TAPE_CAPACITY){overflowed=1;return result;}
    result.id=used;tape[used++]={-1,-1,0,0,0};return result;
}
static Var operator+(Var a,Var b){return node(a.v+b.v,a.id,b.id,1,1);}
static Var operator-(Var a,Var b){return node(a.v-b.v,a.id,b.id,1,-1);}
static Var operator*(Var a,Var b){return node(a.v*b.v,a.id,b.id,b.v,a.v);}
static Var operator/(Var a,Var b){return node(a.v/b.v,a.id,b.id,1/b.v,-a.v/(b.v*b.v));}
static Var operator-(Var a){return node(-a.v,a.id,-1,-1,0);}
static Var &operator+=(Var &a,Var b){a=a+b;return a;}
static Var &operator*=(Var &a,Var b){a=a*b;return a;}
static bool operator<(Var a,Var b){return a.v<b.v;}
static bool operator>(Var a,Var b){return a.v>b.v;}
static Var ex(Var x){double y=raw_ex(x.v);return node(y,x.id,-1,(x.v< -60||x.v>60)?0:y,0);}
static Var ln(Var x){return node(raw_ln(x.v),x.id,-1,1/x.v,0);}
static Var sq(Var x){double y=raw_sq(x.v);return node(y,x.id,-1,.5/y,0);}
static Var sig(Var x){double y=raw_sig(x.v);return node(y,x.id,-1,y*(1-y),0);}
static Var lg(Var x){
    double derivative=(x.v<1e-6||x.v>1-1e-6)?0:1/(x.v*(1-x.v));
    return node(raw_lg(x.v),x.id,-1,derivative,0);
}
static void backward(Var loss){
    if(loss.id<0)return;
    tape[loss.id].grad=1;
    for(int i=used-1;i>=0;i--){
        Node &n=tape[i];double g=n.grad;if(g==0)continue;
        if(n.left>=0)tape[n.left].grad+=g*n.dl;
        if(n.right>=0)tape[n.right].grad+=g*n.dr;
    }
}
