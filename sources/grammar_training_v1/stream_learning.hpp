/* Successive source-bound frames train the same resident parameter state.
 * Wire format: VGIS1\0\0\0, u32 count, then count fixed-size frames.
 * Frame: u16 kind, unroll, steps, bits; u16 tokens[T]; i16 labels[T].
 * kind 0 is support-token CE, 1 candidate evidence, 2 carrier evidence.
 */
static int stream_labels[TOKENS];
static double stream_gradient_norm;
static long stream_completed;
static long stream_steps_completed;
extern "C" void vgi_stream_reset(){reset_parameters();stream_completed=0;}
extern "C" long vgi_stream_token(long index,long token,long label){
    if(index<0||index>=TOKENS||token<0||token>=V||!(label== -100||(label>=0&&label<V)))return -1;
    tokens[index]=(int)token;stream_labels[index]=(int)label;return 0;
}
static Var stream_loss(int kind,long bits){
    Var loss=0;
    if(kind==0){
        int count=0;
        for(int i=0;i<TOKENS;i++)if(stream_labels[i]>=0){
            Var maximum=out_t[i][0];
            for(int c=1;c<V;c++)if(out_t[i][c]>maximum)maximum=out_t[i][c];
            Var mass=0;for(int c=0;c<V;c++)mass+=ex(out_t[i][c]-maximum);
            loss+=ln(mass)+maximum-out_t[i][stream_labels[i]];count++;
        }
        return loss/count;
    }
    int width=kind==1?8:4,base=kind==1?261:259;
    for(int k=0;k<width;k++){
        Var z=(k%2)?out_f[TOKENS-1][base+k/2]:out_t[TOKENS-1][base+k/2];
        Var soft=(z.v>0)?z+ln(1+ex(-z)):ln(1+ex(z));
        loss+=soft-((bits>>k)&1)*z;
    }return loss/width;
}
static int finite_number(double value){
    union {double d;unsigned long long u;} bits={.d=value};
    return ((bits.u>>52)&2047)!=2047;
}
extern "C" long vgi_stream_update(long kind,long bits,long unroll,long steps){
    stream_steps_completed=0;
    if(kind<0||kind>2||steps<0||steps>100||unroll<0||unroll>8)return -10;
    if(kind==1&&(V<=264||bits<0||bits>255))return -11;
    if(kind==2&&(V<=260||bits<0||bits>15))return -12;
    int labels=0;for(int i=0;i<TOKENS;i++)labels+=stream_labels[i]>=0;
    if(kind==0&&!labels)return -13;
    double first_loss=0;
    for(long step=0;step<(steps?steps:1);step++){
        prepare();forward((int)unroll);Var loss=stream_loss((int)kind,bits);
        if(overflowed)return -20;if(!finite_number(loss.v))return -21;
        if(step==0)first_loss=loss.v;
        if(!steps)break;
        backward(loss);double norm_squared=0;int index=0;
        for(const auto &p:parameters)for(int j=0;j<p.size;j++){
            double g=tape[p.var[j].id].grad;if(!finite_number(g))return -22;
            initial_gradient[index++]=g;norm_squared+=g*g;
        }
        if(!finite_number(norm_squared))return -23;
        stream_gradient_norm=raw_sq(norm_squared);
        double scale=(STREAM_CLIP>0&&stream_gradient_norm>STREAM_CLIP)?STREAM_CLIP/stream_gradient_norm:1;
        optimizer_apply(initial_gradient,scale);
        stream_steps_completed++;
    }
    stream_completed++;return (long)(first_loss*1000000);
}
extern "C" double vgi_stream_norm(){return stream_gradient_norm;}
static long stream_read(void *buffer,unsigned long size){
    unsigned long total=0;
    while(total<size){long n;unsigned char*position=(unsigned char*)buffer+total;
        __asm__ volatile("syscall":"=a"(n):"a"(0L),"D"(0L),"S"(position),"d"(size-total):"rcx","r11","memory");
        if(n<0)return -1;if(n==0)return total==0?0:-1;total+=(unsigned long)n;
    }return (long)total;
}
static unsigned int u16(const unsigned char*p){return (unsigned int)p[0]|((unsigned int)p[1]<<8);}
static void stream_log(long index,long loss,long kind){
    char line[160];int length=0;const char*prefix="vgi-stream ";
    while(*prefix)line[length++]=*prefix++;
    long fields[5]={index,kind,loss,used,stream_steps_completed};
    for(int k=0;k<5;k++){
        long n=fields[k];if(n<0){line[length++]='-';n= -n;}
        char digits[24];int count=0;do{digits[count++]=(char)('0'+n%10);n/=10;}while(n);
        while(count)line[length++]=digits[--count];line[length++]=k==4?'\n':' ';
    }
    long ignored;__asm__ volatile("syscall":"=a"(ignored):"a"(1L),"D"(2L),"S"(line),"d"((unsigned long)length):"rcx","r11","memory");
}
extern "C" long vgi_stream_train(){
    vgi_stream_reset();unsigned char header[12];
    if(stream_read(header,12)!=12)return -30;
    const unsigned char magic[8]={'V','G','I','S','1',0,0,0};
    for(int i=0;i<8;i++)if(header[i]!=magic[i])return -31;
    unsigned long frames=(unsigned long)u16(header+8)|((unsigned long)u16(header+10)<<16);
    for(unsigned long frame=0;frame<frames;frame++){
        stream_steps_completed=0;
        unsigned char bytes[8+4*TOKENS];
        if(stream_read(bytes,sizeof(bytes))!=(long)sizeof(bytes)){stream_log((long)frame,-32,-1);return -32;}
        int kind=(int)u16(bytes),unroll=(int)u16(bytes+2),steps=(int)u16(bytes+4),bits=(int)u16(bytes+6);
        for(int i=0;i<TOKENS;i++){
            long token=u16(bytes+8+2*i),label=(short)u16(bytes+8+2*TOKENS+2*i);
            if(vgi_stream_token(i,token,label)<0){stream_log((long)frame,-33,kind);return -33;}
        }
        long loss=vgi_stream_update(kind,bits,unroll,steps);stream_log((long)frame,loss,kind);
        if(loss<0)return loss;
    }
    unsigned char trailing;if(stream_read(&trailing,1)!=0)return -34;
    return stream_completed;
}
