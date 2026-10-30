/* Whole-source recurrent conditioning and exact checkpointed reverse accumulation.
 * Checkpoint boundaries change storage/recomputation, not the gradient domain.
 */
#ifndef RETAINED_SOURCE_CAPACITY
#define RETAINED_SOURCE_CAPACITY 1048576
#endif
static unsigned short retained_a[RETAINED_SOURCE_CAPACITY],retained_b[RETAINED_SOURCE_CAPACITY];
static unsigned short *retained_bank=retained_a,*retained_pending=retained_b;
static double retained_gradient[PARAMETERS],retained_memory_t[D],retained_memory_f[D];
static double retained_first_t[V],retained_first_f[V];
static double source_checkpoints[RETAINED_SOURCE_CAPACITY/(TOKENS-1)+2][2][D];
static unsigned long retained_prefix,retained_target,retained_position,retained_chunks,retained_records;
static unsigned long retained_committed,retained_requested,retained_kind,retained_bits,retained_window,retained_complete;
static unsigned long retained_peak,retained_recomputed;
static int retained_active,retained_capture;
static unsigned long retained_settlement;
static double retained_before_t[D],retained_before_f[D];
static int retained_context_active;
static double retained_context_t[D],retained_context_f[D];
static void retained_context_begin(const double*t,const double*f){
    for(int c=0;c<D;c++){retained_context_t[c]=t[c];retained_context_f[c]=f[c];}retained_context_active=1;
}
static void retained_context_end(){retained_context_active=0;}
static void retained_prepare(int tracking){
    if(tracking){prepare();return;}
    used=0;overflowed=0;
    for(const auto &p:parameters)for(int j=0;j<p.size;j++)p.var[j]=Var(p.raw[j]);
    for(const auto &p:buffers)for(int j=0;j<p.size;j++)p.var[j]=Var(p.raw[j]);
}
extern "C" void vgi_retained_reset(){
    reset_parameters();retained_prefix=retained_target=retained_position=retained_chunks=retained_records=0;
    retained_active=0;for(int c=0;c<D;c++)retained_memory_t[c]=retained_memory_f[c]=.05;
    retained_committed=retained_requested=retained_kind=retained_bits=retained_window=retained_complete=0;
    retained_peak=retained_recomputed=0;
    retained_settlement=0;
    retained_context_active=0;
}
extern "C" long vgi_retained_token(long index,long value){
    if(index<0||index>=RETAINED_SOURCE_CAPACITY||value<0||value>=V)return -1;
    retained_bank[index]=(unsigned short)value;return 0;
}
static void retained_output(Evidence *x,int row,Var *support,Var *refute,int both){
    Var sum=0,t[D],f[D];
    for(int c=0;c<D;c++){t[c]=lg(x->t[row][c]);f[c]=lg(x->f[row][c]);sum+=t[c]*t[c]+f[c]*f[c];}
    Var inv=1/sq(sum/(2*D)+1e-8);
    for(int c=0;c<D;c++){t[c]=lg(sig(final_norm[c]*t[c]*inv));f[c]=lg(sig(final_norm[c]*f[c]*inv));}
    if(both)linear(headA,headB,headb,t,f,D,V,support,refute);
    else for(int output=0;output<V;output++){
        Var value=headb[output];
        for(int c=0;c<D;c++)value+=headA[output*D+c]*t[c]+headB[output*D+c]*f[c];
        support[output]=value;
    }
}
static Var retained_chunk(unsigned long offset,unsigned long total,long kind,long bits,Var *mt,Var *mf,long *labels){
    int count=(total-offset<TOKENS-1)?(int)(total-offset):TOKENS-1;
    Evidence x,carried;Var loss=0;
    for(int c=0;c<D;c++){x.t[0][c]=mt[c];x.f[0][c]=mf[c];}
    for(int i=1;i<TOKENS;i++)for(int c=0;c<D;c++){
        int token=i<=count?retained_bank[offset+i-1]:0;
        unsigned long p=(offset+i-1)%MAXLEN;
        x.t[i][c]=sig(embedding_t[token*D+c]+.5*position[p*D+c]);
        x.f[i][c]=sig(embedding_f[token*D+c]+.5*position[((p+7)%MAXLEN)*D+c]);
    }
    for(int layer=0;layer<LAYERS;layer++)block(&x,layer);
    for(int i=0;i<TOKENS;i++)for(int c=0;c<D;c++){carried.t[i][c]=mt[c];carried.f[i][c]=mf[c];}
    residual(&carried,&x,blocks[LAYERS-1].gate1);
    for(int i=1;i<=count;i++){
        int supervised=kind==0?(offset+i>=retained_prefix):(offset+i==total);
        if(!supervised)continue;
        Var support[V],refute[V];retained_output(&carried,i,support,refute,kind!=0||retained_capture);
        if(retained_capture){for(int c=0;c<V;c++){retained_first_t[c]=support[c].v;retained_first_f[c]=refute[c].v;}retained_capture=0;}
        if(kind==0){
            Var maximum=support[0];for(int c=1;c<V;c++)if(support[c]>maximum)maximum=support[c];
            Var mass=0;for(int c=0;c<V;c++)mass+=ex(support[c]-maximum);
            loss+=ln(mass)+maximum-support[retained_bank[offset+i]];(*labels)++;
        }else{
            int width=kind==1?8:kind==3?2:4,base=kind==1?261:kind==3?265:259;
            for(int k=0;k<width;k++){
                Var z=(k%2)?refute[base+k/2]:support[base+k/2];
                Var soft=z.v>0?z+ln(1+ex(-z)):ln(1+ex(z));loss+=soft-((bits>>k)&1)*z;
            }(*labels)+=width;
        }
    }
    for(int c=0;c<D;c++){mt[c]=carried.t[count][c];mf[c]=carried.f[count][c];}
    if((unsigned long)used>retained_peak)retained_peak=(unsigned long)used;
    return loss;
}
extern "C" long vgi_retained_update(long prefix,long target,long kind,long bits,long steps,long window){
    if(prefix<1||target<0||prefix>RETAINED_SOURCE_CAPACITY||target>RETAINED_SOURCE_CAPACITY||
       prefix+target>RETAINED_SOURCE_CAPACITY||steps<0||steps>100||window<0||window>RETAINED_SOURCE_CAPACITY)return -40;
    if(kind<0||kind>3||(kind==0&&!target)||(kind!=0&&target))return -41;
    if((kind==1&&(V<=264||bits<0||bits>255))||(kind==2&&(V<=260||bits<0||bits>15)))return -42;
    if(kind==3&&(V<=265||bits<0||bits>3))return -42;
    retained_prefix=(unsigned long)prefix;retained_target=(unsigned long)target;retained_active=1;
    retained_committed=0;retained_requested=(unsigned long)steps;retained_kind=(unsigned long)kind;
    retained_bits=(unsigned long)bits;retained_window=(unsigned long)window;retained_complete=0;
    retained_settlement=0;
    double first_loss=0;
    for(long step=0;step<(steps?steps:1);step++){
        for(int i=0;i<PARAMETERS;i++)retained_gradient[i]=0;
        unsigned long total=(unsigned long)(kind==0?prefix+target-1:prefix);
        unsigned long stride=TOKENS-1,chunk_count=(total+stride-1)/stride;
        unsigned long window_chunks=window?(unsigned long)window:chunk_count;
        unsigned long width=window_chunks*stride,windows=(total+width-1)/width;
        Var mt[D],mf[D];for(int c=0;c<D;c++){
            mt[c]=Var(retained_context_active?retained_context_t[c]:.05);
            mf[c]=Var(retained_context_active?retained_context_f[c]:.05);
        }
        retained_position=retained_chunks=retained_peak=retained_recomputed=0;
        retained_prepare(0);retained_capture=1;long labels=0;double loss_sum=0;
        for(unsigned long w=0;w<windows;w++){
            for(int c=0;c<D;c++){source_checkpoints[w][0][c]=mt[c].v;source_checkpoints[w][1][c]=mf[c].v;}
            unsigned long end=(w+1)*width;if(end>total)end=total;
            for(unsigned long offset=w*width;offset<end;offset+=stride){
                Var loss=retained_chunk(offset,total,kind,bits,mt,mf,&labels);
                if(!finite_number(loss.v))return -44;loss_sum+=loss.v;
                retained_position=offset+((total-offset<stride)?total-offset:stride);retained_chunks++;
                for(int c=0;c<D;c++){retained_memory_t[c]=mt[c].v;retained_memory_f[c]=mf[c].v;}
            }
        }
        if(!labels)return -45;if(step==0)first_loss=loss_sum/labels;
        if(steps){
            double incoming_t[D],incoming_f[D];for(int c=0;c<D;c++)incoming_t[c]=incoming_f[c]=0;
            for(long w=(long)windows-1;w>=0;w--){
                prepare();int ti[D],fi[D];
                for(int c=0;c<D;c++){
                    mt[c]=leaf(source_checkpoints[w][0][c]);mf[c]=leaf(source_checkpoints[w][1][c]);
                    ti[c]=mt[c].id;fi[c]=mf[c].id;
                }
                Var objective=0;long local_labels=0;
                unsigned long end=((unsigned long)w+1)*width;if(end>total)end=total;
                for(unsigned long offset=(unsigned long)w*width;offset<end;offset+=stride){
                    objective+=retained_chunk(offset,total,kind,bits,mt,mf,&local_labels);retained_recomputed++;
                    if(overflowed)return -43;
                }
                for(int c=0;c<D;c++)objective+=mt[c]*incoming_t[c]+mf[c]*incoming_f[c];
                if(overflowed||!finite_number(objective.v))return -46;
                backward(objective);int index=0;
                for(const auto &p:parameters)for(int j=0;j<p.size;j++)retained_gradient[index++]+=tape[p.var[j].id].grad;
                for(int c=0;c<D;c++){incoming_t[c]=tape[ti[c]].grad;incoming_f[c]=tape[fi[c]].grad;}
            }
            double norm2=0;
            for(int i=0;i<PARAMETERS;i++){
                retained_gradient[i]/=labels;if(!finite_number(retained_gradient[i]))return -47;
                initial_gradient[i]=retained_gradient[i];norm2+=retained_gradient[i]*retained_gradient[i];
            }
            if(!finite_number(norm2))return -48;stream_gradient_norm=raw_sq(norm2);
            double scale=(STREAM_CLIP>0&&stream_gradient_norm>STREAM_CLIP)?STREAM_CLIP/stream_gradient_norm:1;
            optimizer_apply(retained_gradient,scale);
            retained_committed++;
        }
    }
    retained_records++;retained_complete=1;return (long)(first_loss*1000000);
}
extern "C" double vgi_retained_memory(long channel,long index){return channel?retained_memory_f[index]:retained_memory_t[index];}
extern "C" double vgi_retained_first(long channel,long index){return channel?retained_first_f[index]:retained_first_t[index];}
extern "C" unsigned long vgi_retained_nodes(){return retained_peak;}
extern "C" long vgi_retained_settle(long commit){
    if(!retained_active)return -1;
    retained_prepare(0);unsigned char history[32][D],state[D];
    for(int c=0;c<D;c++){
        retained_before_t[c]=retained_memory_t[c];retained_before_f[c]=retained_memory_f[c];
        state[c]=(retained_memory_t[c]>.5)|((retained_memory_f[c]>.5)<<1);
    }
    for(int step=0;step<32;step++){
        for(int previous=0;previous<step;previous++){
            int equal=1;for(int c=0;c<D;c++)if(history[previous][c]!=state[c])equal=0;
            if(equal){
                long status=step-previous==1?1:2;retained_settlement=(unsigned long)status+1;
                if(commit)for(int c=0;c<D;c++){
                    int joined=0;for(int j=previous;j<step;j++)joined|=history[j][c];
                    retained_memory_t[c]=(joined&1)?.95:.05;retained_memory_f[c]=(joined&2)?.95:.05;
                }return status;
            }
        }
        Evidence x;
        for(int c=0;c<D;c++){
            history[step][c]=state[c];
            for(int i=0;i<TOKENS;i++){x.t[i][c]=Var((state[c]&1)?.95:.05);x.f[i][c]=Var((state[c]&2)?.95:.05);}
        }
        block(&x,LAYERS-1);
        for(int c=0;c<D;c++)state[c]=(x.t[0][c].v>.5)|((x.f[0][c].v>.5)<<1);
    }
    retained_settlement=1;return 0;
}
/* VGIR2, count; record: kind,checkpoint-chunks,steps,bits (u16), P,Q (u32), tokens. */
static long retained_write_state();
static void retained_state_header(unsigned long*header);
static long retained_oracle_validate(const unsigned short*query);
static long retained_oracle_record(long steps,long record);
static long retained_graph_local_validate(const unsigned short*query);
static long retained_graph_local_record(long steps,long record);
static long retained_check_validate(const unsigned short*query);
static long retained_check_record(long steps,long record);
extern "C" long vgi_retained_train(){
    vgi_retained_reset();unsigned char header[12];if(stream_read(header,12)!=12)return -50;
    const unsigned char magic[8]={'V','G','I','R','2',0,0,0};
    for(int i=0;i<8;i++)if(header[i]!=magic[i])return -51;
    unsigned long records=(unsigned long)u16(header+8)|((unsigned long)u16(header+10)<<16);
    for(unsigned long record=0;record<records;record++){
        unsigned char h[16];if(stream_read(h,16)!=16)return -52;
        long kind=u16(h),window=u16(h+2),steps=u16(h+4),bits=u16(h+6);
        unsigned long prefix=(unsigned long)u16(h+8)|((unsigned long)u16(h+10)<<16);
        unsigned long target=(unsigned long)u16(h+12)|((unsigned long)u16(h+14)<<16);
        if(!prefix||kind>5||steps>100||(kind==0&&!target)||(kind!=0&&target))return -57;
        if((kind==1&&(V<=264||bits>255))||(kind==2&&(V<=260||bits>15)))return -58;
        if((kind==3||kind==4)&&(prefix!=8||target||bits||V<=264))return -58;
        if(kind==5&&(prefix!=12||target||bits||V<=265))return -58;
        if(prefix+target>RETAINED_SOURCE_CAPACITY)return -53;
        if(stream_read(retained_pending,2*(prefix+target))!=(long)(2*(prefix+target)))return -54;
        for(unsigned long i=0;i<prefix+target;i++)if(retained_pending[i]>=V)return -55;
        if(kind==3&&retained_oracle_validate(retained_pending)<0)return -58;
        if(kind==4&&retained_graph_local_validate(retained_pending)<0)return -58;
        if(kind==5&&retained_check_validate(retained_pending)<0)return -58;
        unsigned short*old=retained_bank;retained_bank=retained_pending;retained_pending=old;
        long loss=kind==3?retained_oracle_record(steps,(long)record):kind==4?retained_graph_local_record(steps,(long)record):kind==5?retained_check_record(steps,(long)record):vgi_retained_update((long)prefix,(long)target,kind,bits,steps,window);
        stream_steps_completed=(long)retained_committed;stream_log((long)record,loss,kind);if(loss<0)return loss;
#if RETAINED_SNAPSHOT_RECORDS > 0
        if(retained_records%RETAINED_SNAPSHOT_RECORDS==0){
            if(vgi_write_current()<0||retained_write_state()<0)return -59;
        }
#endif
    }
    unsigned char trailing;if(stream_read(&trailing,1)!=0)return -56;return (long)retained_records;
}
static long retained_write_state(){
    const unsigned char magic[8]={'V','G','I','R','M','2',0,0};
    unsigned long header[15];retained_state_header(header);
    long total=0,n;
    n=checkpoint_write(magic,8);if(n<0)return -1;total+=n;
    n=checkpoint_write(header,sizeof(header));if(n<0)return -1;total+=n;
    n=checkpoint_write(retained_memory_t,8*D);if(n<0)return -1;total+=n;
    n=checkpoint_write(retained_memory_f,8*D);if(n<0)return -1;total+=n;
    n=checkpoint_write(retained_settlement?retained_before_t:retained_memory_t,8*D);if(n<0)return -1;total+=n;
    n=checkpoint_write(retained_settlement?retained_before_f:retained_memory_f,8*D);if(n<0)return -1;total+=n;
    unsigned long length=retained_active?retained_prefix+retained_target:0;
    n=checkpoint_write(retained_bank,2*length);if(n<0)return -1;total+=n;return total;
}
