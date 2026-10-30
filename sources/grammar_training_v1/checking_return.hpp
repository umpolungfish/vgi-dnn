/* A checking continuation returns the model, optimizer, retained source/memory,
 * proposal, checking evidence, observation, and its next re-entry together.
 */
static unsigned short checking_query[12];
static unsigned long checking_state_active,checking_phase,checking_sequence;
static double checking_initial_t[D],checking_initial_f[D];
static unsigned long checking_model_fingerprint(){
    unsigned long hash=14695981039346656037UL;
    unsigned long config[]={V,D,H,FF,LAYERS,MAXLEN,TOKENS,BLEND,CAUSAL,OPTIMIZER_ADAM};
    for(unsigned long value:config)hash=retained_mix(hash,value);
    hash=retained_mix(hash,retained_double_word(LEARNING_RATE));hash=retained_mix(hash,retained_double_word(STREAM_CLIP));
    for(const auto&p:parameters)for(int j=0;j<p.size;j++)hash=retained_mix(hash,retained_double_word(p.raw[j]));
    for(const auto&p:buffers)for(int j=0;j<p.size;j++)hash=retained_mix(hash,retained_double_word(p.raw[j]));
    for(int i=0;i<PARAMETERS;i++){hash=retained_mix(hash,retained_double_word(optimizer_m[i]));hash=retained_mix(hash,retained_double_word(optimizer_v[i]));}
    hash=retained_mix(hash,(unsigned long)optimizer_clock);
    for(int i=0;i<2;i++)hash=retained_mix(hash,retained_double_word(optimizer_powers[i]));return hash;
}
static int checking_bound(){
    if(!checking_state_active||!retained_active||retained_prefix!=12||retained_target)return 0;
    for(int i=0;i<12;i++)if(retained_bank[i]!=checking_query[i])return 0;return 1;
}
static long checking_result(long readout){
    if(readout==0)return checking_trace[0];if(readout==1)return checking_trace[3];if(readout==2)return checking_trace[5];
    if(readout==3)return checking_trace[0]|(checking_trace[3]<<2)|(checking_trace[5]<<6);
    if(readout>=10&&readout<18)return checking_trace[readout-10];
    if(readout==18)return checking_invocations;return -110;
}
extern "C" long vgi_check_enter(unsigned long a,unsigned long b,unsigned long c,long readout){
    if(readout<0||(readout>3&&(readout<10||readout>=18)))return -110;
    unsigned short query[12];if(checking_unpack(a,b,c,query)<0)return -100;
    for(int i=0;i<D;i++){
        checking_initial_t[i]=retained_context_active?retained_context_t[i]:.05;
        checking_initial_f[i]=retained_context_active?retained_context_f[i]:.05;
    }
    long result=vgi_check_execute(a,b,c,readout);if(result<0)return result;
    for(int i=0;i<12;i++)checking_query[i]=query[i];
    checking_state_active=1;checking_phase=1;checking_sequence++;return result;
}
extern "C" long vgi_check_train_return(long steps){
    if(!checking_bound()||steps<0||steps>100)return -111;
    retained_context_begin(checking_initial_t,checking_initial_f);
    long loss=retained_check_record(steps,(long)checking_sequence-1);retained_context_end();if(loss<0)return loss;
    checking_phase=2;return loss;
}
extern "C" long vgi_check_continue(long readout){
    if(!checking_bound())return -111;
    unsigned short query[12];for(int i=0;i<12;i++)query[i]=checking_query[i];
    query[9]=(unsigned short)(266+checking_trace[3]);query[6]=query[8];
    unsigned long words[3]={};for(int i=0;i<12;i++)words[i/4]|=((unsigned long)query[i])<<(9*(i%4));
    retained_context_begin(retained_memory_t,retained_memory_f);
    long result=vgi_check_enter(words[0],words[1],words[2],readout);retained_context_end();return result;
}
static unsigned long checking_checksum(const unsigned long*header,const unsigned short*query,const long*trace,const long*teacher,const double*t,const double*f){
    unsigned long hash=14695981039346656037UL;
    for(int i=0;i<9;i++)hash=retained_mix(hash,header[i]);
    for(int i=0;i<12;i++)hash=retained_mix(hash,query[i]);
    for(int i=0;i<8;i++)hash=retained_mix(hash,(unsigned long)trace[i]);
    for(int i=0;i<13;i++)hash=retained_mix(hash,(unsigned long)teacher[i]);
    for(int i=0;i<D;i++)hash=retained_mix(hash,retained_double_word(t[i]));
    for(int i=0;i<D;i++)hash=retained_mix(hash,retained_double_word(f[i]));return hash;
}
static long checking_write_state(){
    const unsigned char magic[8]={'C','G','S','T','A','T','E','2'};
    unsigned long header[10]={checking_state_active,checking_phase,checking_sequence,(unsigned long)checking_calls,
        checking_model_fingerprint(),12,8,13,retained_state_fingerprint(),0};
    header[9]=checking_checksum(header,checking_query,checking_trace,retained_oracle_trace,checking_initial_t,checking_initial_f);
    if(checkpoint_write(magic,8)<0||checkpoint_write(header,sizeof(header))<0||checkpoint_write(checking_query,sizeof(checking_query))<0||
       checkpoint_write(checking_trace,sizeof(checking_trace))<0||checkpoint_write(retained_oracle_trace,sizeof(retained_oracle_trace))<0||
       checkpoint_write(checking_initial_t,sizeof(checking_initial_t))<0||checkpoint_write(checking_initial_f,sizeof(checking_initial_f))<0)return -112;
    return 0;
}
extern "C" long vgi_check_return(){
    if(!checking_bound())return -111;
    if(vgi_write_current()<0||retained_write_state()<0||checking_write_state()<0)return -112;return 0;
}
static long checking_restore(){
    unsigned char magic[8];unsigned long header[10];unsigned short query[12];long trace[8],teacher[13];double initial[2][D];
    if(stream_read(magic,8)!=8||stream_read(header,sizeof(header))!=(long)sizeof(header)||
       stream_read(query,sizeof(query))!=(long)sizeof(query)||stream_read(trace,sizeof(trace))!=(long)sizeof(trace)||
       stream_read(teacher,sizeof(teacher))!=(long)sizeof(teacher)||stream_read(initial,sizeof(initial))!=(long)sizeof(initial))return -113;
    const unsigned char wanted[8]={'C','G','S','T','A','T','E','2'};
    for(int i=0;i<8;i++)if(magic[i]!=wanted[i])return -113;
    if(header[0]!=1||header[1]<1||header[1]>2||header[2]<1||header[2]>0x7fffffffffffffffUL||
       header[3]>0x7fffffffffffffffUL||header[5]!=12||header[6]!=8||header[7]!=13)return -113;
    if(header[4]!=checking_model_fingerprint()||header[9]!=checking_checksum(header,query,trace,teacher,initial[0],initial[1]))return -114;
    for(int j=0;j<2;j++)for(int i=0;i<D;i++)if(!finite_number(initial[j][i]))return -113;
    if(retained_check_validate(query)<0||trace[0]<0||trace[0]>3||trace[1]!=(trace[0]&1)||trace[2]!=query[9]-266||
       trace[3]<0||trace[3]>15||trace[4]!=query[8]-290||trace[5]!=trace[1]||header[3]<(unsigned long)trace[5]||
       trace[6]<-1||trace[6]>3||((trace[5]==0)!=(trace[6]==-1))||trace[7]!=(trace[0]==0||trace[0]==3)||
       (!trace[5]&&trace[3]!=trace[2]))return -113;
    if(retained_read_state(query,12,1,header[8],1)<0)return -115;
    for(int i=0;i<12;i++)checking_query[i]=query[i];
    for(int i=0;i<8;i++)checking_trace[i]=trace[i];
    for(int i=0;i<13;i++)retained_oracle_trace[i]=teacher[i];
    for(int i=0;i<D;i++){checking_initial_t[i]=initial[0][i];checking_initial_f[i]=initial[1][i];}
    checking_state_active=header[0];checking_phase=header[1];checking_sequence=header[2];checking_calls=(long)header[3];return 0;
}
/* CGRUN1\0\0 + four u64: command, extra rounds, learning steps, readout.
 * command 0: new 12-u16 query; 1: restore; 2: restore then re-enter.
 * Restores carry CGSTATE2 followed by VGIRM2; model/Adam state is baked from
 * the returned safetensors and matched before accepting the continuation.
 */
extern "C" long vgi_check_stream(){
    if(!parameters_initialized)vgi_retained_reset();
    unsigned char magic[8];unsigned long options[4];
    if(stream_read(magic,8)!=8||stream_read(options,sizeof(options))!=(long)sizeof(options))return -116;
    const unsigned char wanted[8]={'C','G','R','U','N','1',0,0};
    for(int i=0;i<8;i++)if(magic[i]!=wanted[i])return -116;
    unsigned long command=options[0],rounds=options[1],steps=options[2],readout=options[3];
    if(command>2||rounds>100||steps>100||(readout>3&&(readout<10||readout>=19))||
       (command==1&&(rounds||steps))||(command==2&&!rounds)||(!rounds&&steps))return -116;
    if(command==0){
        unsigned short query[12];unsigned char extra;
        if(stream_read(query,sizeof(query))!=(long)sizeof(query)||stream_read(&extra,1)!=0||retained_check_validate(query)<0)return -116;
        unsigned long words[3]={};for(int i=0;i<12;i++)words[i/4]|=((unsigned long)query[i])<<(9*(i%4));
        long result=vgi_check_enter(words[0],words[1],words[2],3);if(result<0)return result;
    }else{long result=checking_restore();if(result<0)return result;}
    for(unsigned long round=0;round<rounds;round++){
        if(steps){long result=vgi_check_train_return((long)steps);if(result<0)return result;}
        long result=vgi_check_continue(3);if(result<0)return result;
    }return checking_result((long)readout);
}
