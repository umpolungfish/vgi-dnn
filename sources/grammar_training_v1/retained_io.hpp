/* Shared retained-state admission for graph and checking continuations.
 * Decode into pending storage; validate the entire request before committing.
 */
static unsigned long retained_mix(unsigned long hash,unsigned long value){return (hash^value)*1099511628211UL;}
static unsigned long retained_double_word(double value){union{double d;unsigned long u;}bits={.d=value};return bits.u;}
static void retained_state_header(unsigned long*header){
    unsigned long values[15]={retained_prefix,retained_target,retained_position,retained_chunks,retained_records,(unsigned long)D,
        retained_committed,retained_requested,retained_kind,retained_bits,retained_window,retained_complete,
        retained_peak,retained_recomputed,retained_settlement};
    for(int i=0;i<15;i++)header[i]=values[i];
}
static unsigned long retained_fingerprint_parts(const unsigned long*header,const double*const*memory,const unsigned short*bank){
    unsigned long hash=14695981039346656037UL;
    for(int i=0;i<15;i++)hash=retained_mix(hash,header[i]);
    for(int j=0;j<4;j++)for(int i=0;i<D;i++)hash=retained_mix(hash,retained_double_word(memory[j][i]));
    for(unsigned long i=0;i<header[0]+header[1];i++)hash=retained_mix(hash,bank[i]);return hash;
}
static unsigned long retained_state_fingerprint(){
    unsigned long header[15];retained_state_header(header);
    const double*memory[4]={retained_memory_t,retained_memory_f,
        retained_settlement?retained_before_t:retained_memory_t,retained_settlement?retained_before_f:retained_memory_f};
    return retained_fingerprint_parts(header,memory,retained_bank);
}
static long retained_read_state(const unsigned short*expected=0,unsigned long expected_size=0,int require_eof=0,
                                unsigned long expected_fingerprint=0,int verify_fingerprint=0){
    unsigned char magic[8];unsigned long header[15];double memory[4][D];
    if(stream_read(magic,8)!=8||stream_read(header,sizeof(header))!=(long)sizeof(header))return -94;
    const unsigned char wanted[8]={'V','G','I','R','M','2',0,0};
    for(int i=0;i<8;i++)if(magic[i]!=wanted[i])return -94;
    if(header[5]!=(unsigned long)D||header[0]>RETAINED_SOURCE_CAPACITY||header[1]>RETAINED_SOURCE_CAPACITY-header[0])return -94;
    unsigned long length=header[0]+header[1];
    if(header[2]>length||header[3]>(length+TOKENS-2)/(TOKENS-1)||header[6]>header[7]||header[7]>100||
       header[8]>5||header[9]>255||header[10]>RETAINED_SOURCE_CAPACITY||header[11]>1||header[14]>3)return -94;
    if(expected&&(header[0]!=expected_size||header[1]))return -94;
    for(int j=0;j<4;j++)if(stream_read(memory[j],8*D)!=8*D)return -94;
    if(stream_read(retained_pending,2*length)!=(long)(2*length))return -94;
    for(unsigned long i=0;i<length;i++)if(retained_pending[i]>=V||(expected&&retained_pending[i]!=expected[i]))return -94;
    for(int j=0;j<4;j++)for(int i=0;i<D;i++)if(!finite_number(memory[j][i]))return -94;
    const double*views[4]={memory[0],memory[1],memory[2],memory[3]};
    if(verify_fingerprint&&retained_fingerprint_parts(header,views,retained_pending)!=expected_fingerprint)return -94;
    if(require_eof){unsigned char extra;if(stream_read(&extra,1)!=0)return -94;}
    unsigned short*old=retained_bank;retained_bank=retained_pending;retained_pending=old;
    for(int i=0;i<D;i++){
        retained_memory_t[i]=memory[0][i];retained_memory_f[i]=memory[1][i];
        retained_before_t[i]=memory[2][i];retained_before_f[i]=memory[3][i];
    }
    retained_prefix=header[0];retained_target=header[1];retained_position=header[2];retained_chunks=header[3];
    retained_records=header[4];retained_committed=header[6];retained_requested=header[7];retained_kind=header[8];
    retained_bits=header[9];retained_window=header[10];retained_complete=header[11];retained_peak=header[12];
    retained_recomputed=header[13];retained_settlement=header[14];retained_active=length>0;return 0;
}
