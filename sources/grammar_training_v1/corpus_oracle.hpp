extern "C" long vgi_corpus_find(unsigned long a,unsigned long b,unsigned long c,unsigned long d){
    for(unsigned long i=0;i<corpus_count;i++){
        const auto&r=corpus_records[i];if(r.key[0]==a&&r.key[1]==b&&r.key[2]==c&&r.key[3]==d)return (long)i;
    }return -70;
}
extern "C" long vgi_corpus_metadata(long record,long offset){
    if(record<0||(unsigned long)record>=corpus_count)return -70;
    const auto&r=corpus_records[record];
    if(offset==-1)return (long)r.metadata_size;
    if(offset<0||(unsigned long)offset>=r.metadata_size)return -71;
    return r.metadata[offset];
}
extern "C" long vgi_corpus_retrieve(long record,long index,long readout){
    if(record<0||(unsigned long)record>=corpus_count)return -70;
    const auto&r=corpus_records[record];
    if(index==-1)return (long)(r.prefix+r.target);
    if(index<0||(unsigned long)index>=r.prefix+r.target)return -71;
    unsigned long start=((unsigned long)index/16)*16;long total=0;
    for(int k=0;k<16;k++)total+=r.mass[start+k];
    long mass=r.mass[index],numerator=20*mass-total;
    if(numerator%80)return -72;
    if(readout==0)return numerator/80;
    if(readout==1)return mass;
    if(readout==2)return -total;
    if(readout==3)return numerator;
    return -73;
}
extern "C" long vgi_corpus_select(long record){
    if(record<0||(unsigned long)record>=corpus_count)return -70;
    if(!parameters_initialized)vgi_retained_reset();
    const auto&r=corpus_records[record];
    for(unsigned long i=0;i<r.prefix+r.target;i++){
        long token=vgi_corpus_retrieve(record,(long)i,0);
        if(token<0||token>=V)return -72;
        retained_pending[i]=(unsigned short)token;
    }
    unsigned short*old=retained_bank;retained_bank=retained_pending;retained_pending=old;
    retained_prefix=r.prefix;retained_target=r.target;retained_kind=r.kind;retained_bits=r.bits;
    retained_active=1;retained_position=retained_chunks=retained_committed=retained_requested=0;
    retained_complete=retained_settlement=0;
    for(int c=0;c<D;c++)retained_memory_t[c]=retained_memory_f[c]=.05;
    return (long)(r.prefix+r.target);
}
extern "C" long vgi_corpus_learn(long record,long steps,long readout){
    if(steps<0||steps>100)return -74;
    long selected=vgi_corpus_select(record);if(selected<0)return selected;
    const auto&r=corpus_records[record];
    if(r.kind==3&&retained_oracle_validate(retained_bank)<0)return -76;
    if(r.kind==4&&retained_graph_local_validate(retained_bank)<0)return -76;
    if(r.kind==5&&retained_check_validate(retained_bank)<0)return -76;
    long loss=r.kind==3?retained_oracle_record(steps,record):r.kind==4?retained_graph_local_record(steps,record):r.kind==5?retained_check_record(steps,record):vgi_retained_update(r.prefix,r.target,r.kind,r.bits,steps,r.window);if(loss<0)return loss;
    if(readout==0)return loss;
    if(readout==1)return parameter_digest();
    if(readout==2){if(vgi_write_current()<0||retained_write_state()<0)return -75;return loss;}
    return -73;
}
extern "C" long vgi_corpus_train(){
    vgi_retained_reset();
    for(unsigned long i=0;i<corpus_count;i++){
        long loss=vgi_corpus_learn((long)i,(long)corpus_records[i].steps,0);stream_steps_completed=(long)retained_committed;
        stream_log((long)i,loss,corpus_records[i].kind);if(loss<0)return loss;
#if RETAINED_SNAPSHOT_RECORDS > 0
        if(retained_records%RETAINED_SNAPSHOT_RECORDS==0)
            if(vgi_write_current()<0||retained_write_state()<0)return -75;
#endif
    }return (long)corpus_count;
}
