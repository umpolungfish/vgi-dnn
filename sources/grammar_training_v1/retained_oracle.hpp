/* Operation-sensitive return within the retained learner. Query criteria enter;
 * candidate outcomes and evidence targets are computed by the executing gates.
 */
static long retained_oracle_trace[13],retained_oracle_returns[4];
extern "C" long vgi_retained_retrieve(long index,long readout);
static long retained_oracle_prediction(){
    long bits=0;for(int action=0;action<4;action++){
        bits|=(retained_first_t[261+action]>0)<<(2*action);
        bits|=(retained_first_f[261+action]>0)<<(2*action+1);
    }return bits;
}
static long retained_oracle_choice(){
    long choice=-1;double best=0;for(int action=0;action<4;action++)
        if(retained_first_t[261+action]>best){best=retained_first_t[261+action];choice=action;}
    return choice;
}
static long retained_oracle_validate(const unsigned short*query){
    if(query[0]!=256||query[7]!=257||query[1]<266||query[1]>281||query[2]<282||query[2]>289||
       query[3]<266||query[3]>281||(query[4]!=298&&(query[4]<266||query[4]>281))||
       query[5]<290||query[5]>293||query[6]<294||query[6]>297)return -61;
    return 0;
}
extern "C" long vgi_retained_oracle(unsigned long first,unsigned long second,long correction_steps,long readout){
    if(first>>36||second>>36||correction_steps<0||correction_steps>100)return -60;
    unsigned short query[8];for(int i=0;i<8;i++)query[i]=((i<4?first:second)>>(9*(i%4)))&511;
    if(retained_oracle_validate(query)<0)return -61;
    if(!parameters_initialized)vgi_retained_reset();
    for(int i=0;i<8;i++)retained_bank[i]=query[i];
    retained_prefix=8;retained_target=0;retained_active=1;
    for(int i=0;i<8;i++){
        long recovered=vgi_retained_retrieve(i,0);if(recovered<0)return recovered;
        query[i]=(unsigned short)recovered;
    }
    long seed=query[1]-266,work=query[2]-282,positive=query[3]-266;
    long negative=query[4]==298?-1:query[4]-266,view=query[6]==294?-1:query[6]==295?0:query[6]==296?2:3;
    retained_oracle_returns[0]=policy_gate(work,seed);
    retained_oracle_returns[1]=policy_gate(work,seed&5)|(seed&10);
    retained_oracle_returns[2]=policy_gate(work,seed&1)|(seed&2)|(seed&12);
    retained_oracle_returns[3]=seed;
    long exact=0;for(int action=0;action<4;action++){
        long arity=action==1?2:action==2?3:0,value=retained_oracle_returns[action];
        exact|=(value==positive&&(view<0||arity==view))<<(2*action);
        exact|=((negative>=0&&value==negative)||(view>=0&&arity!=view))<<(2*action+1);
    }
    for(int i=0;i<8;i++)retained_bank[i]=query[i];
    long before=vgi_retained_update(8,0,1,exact,0,1);if(before<0)return before;
    long prediction=retained_oracle_prediction(),choice=retained_oracle_choice();
    retained_oracle_trace[0]=exact;retained_oracle_trace[1]=prediction;
    retained_oracle_trace[3]=prediction^exact;retained_oracle_trace[5]=choice;
    retained_oracle_trace[7]=choice<0?seed:retained_oracle_returns[choice];
    retained_oracle_trace[11]=before;
    long applied=0;if(prediction!=exact&&correction_steps){
        long loss=vgi_retained_update(8,0,1,exact,correction_steps,1);if(loss<0)return loss;
        applied=correction_steps;
    }
    long after=vgi_retained_update(8,0,1,exact,0,1);if(after<0)return after;
    prediction=retained_oracle_prediction();choice=retained_oracle_choice();
    retained_oracle_trace[2]=prediction;retained_oracle_trace[4]=prediction^exact;
    retained_oracle_trace[6]=choice;retained_oracle_trace[8]=choice<0?seed:retained_oracle_returns[choice];
    retained_oracle_trace[9]=choice<0?0:(exact>>(2*choice))&1;
    retained_oracle_trace[10]=applied;retained_oracle_trace[12]=after;
    if(readout>=0&&readout<13)return retained_oracle_trace[readout];
    if(readout>=20&&readout<24)return retained_oracle_returns[readout-20];
    if(readout==30)return (choice+1)|(retained_oracle_trace[8]<<3)|(prediction<<7)|((prediction^exact)<<15);
    return -62;
}

/* Dimension-four SIC Gram analysis and signed dual synthesis. Masses have
 * denominator 5. Dual recovery is (20*mass_i-sum(mass))/80 exactly.
 */
extern "C" long vgi_retained_retrieve(long index,long readout){
    unsigned long length=retained_prefix+retained_target;
    if(!retained_active||index<0||(unsigned long)index>=length)return -63;
    unsigned long start=((unsigned long)index/16)*16;long coordinates[16],sum=0,masses[16],mass_sum=0;
    for(int i=0;i<16;i++){coordinates[i]=start+i<length?retained_bank[start+i]:0;sum+=coordinates[i];}
    for(int i=0;i<16;i++){masses[i]=4*coordinates[i]+sum;mass_sum+=masses[i];}
    long numerator=20*masses[index%16]-mass_sum;
    if(numerator%80)return -64;
    if(readout==0)return numerator/80;
    if(readout==1)return masses[index%16];
    if(readout==2)return 5;
    if(readout==3)return -mass_sum;
    if(readout==4)return numerator;
    if(readout==5)return 80;
    return -65;
}
extern "C" long vgi_retained_oracle_observe(long readout){
    if(readout>=0&&readout<13)return retained_oracle_trace[readout];
    if(readout>=20&&readout<24)return retained_oracle_returns[readout-20];
    return -62;
}

static void retained_oracle_log(long index){
    char line[512];int length=0;const char*prefix="vgi-oracle ";
    while(*prefix)line[length++]=*prefix++;
    for(int k=-1;k<13;k++){
        long n=k<0?index:retained_oracle_trace[k];if(n<0){line[length++]='-';n=-n;}
        char digits[24];int count=0;do{digits[count++]=(char)('0'+n%10);n/=10;}while(n);
        while(count)line[length++]=digits[--count];line[length++]=k==12?'\n':' ';
    }
    long ignored;__asm__ volatile("syscall":"=a"(ignored):"a"(1L),"D"(2L),"S"(line),"d"((unsigned long)length):"rcx","r11","memory");
}
static long retained_oracle_record(long steps,long record){
    unsigned long first=0,second=0,previous_records=retained_records;
    for(int i=0;i<4;i++){first|=((unsigned long)retained_bank[i])<<(9*i);second|=((unsigned long)retained_bank[i+4])<<(9*i);}
    long loss=vgi_retained_oracle(first,second,steps,12);if(loss<0)return loss;
    retained_records=previous_records+1;retained_kind=3;retained_bits=retained_oracle_trace[0];
    retained_committed=retained_oracle_trace[10];retained_requested=steps;retained_complete=1;
    retained_oracle_log(record);return loss;
}
