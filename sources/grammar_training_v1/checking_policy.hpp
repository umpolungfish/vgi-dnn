/* Optional checking as paired evidence. The query carries a candidate proposal;
 * inference obtains no reference target before deciding whether to check.
 * Tokens: BOS,C,seed,work,positive,negative,previous,view,action,proposal,0,SEP.
 */
static long checking_calls,checking_trace[8],checking_invocations;
static long retained_check_validate(const unsigned short*q){
    if(q[0]!=256||q[1]!=67||q[8]<290||q[8]>293||q[9]<266||q[9]>281||q[10]!=0||q[11]!=257)return -100;
    unsigned short original[8]={256,q[2],q[3],q[4],q[5],q[6],q[7],257};
    return retained_oracle_validate(original);
}
static long checking_reference(const unsigned short*q,long*carrier){
    checking_calls++;checking_invocations++;
    long seed=q[2]-266,work=q[3]-282,positive=q[4]-266,negative=q[5]==298?-1:q[5]-266;
    long view=q[7]==294?-1:q[7]==295?0:q[7]==296?2:3,action=q[8]-290;
    long value=action==0?policy_gate(work,seed):action==1?(policy_gate(work,seed&5)|(seed&10)):
               action==2?(policy_gate(work,seed&1)|(seed&2)|(seed&12)):seed;
    long arity=action==1?2:action==2?3:0;
    long support=value==positive&&(view<0||view==arity);
    long refute=(negative>=0&&value==negative)||(view>=0&&view!=arity);
    *carrier=value;
    // Evidence to check: wrong carrier or refuted candidate.
    // Evidence to retain without checking: exact carrier and supported candidate.
    return ((q[9]-266!=value)||refute)|((q[9]-266==value&&support)<<1);
}
static long checking_unpack(unsigned long a,unsigned long b,unsigned long c,unsigned short*q){
    if(a>>36||b>>36||c>>36)return -100;
    unsigned long words[3]={a,b,c};for(int i=0;i<12;i++)q[i]=(words[i/4]>>(9*(i%4)))&511;
    return retained_check_validate(q);
}
static long checking_prediction(){return (retained_first_t[265]>0)|((retained_first_f[265]>0)<<1);}
extern "C" long vgi_check_observe(long index){
    if(index>=0&&index<8)return checking_trace[index];if(index==8)return checking_calls;if(index==9)return checking_invocations;return -101;
}
extern "C" long vgi_check_reference(unsigned long a,unsigned long b,unsigned long c,long readout){
    unsigned short query[12];if(checking_unpack(a,b,c,query)<0)return -100;
    long carrier,bits=checking_reference(query,&carrier);
    if(readout==0)return bits;if(readout==1)return carrier;if(readout==2)return checking_calls;return -101;
}
extern "C" long vgi_check_execute(unsigned long a,unsigned long b,unsigned long c,long readout){
    unsigned short query[12];if(checking_unpack(a,b,c,query)<0)return -100;
    if(!parameters_initialized)vgi_retained_reset();
    for(int i=0;i<12;i++)retained_bank[i]=query[i];
    // Zero supervision bits are inert in this forward-only proposal.
    long loss=vgi_retained_update(12,0,3,0,0,1);if(loss<0)return loss;
    long bits=checking_prediction(),before=checking_calls,value=query[9]-266,reference=-1;
    if(bits&1)reference=checking_reference(query,&value);
    checking_trace[0]=bits;checking_trace[1]=bits&1;checking_trace[2]=query[9]-266;checking_trace[3]=value;
    checking_trace[4]=query[8]-290;checking_trace[5]=checking_calls-before;checking_trace[6]=reference;
    checking_trace[7]=bits==0||bits==3;
    if(readout==0)return bits;if(readout==1)return value;if(readout==2)return checking_calls-before;
    if(readout==3)return bits|(value<<2)|((bits&1)<<6);
    if(readout>=10&&readout<18)return checking_trace[readout-10];return -101;
}
static long retained_check_record(long steps,long record){
    if(retained_check_validate(retained_bank)<0)return -100;
    unsigned short query[12];for(int i=0;i<12;i++)query[i]=retained_bank[i];
    long carrier,exact=checking_reference(query,&carrier);unsigned long previous=retained_records;
    long before_loss=vgi_retained_update(12,0,3,exact,0,1);if(before_loss<0)return before_loss;
    long before=checking_prediction(),applied=0;
    if(before!=exact&&steps){long loss=vgi_retained_update(12,0,3,exact,steps,1);if(loss<0)return loss;applied=steps;}
    long after_loss=vgi_retained_update(12,0,3,exact,0,1);if(after_loss<0)return after_loss;
    long after=checking_prediction();
    long fields[13]={exact,before,after,before^exact,after^exact,-1,-1,query[9]-266,carrier,(after&1)!=0,applied,before_loss,after_loss};
    for(int i=0;i<13;i++)retained_oracle_trace[i]=fields[i];retained_oracle_log(record);
    retained_records=previous+1;retained_kind=5;retained_bits=exact;retained_committed=applied;retained_requested=steps;retained_complete=1;
    return after_loss;
}
